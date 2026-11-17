"""Offline-screened position/orientation graph; no online IK or collision search."""
import hashlib
import json
import math
from pathlib import Path
import torch


class FeasibleWorkspaceSampler:
    def __init__(self, filename, root, num_envs, device, center, home, induced_pitch):
        self.bank = json.loads(Path(filename).read_text())
        b = self.bank
        if b['schema_version'] != 1:
            raise ValueError('Unsupported feasible workspace bank schema')
        for relative, digest in b['asset_sha256'].items():
            asset = Path(root) / relative
            if hashlib.sha256(asset.read_bytes()).hexdigest() != digest:
                raise ValueError('Workspace bank is stale for asset: ' + str(asset))
        if any(abs(a-c)>1e-8 for a,c in zip(b['center'], center)):
            raise ValueError('Workspace bank virtual center differs from configuration')
        if any(abs(a-c)>1e-8 for a,c in zip(b['home_sphere'], home)):
            raise ValueError('Workspace bank reset goal differs from configuration')
        if abs(b['orientation']['arm_induced_pitch']-induced_pitch)>1e-8:
            raise ValueError('Workspace bank orientation differs from configuration')
        self.spheres = torch.tensor([n['sphere'] for n in b['nodes']], device=device, dtype=torch.float)
        if not torch.isfinite(self.spheres).all() or not (self.spheres[:,0]>0).all():
            raise ValueError('Invalid workspace goal')
        adjacency = [set() for _ in b['nodes']]
        for e in b['edges']:
            a,c = e['source'],e['target']
            if a==c or not (0<=a<len(adjacency) and 0<=c<len(adjacency)):
                raise ValueError('Invalid workspace edge')
            adjacency[a].add(c);adjacency[c].add(a)
        if not all(adjacency):
            raise ValueError('Isolated workspace goal')
        seen={0};pending=[0]
        while pending:
            for v in adjacency[pending.pop()]:
                if v not in seen:
                    seen.add(v);pending.append(v)
        if len(seen)!=len(adjacency):
            raise ValueError('Unreachable workspace component')
        self.degree = torch.tensor([len(v) for v in adjacency],device=device,dtype=torch.long)
        table=[[0]*max(map(len,adjacency)) for _ in adjacency]
        for i,values in enumerate(adjacency):
            table[i][:len(values)]=sorted(values)
        self.neighbors=torch.tensor(table,device=device,dtype=torch.long)
        self.current=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.center_z=float(b['center'][2])
        self.center_x=float(b['center'][0])

    def reset(self, ids):
        self.current[ids]=0
        return self.spheres[0].expand(len(ids),-1)

    def sample(self, ids):
        source=self.current[ids]
        slot=(torch.rand(len(ids),device=self.spheres.device)*self.degree[source]).long()
        target=self.neighbors[source,slot]
        self.current[ids]=target
        return self.spheres[target]

    def pitch(self, sphere):
        """Same continuous position-linked pitch used to screen every edge."""
        r,phi=sphere[:,0],sphere[:,1]
        z=self.center_z+r*torch.sin(phi)
        x=self.center_x+r*torch.cos(phi)
        blend=torch.clamp((.25-z)/.15,0.,1.)
        floor=(175.-25.*torch.clamp((x-.55)/.425,0.,1.))*math.pi/180.
        return (1.-blend)*(self.bank['orientation']['arm_induced_pitch']-phi)+blend*floor
