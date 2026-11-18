"""Metric CAD registration helpers. No robot control or model inference."""
import numpy as np


def planar_circles(mesh,min_radius=.5,max_radius=30.,min_area=5.,max_error=.02,min_vertices=12):
    """Fit circles to closed planar facet boundaries. Units follow input mesh.

    Returned circles are geometric candidates; callers must distinguish mounting
    holes from outer boundaries, bores, counterbores and decorative geometry.
    """
    records=[]
    for fi,(edges,n,area) in enumerate(zip(mesh.facets_boundary,mesh.facets_normal,mesh.facets_area)):
        if area<min_area:continue
        graph={}
        for a,b in edges:
            graph.setdefault(int(a),set()).add(int(b));graph.setdefault(int(b),set()).add(int(a))
        remaining=set(graph)
        while remaining:
            todo=[remaining.pop()];group=set(todo)
            while todo:
                for j in graph[todo.pop()]:
                    if j not in group:group.add(j);remaining.discard(j);todo.append(j)
            if len(group)<min_vertices or any(len(graph[i])!=2 for i in group):continue
            xyz=mesh.vertices[list(group)];seed=np.eye(3)[np.argmin(np.abs(n))]
            x=np.cross(n,seed);x/=np.linalg.norm(x);y=np.cross(n,x)
            origin=xyz.mean(0);uv=(xyz-origin)@np.array([x,y]).T
            sol=np.linalg.lstsq(np.column_stack((2*uv,np.ones(len(uv)))),np.sum(uv**2,axis=1),rcond=None)[0]
            squared=sol[2]+sol[:2]@sol[:2]
            if squared<=0:continue
            radius=float(np.sqrt(squared));error=float(np.abs(np.linalg.norm(uv-sol[:2],axis=1)-radius).max())
            if error>max_error or not min_radius<=radius<=max_radius:continue
            center=origin+x*sol[0]+y*sol[1]
            records.append(dict(facet=fi,center=center.tolist(),normal=n.tolist(),radius=radius,
                                max_fit_error=error,vertices=len(xyz),facet_area=float(area)))
    return records


def point_transform(points,T):return np.asarray(points)@T[:3,:3].T+T[:3,3]


def frame_from_pair(a,b,normal):
    a=np.asarray(a,float);b=np.asarray(b,float);z=np.asarray(normal,float);z/=np.linalg.norm(z)
    x=b-a;x-=z*(x@z)
    if np.linalg.norm(x)<1e-10:raise ValueError('Pair direction parallel to face normal')
    x/=np.linalg.norm(x);y=np.cross(z,x)
    T=np.eye(4);T[:3,:3]=np.column_stack((x,y,z));T[:3,3]=(a+b)/2
    return T


def align_pair(source,target,source_normal,target_normal):
    """Rigid registration of two hole centers and oriented face normal."""
    source=np.asarray(source,float);target=np.asarray(target,float)
    if source.shape!=(2,3) or target.shape!=(2,3):raise ValueError('Two 3D hole centers required')
    T=frame_from_pair(*target,target_normal)@np.linalg.inv(frame_from_pair(*source,source_normal))
    residual=np.linalg.norm(point_transform(source,T)-target,axis=1)
    return T,residual
