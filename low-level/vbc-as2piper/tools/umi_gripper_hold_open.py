#!/usr/bin/env python3
"""Minimal example of keeping PiPER's two simulated finger DOFs open.

Call `configure_open_gripper(gym, env, actor)` for each actor AFTER it is created.
This is the per-actor API. For thousands of RL envs use batched position-target
TENSORS instead of calling this every simulation step. Keeps main policy at 18
outputs but DOES NOT automatically adapt VBC's old 18-DOF buffer dimensions.
"""
import numpy as np
from isaacgym import gymapi

FINGER = ('piper_gripper_joint1','piper_gripper_joint2')
OPEN = (0.05,-0.05)


def configure_open_gripper(gym, env, actor, stiffness=200., damping=5.):
    names=list(gym.get_actor_dof_names(env,actor))
    assert len(names)==20,(len(names),names)
    ix=[names.index(n) for n in FINGER]
    props=gym.get_actor_dof_properties(env,actor)
    for i in ix:
        props['driveMode'][i]=gymapi.DOF_MODE_POS
        props['stiffness'][i]=stiffness
        props['damping'][i]=damping
        # Do not silently raise the manufacturer's max effort.
    gym.set_actor_dof_properties(env,actor,props)
    # Initialize both gripper DOFs already open; prevents closing on sim startup.
    states=gym.get_actor_dof_states(env,actor,gymapi.STATE_ALL)
    targets=np.array(states['pos'],copy=True)
    for i,q in zip(ix,OPEN):
        states['pos'][i]=q
        states['vel'][i]=0
        targets[i]=q
    gym.set_actor_dof_states(env,actor,states,gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env,actor,targets)
    return ix


def set_open_target(gym,env,actor):
    """Repeated target command in a single-env test, not optimized for 6k envs."""
    names=list(gym.get_actor_dof_names(env,actor))
    ix=[names.index(n) for n in FINGER]
    state=gym.get_actor_dof_states(env,actor,gymapi.STATE_POS)
    targets=np.array(state['pos'],copy=True)
    for i,q in zip(ix,OPEN):targets[i]=q
    gym.set_actor_dof_position_targets(env,actor,targets)
