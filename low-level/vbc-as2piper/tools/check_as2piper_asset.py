from pathlib import Path
from isaacgym import gymapi

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "low-level/legged_gym/resources/robots/as2piper"

gym = gymapi.acquire_gym()

# CPU PhysX로 URDF 로딩 검증
params = gymapi.SimParams()
params.dt = 1.0 / 60.0
params.up_axis = gymapi.UP_AXIS_Z
params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
params.use_gpu_pipeline = False
params.physx.use_gpu = False

# -1: 그래픽 디바이스 사용 안 함 (Headless)
sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)

if sim is None:
    raise RuntimeError("Failed to create Isaac Gym simulation")

try:
    options = gymapi.AssetOptions()
    options.fix_base_link = False
    options.collapse_fixed_joints = True

    asset = gym.load_asset(
        sim,
        str(ASSET_ROOT),
        "urdf/as2piper.urdf",
        options
    )

    if asset is None:
        raise RuntimeError("Failed to load AS2+PiPER asset")

    dof_names = list(gym.get_asset_dof_names(asset))
    body_names = list(gym.get_asset_rigid_body_names(asset))
    properties = gym.get_asset_dof_properties(asset)

    print("\n===== AS2 + PiPER =====")
    print("DOF count:", len(dof_names))
    print("Rigid body count:", len(body_names))

    print("\n===== DOF ORDER =====")
    for i, name in enumerate(dof_names):
        print(
            f"{i:2d}: {name:22s} "
            f"limits=[{properties['lower'][i]:.3f}, "
            f"{properties['upper'][i]:.3f}]"
        )

    print("\n===== RIGID BODY ORDER =====")
    for i, name in enumerate(body_names):
        print(f"{i:2d}: {name}")

    print("\n===== VALIDATION =====")
    assert len(dof_names) == 18, "Expected 18 DOFs"
    assert sum(n.startswith("piper_") for n in dof_names) == 6
    assert "piper_link6" in body_names, "PiPER end link missing"

    print("18 DOF: OK")
    print("PiPER arm: OK")
    print("PiPER end link: OK")
    print("ASSET LOAD SUCCESS")

finally:
    gym.destroy_sim(sim)
