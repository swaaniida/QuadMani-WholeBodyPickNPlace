"""Read-only PiPER V2 SDK queries. No enable, move, config, zeroing or reset.

Not executed by Codex. Explicit CAN interface required; use the SDK/protocol
matching the device firmware. Saves raw responses, never guesses unit versions.
"""
import argparse,json,time,re
from pathlib import Path

def speed_unit(firmware):
    found=re.search(r'S-V(\d+)\.(\d+)-(\d+)',str(firmware))
    if not found:return None
    version=tuple(map(int,found.groups()))
    return .01 if version >= (1,5,8) else .001

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--can',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import piper_sdk
    interface=getattr(piper_sdk,'C_PiperInterface_V2',None)
    if interface is None:raise RuntimeError('Install the device-matching V2 SDK; refusing to guess a protocol class')
    arm=interface(can_name=args.can)
    arm.ConnectPort(piper_init=False)
    try:
        arm.SearchPiperFirmwareVersion()
        arm.SearchAllMotorMaxAngleSpd()
        arm.SearchAllMotorMaxAccLimit()
        time.sleep(1)
        firmware=arm.GetPiperFirmwareVersion()
        result=dict(firmware=firmware,speed_raw_unit_rad_s=speed_unit(firmware),
                    raw_angle_speed=str(arm.GetAllMotorAngleLimitMaxSpd()),
                    raw_acceleration=str(arm.GetAllMotorMaxAccLimit()),
                    acceleration_raw_unit_rad_s2=.001,
                    note='Raw query responses only. Check per-joint response freshness before use. No motion/config commands sent.')
        with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
        print(args.output)
    finally:arm.DisconnectPort()
if __name__=='__main__':main()
