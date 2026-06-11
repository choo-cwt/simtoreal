"""Read XLeRobot head servo positions without commanding the robot.

This bypasses the default SO101 follower config, which only registers arm
motors 1-6. The head camera servos are expected to be Feetech STS3215 motors
with IDs 9 and 10 on the same serial bus.
"""

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def main():
    parser = argparse.ArgumentParser(description="Read XLeRobot head servo angles")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--pan-id", type=int, default=9)
    parser.add_argument("--tilt-id", type=int, default=10)
    parser.add_argument(
        "--center",
        type=float,
        default=2048.0,
        help="Raw tick treated as zero degrees for the approximate angle printout",
    )
    args = parser.parse_args()

    bus = FeetechMotorsBus(
        port=args.port,
        motors={
            "head_pan": Motor(args.pan_id, "sts3215", MotorNormMode.RANGE_M100_100),
            "head_tilt": Motor(args.tilt_id, "sts3215", MotorNormMode.RANGE_M100_100),
        },
    )

    try:
        bus.connect()
        pos = bus.sync_read("Present_Position", normalize=False, num_retry=3)
        for name, value in pos.items():
            approx_deg = (value - args.center) * 360.0 / 4095.0
            print(f"{name}: raw={value:.0f} approx_deg={approx_deg:.3f}")
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
