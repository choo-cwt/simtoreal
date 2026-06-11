"""Calibrate and read XLeRobot head servo angles from raw Feetech ticks."""

import argparse
import json
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


DEFAULT_CONFIG = Path(__file__).with_name("xlerobot_head_servos.json")
TICKS_PER_REV = 4095.0


def read_raw(port: str, pan_id: int, tilt_id: int) -> dict[str, float]:
    bus = FeetechMotorsBus(
        port=port,
        motors={
            "head_pan": Motor(pan_id, "sts3215", MotorNormMode.RANGE_M100_100),
            "head_tilt": Motor(tilt_id, "sts3215", MotorNormMode.RANGE_M100_100),
        },
    )
    try:
        bus.connect(handshake=False)
        return bus.sync_read("Present_Position", normalize=False, num_retry=3)
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


def signed_tick_delta(raw: float, center: float) -> float:
    return ((raw - center + TICKS_PER_REV / 2) % TICKS_PER_REV) - TICKS_PER_REV / 2


def raw_to_deg(raw: float, center: float, sign: float) -> float:
    return sign * signed_tick_delta(raw, center) * 360.0 / TICKS_PER_REV


def main():
    parser = argparse.ArgumentParser(description="Calibrate/read XLeRobot head servo angles")
    parser.add_argument("--port", default="/dev/ttyACM1")
    parser.add_argument("--pan-id", type=int, default=9)
    parser.add_argument("--tilt-id", type=int, default=10)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--save-zero", action="store_true", help="Save current raw ticks as 0 deg")
    parser.add_argument("--pan-sign", type=float, default=1.0)
    parser.add_argument("--tilt-sign", type=float, default=1.0)
    args = parser.parse_args()

    raw = read_raw(args.port, args.pan_id, args.tilt_id)

    if args.save_zero:
        config = {
            "port": args.port,
            "pan_id": args.pan_id,
            "tilt_id": args.tilt_id,
            "pan_center": raw["head_pan"],
            "tilt_center": raw["head_tilt"],
            "pan_sign": args.pan_sign,
            "tilt_sign": args.tilt_sign,
            "ticks_per_rev": TICKS_PER_REV,
        }
        args.config.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Saved head servo zero calibration to {args.config}")
        print(f"pan_center={raw['head_pan']:.0f} tilt_center={raw['head_tilt']:.0f}")
        return

    if not args.config.exists():
        print(f"No calibration file found: {args.config}")
        print("Run with --save-zero first after placing the camera at your chosen 0 deg pose.")
        print(f"Current raw: pan={raw['head_pan']:.0f} tilt={raw['head_tilt']:.0f}")
        return

    config = json.loads(args.config.read_text())
    pan_deg = raw_to_deg(raw["head_pan"], config["pan_center"], config["pan_sign"])
    tilt_deg = raw_to_deg(raw["head_tilt"], config["tilt_center"], config["tilt_sign"])
    print(f"head_pan: raw={raw['head_pan']:.0f} deg={pan_deg:.3f}")
    print(f"head_tilt: raw={raw['head_tilt']:.0f} deg={tilt_deg:.3f}")


if __name__ == "__main__":
    main()
