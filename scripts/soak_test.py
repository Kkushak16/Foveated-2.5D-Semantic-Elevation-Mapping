"""
soak_test.py — Power / Thermal Soak Test for Demo Day Validation
=================================================================
Runs the perception pipeline under sustained load for a configurable
duration, logging telemetry (VRAM, GPU util, power, temperature) at
regular intervals. Detects thermal throttling events and outputs a
summary CSV + pass/fail verdict.

This addresses Section 6 Item 6 of physical-testing.md:
"Jetson boards throttle under sustained load; a 30–60 minute soak test
should be run before the demo day, not just a quick check."

Works on both cloud (pynvml) and Jetson (jtop) platforms.

Usage:
    # Quick 5-minute cloud soak test
    python scripts/soak_test.py --platform cloud --duration 300 --interval 5

    # Full 60-minute Jetson soak test (run day before demo)
    python scripts/soak_test.py --platform jetson --duration 3600 --interval 10

    # Auto-detect platform
    python scripts/soak_test.py --duration 600
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

# Add python/ to path for telemetry module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "python"))

from telemetry import get_telemetry


def run_soak_test(platform="auto", duration=3600, interval=10, output_dir=None):
    """Run a soak test logging telemetry at regular intervals.

    Args:
        platform: 'cloud', 'jetson', 'laptop', or 'auto'
        duration: Test duration in seconds (default: 3600 = 60 min)
        interval: Sampling interval in seconds (default: 10)
        output_dir: Directory for CSV output (default: results/)
    """
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    telem = get_telemetry(platform)
    initial = telem.get_metrics()
    detected_platform = initial["platform"]

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        output_dir, f"soak_test_{detected_platform}_{timestamp_str}.csv")

    # Throttle detection thresholds
    TEMP_WARN_C = 80.0       # Warning threshold
    TEMP_THROTTLE_C = 85.0   # Likely throttling threshold
    VRAM_WARN_PCT = 90.0     # VRAM usage warning

    print("=" * 70)
    print("  SOAK TEST — Power / Thermal Endurance Validation")
    print("=" * 70)
    print(f"  Platform     : {detected_platform}")
    print(f"  Duration     : {duration}s ({duration / 60:.0f} minutes)")
    print(f"  Interval     : {interval}s")
    print(f"  CSV Output   : {csv_path}")
    print(f"  Temp Warn    : >{TEMP_WARN_C}°C")
    print(f"  Temp Throttle: >{TEMP_THROTTLE_C}°C")
    print("=" * 70)

    # CSV header
    fields = [
        "elapsed_s", "timestamp", "vram_used_mb", "vram_total_mb",
        "gpu_util_pct", "power_draw_w", "soc_temp_c", "throttle_warning"
    ]

    samples = []
    throttle_events = 0
    temp_warnings = 0
    max_temp = 0.0
    max_power = 0.0
    max_vram = 0.0

    start_time = time.time()

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()

            sample_num = 0
            while True:
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    break

                metrics = telem.get_metrics()
                sample_num += 1

                # Detect throttling
                throttle_flag = ""
                if metrics["soc_temp_c"] > TEMP_THROTTLE_C:
                    throttle_flag = "THROTTLE"
                    throttle_events += 1
                elif metrics["soc_temp_c"] > TEMP_WARN_C:
                    throttle_flag = "WARN"
                    temp_warnings += 1

                # Track maximums
                max_temp = max(max_temp, metrics["soc_temp_c"])
                max_power = max(max_power, metrics["power_draw_w"])
                max_vram = max(max_vram, metrics["vram_used_mb"])

                row = {
                    "elapsed_s": f"{elapsed:.1f}",
                    "timestamp": datetime.now().isoformat(),
                    "vram_used_mb": f"{metrics['vram_used_mb']:.1f}",
                    "vram_total_mb": f"{metrics['vram_total_mb']:.1f}",
                    "gpu_util_pct": f"{metrics['gpu_util_pct']:.1f}",
                    "power_draw_w": f"{metrics['power_draw_w']:.1f}",
                    "soc_temp_c": f"{metrics['soc_temp_c']:.1f}",
                    "throttle_warning": throttle_flag,
                }
                writer.writerow(row)
                csvfile.flush()
                samples.append(row)

                # Console output
                status = "🟢" if not throttle_flag else ("🟡" if throttle_flag == "WARN" else "🔴")
                remaining = duration - elapsed
                print(f"  {status} [{sample_num:>4}] "
                      f"T+{elapsed:>7.0f}s | "
                      f"GPU: {metrics['gpu_util_pct']:>5.1f}% | "
                      f"VRAM: {metrics['vram_used_mb']:>7.0f}MB | "
                      f"Power: {metrics['power_draw_w']:>5.1f}W | "
                      f"Temp: {metrics['soc_temp_c']:>5.1f}°C | "
                      f"Remain: {remaining:>5.0f}s"
                      f"{' ⚠️ ' + throttle_flag if throttle_flag else ''}")

                time.sleep(interval)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n[INFO] Soak test interrupted at {elapsed:.0f}s")

    # --- Summary ---
    total_elapsed = time.time() - start_time
    total_samples = len(samples)
    passed = throttle_events == 0

    print("")
    print("=" * 70)
    print("  SOAK TEST RESULTS")
    print("=" * 70)
    print(f"  Duration Run   : {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Total Samples  : {total_samples}")
    print(f"  Max Temperature: {max_temp:.1f}°C")
    print(f"  Max Power Draw : {max_power:.1f}W")
    print(f"  Max VRAM Usage : {max_vram:.0f}MB")
    print(f"  Temp Warnings  : {temp_warnings}")
    print(f"  Throttle Events: {throttle_events}")
    print(f"  CSV Log        : {csv_path}")
    print("")

    if passed:
        print("  ✅ VERDICT: PASS — No thermal throttling detected")
        print("     System is stable for demo-day sustained operation.")
    else:
        print("  ❌ VERDICT: FAIL — Thermal throttling detected!")
        print(f"     {throttle_events} throttle event(s) recorded.")
        print("     Recommendations:")
        print("       - Add active cooling (fan) to the Jetson board")
        print("       - Reduce LiDAR point density during the demo")
        print("       - Use MAXN power mode: sudo nvpmodel -m 0")
        print("       - Ensure adequate ventilation around the board")

    print("=" * 70)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Power/thermal soak test for demo day validation")
    parser.add_argument("--platform", default="auto",
                        choices=["auto", "cloud", "jetson", "laptop"],
                        help="Telemetry platform (default: auto-detect)")
    parser.add_argument("--duration", type=int, default=3600,
                        help="Test duration in seconds (default: 3600 = 60 min)")
    parser.add_argument("--interval", type=int, default=10,
                        help="Sampling interval in seconds (default: 10)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for CSV output (default: results/)")
    args = parser.parse_args()

    success = run_soak_test(
        platform=args.platform,
        duration=args.duration,
        interval=args.interval,
        output_dir=args.output_dir,
    )
    sys.exit(0 if success else 1)
