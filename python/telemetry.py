"""
telemetry.py — Platform-Aware GPU/System Telemetry Module
==========================================================
Provides a unified telemetry API that works across deployment tiers:
  - CloudGPUTelemetry: pynvml-based (Colab / RunPod / cloud GPU)
  - JetsonTelemetry: jtop/tegrastats-based (Jetson Orin / Xavier)
  - LaptopTelemetry: psutil-based CPU fallback (no GPU required)

Usage:
    from telemetry import get_telemetry
    telem = get_telemetry('cloud')   # or 'jetson' or 'laptop'
    metrics = telem.get_metrics()
    # -> {'vram_used_mb', 'vram_total_mb', 'gpu_util_pct',
    #     'power_draw_w', 'soc_temp_c', 'latency_ms', 'platform'}

Selftest:
    python python/telemetry.py --selftest
"""

import time
import json
import os
import sys

# ---------------------------------------------------------------------------
# Base class — uniform interface for all telemetry backends
# ---------------------------------------------------------------------------
class BaseTelemetry:
    """Abstract telemetry collector with a uniform get_metrics() API."""

    PLATFORM = "unknown"

    def get_metrics(self):
        """Return a dict with standardised telemetry keys."""
        raise NotImplementedError

    @staticmethod
    def _default_metrics():
        return {
            "vram_used_mb": 0.0,
            "vram_total_mb": 0.0,
            "gpu_util_pct": 0.0,
            "power_draw_w": 0.0,
            "soc_temp_c": 0.0,
            "latency_ms": 0.0,
            "platform": "unknown",
            "timestamp": time.time(),
        }


# ---------------------------------------------------------------------------
# Tier 1: Cloud GPU (Colab / RunPod) — uses pynvml
# ---------------------------------------------------------------------------
class CloudGPUTelemetry(BaseTelemetry):
    """Reads telemetry from a discrete NVIDIA GPU via pynvml."""

    PLATFORM = "cloud"

    def __init__(self, gpu_index=0):
        self.gpu_index = gpu_index
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception as exc:
            print(f"[telemetry][cloud] pynvml init failed: {exc}", file=sys.stderr)
            self._nvml = None

    def get_metrics(self):
        m = self._default_metrics()
        m["platform"] = self.PLATFORM
        if self._nvml is None or self._handle is None:
            return m
        try:
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            m["vram_used_mb"] = mem.used / (1024 * 1024)
            m["vram_total_mb"] = mem.total / (1024 * 1024)
        except Exception:
            pass
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            m["gpu_util_pct"] = float(util.gpu)
        except Exception:
            pass
        try:
            power_mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
            m["power_draw_w"] = power_mw / 1000.0
        except Exception:
            pass
        try:
            temp = self._nvml.nvmlDeviceGetTemperature(
                self._handle, self._nvml.NVML_TEMPERATURE_GPU)
            m["soc_temp_c"] = float(temp)
        except Exception:
            pass
        m["timestamp"] = time.time()
        return m


# ---------------------------------------------------------------------------
# Tier 2: Jetson board — uses jtop (jetson-stats) or tegrastats parsing
# ---------------------------------------------------------------------------
class JetsonTelemetry(BaseTelemetry):
    """Reads telemetry from a Jetson board via jtop (jetson-stats package).
    pynvml does NOT work on Jetson's integrated GPU, so this uses the
    jetson-stats library (pip install jetson-stats) or falls back to
    parsing /usr/bin/tegrastats output."""

    PLATFORM = "jetson"

    def __init__(self):
        self._jtop = None
        try:
            from jtop import jtop
            self._jtop_cls = jtop
        except ImportError:
            self._jtop_cls = None
            print("[telemetry][jetson] jetson-stats (jtop) not installed. "
                  "Install with: sudo pip3 install jetson-stats",
                  file=sys.stderr)

    def get_metrics(self):
        m = self._default_metrics()
        m["platform"] = self.PLATFORM
        if self._jtop_cls is None:
            return m
        try:
            with self._jtop_cls() as jetson:
                stats = jetson.stats
                # GPU utilization
                m["gpu_util_pct"] = float(stats.get("GPU", 0))
                # Power (total board power in mW)
                power = jetson.power
                if power and "tot" in power:
                    pw = power["tot"]
                    # jtop reports power in mW
                    m["power_draw_w"] = float(pw.get("cur", 0)) / 1000.0
                # Temperature — use GPU temp if available, else SoC
                temps = jetson.temperature
                if temps:
                    if "GPU" in temps:
                        m["soc_temp_c"] = float(temps["GPU"])
                    elif "SoC" in temps:
                        m["soc_temp_c"] = float(temps["SoC"])
                    elif "CPU" in temps:
                        m["soc_temp_c"] = float(temps["CPU"])
                # VRAM — Jetson uses shared memory; report from RAM stats
                ram = jetson.memory
                if ram and "RAM" in ram:
                    r = ram["RAM"]
                    m["vram_used_mb"] = float(r.get("used", 0)) / (1024 * 1024)
                    m["vram_total_mb"] = float(r.get("tot", 0)) / (1024 * 1024)
        except Exception as exc:
            print(f"[telemetry][jetson] jtop read failed: {exc}", file=sys.stderr)
        m["timestamp"] = time.time()
        return m


# ---------------------------------------------------------------------------
# Tier 3: Laptop CPU (no GPU) — uses psutil
# ---------------------------------------------------------------------------
class LaptopTelemetry(BaseTelemetry):
    """CPU-only fallback telemetry using psutil."""

    PLATFORM = "laptop"

    def __init__(self):
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            self._psutil = None
            print("[telemetry][laptop] psutil not installed. "
                  "Install with: pip install psutil", file=sys.stderr)

    def get_metrics(self):
        m = self._default_metrics()
        m["platform"] = self.PLATFORM
        if self._psutil is None:
            return m
        try:
            vm = self._psutil.virtual_memory()
            m["vram_used_mb"] = vm.used / (1024 * 1024)
            m["vram_total_mb"] = vm.total / (1024 * 1024)
        except Exception:
            pass
        try:
            m["gpu_util_pct"] = self._psutil.cpu_percent(interval=0.1)
        except Exception:
            pass
        # No power / temperature on generic laptops without hw-specific libs
        m["power_draw_w"] = 0.0
        m["soc_temp_c"] = 0.0
        m["timestamp"] = time.time()
        return m


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def get_telemetry(platform="auto"):
    """Return the appropriate telemetry backend for the given platform.

    Args:
        platform: 'cloud', 'jetson', 'laptop', or 'auto' (auto-detect).
    """
    if platform == "auto":
        # Auto-detect: try pynvml (cloud), then jtop (jetson), then psutil
        try:
            import pynvml
            pynvml.nvmlInit()
            pynvml.nvmlDeviceGetHandleByIndex(0)
            return CloudGPUTelemetry()
        except Exception:
            pass
        try:
            from jtop import jtop
            return JetsonTelemetry()
        except ImportError:
            pass
        return LaptopTelemetry()

    platform = platform.lower()
    if platform == "cloud":
        return CloudGPUTelemetry()
    elif platform == "jetson":
        return JetsonTelemetry()
    elif platform == "laptop":
        return LaptopTelemetry()
    else:
        print(f"[telemetry] Unknown platform '{platform}', falling back to laptop",
              file=sys.stderr)
        return LaptopTelemetry()


# ---------------------------------------------------------------------------
# JSON serialiser for the WebSocket bridge
# ---------------------------------------------------------------------------
def telemetry_to_json(platform="auto"):
    """Return telemetry metrics as a JSON string (used by the Node bridge)."""
    telem = get_telemetry(platform)
    return json.dumps(telem.get_metrics())


# ---------------------------------------------------------------------------
# Self-test CLI
# ---------------------------------------------------------------------------
def _selftest():
    print("=" * 60)
    print("  TELEMETRY MODULE SELF-TEST")
    print("=" * 60)
    for p in ["cloud", "jetson", "laptop"]:
        print(f"\n--- Platform: {p} ---")
        try:
            t = get_telemetry(p)
            m = t.get_metrics()
            for k, v in m.items():
                print(f"  {k:<20}: {v}")
            print(f"  [OK] {p} backend OK")
        except Exception as exc:
            print(f"  [FAIL] {p} backend error: {exc}")

    print(f"\n--- Auto-detect ---")
    t = get_telemetry("auto")
    m = t.get_metrics()
    print(f"  Detected platform: {m['platform']}")
    for k, v in m.items():
        print(f"  {k:<20}: {v}")
    print(f"\n[OK] Self-test complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Platform-aware GPU telemetry")
    parser.add_argument("--selftest", action="store_true", help="Run self-test")
    parser.add_argument("--platform", default="auto",
                        choices=["auto", "cloud", "jetson", "laptop"])
    parser.add_argument("--json", action="store_true",
                        help="Output metrics as JSON")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
    elif args.json:
        print(telemetry_to_json(args.platform))
    else:
        t = get_telemetry(args.platform)
        m = t.get_metrics()
        for k, v in m.items():
            print(f"{k}: {v}")
