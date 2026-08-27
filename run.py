"""
run.py — Project Launcher
===========================
Runs any src/ module from the project root, avoiding the Python
prefix-detection issue with embedded/portable Python installations.

Usage (from the Lidar Mapping directory):
    python run.py train_classifier --synthetic
    python run.py evaluate --synthetic
    python run.py classify_clusters
    python run.py validate_ground_seg --synthetic --no-vis
    python run.py clustering
    python run.py feature_extraction
    python run.py dataset_loader /path/to/semantickitti
"""

import importlib
import os
import sys

# Add src/ to Python's module search path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <module_name> [args...]")
        print("\nAvailable modules:")
        for f in sorted(os.listdir(SRC_DIR)):
            if f.endswith(".py") and f != "__init__.py":
                print(f"  {f[:-3]}")
        sys.exit(1)

    module_name = sys.argv[1]

    # Remove 'run.py' and module name from argv so the target script
    # sees its own args correctly
    sys.argv = sys.argv[1:]  # ['module_name', '--synthetic', ...]
    sys.argv[0] = os.path.join(SRC_DIR, f"{module_name}.py")

    # Import and run the module's __main__ block
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, "main"):
            mod.main()
        else:
            print(f"Module '{module_name}' has no main() function.")
            print("It was imported successfully (side-effects ran).")
    except ModuleNotFoundError as e:
        print(f"ERROR: Module '{module_name}' not found in src/")
        print(f"  Detail: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
