"""Portable setup and development commands; requires Python 3.10 or newer."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
ENV_DIR = ROOT / "pipeline" / ".venv"
PYTHON = ENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
METRICA_REVISION = "e706dd506b360d69d9d123d5b8026e7294b13996"
METRICA_URL = "https://github.com/metrica-sports/sample-data.git"


def run(*args, capture=False):
    # Do not let a developer's private dependency folders mask missing packages.
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    command = [str(arg) for arg in args]
    print("+ " + subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=True,
                          text=True, capture_output=capture)


def require_environment():
    if not PYTHON.is_file():
        raise ValueError("Environment missing. Run: python dev.py setup")


def test():
    require_environment()
    run(PYTHON, "-m", "pip", "check")
    # The model tests can skip torch in minimal environments; this command must not.
    run(PYTHON, "-c", "import numpy, pandas, scipy, sklearn, kloppy, torch; "
        "print('Dependencies ready; PyTorch', torch.__version__)")
    run(PYTHON, "-m", "unittest", "discover", "-s", "pipeline", "-t", ".", "-v")


def setup():
    if not PYTHON.is_file():
        print("Creating isolated environment in pipeline/.venv", flush=True)
        venv.EnvBuilder(with_pip=True).create(ENV_DIR)
    run(PYTHON, "-m", "pip", "install", "--upgrade", "pip")
    # CPU builds keep the default setup independent of NVIDIA drivers.
    # PyTorch publishes macOS wheels on PyPI, not the CPU wheel index.
    torch_args = [] if sys.platform == "darwin" else [
        "--index-url", "https://download.pytorch.org/whl/cpu"]
    run(PYTHON, "-m", "pip", "install", "torch>=2.6,<3", *torch_args)
    run(PYTHON, "-m", "pip", "install", "-r", ROOT / "pipeline/requirements.txt")
    test()
    print("\nSetup and tests passed. Start the viewer with: python dev.py viewer")


def prepare(bench_src):
    require_environment()
    if not shutil.which("git"):
        raise ValueError("Install Git and make it available on PATH, then retry.")
    checkout = bench_src.resolve() if bench_src else ROOT / "data/metrica-sample-data"
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", METRICA_URL, checkout)
        run("git", "-C", checkout, "checkout", "--detach", METRICA_REVISION)
    else:
        # Never reset an existing checkout or discard someone else's changes.
        revision = run("git", "-C", checkout, "rev-parse", "HEAD", capture=True).stdout.strip()
        if revision != METRICA_REVISION:
            raise ValueError(f"{checkout} must be at Metrica revision {METRICA_REVISION}. "
                             "Use --bench-src with a separate checkout at that revision.")
        dirty = run("git", "-C", checkout, "status", "--porcelain", capture=True).stdout.strip()
        if dirty:
            raise ValueError(f"Metrica checkout has local changes: {checkout}. "
                             "Use a clean checkout for reproducible preparation.")
    run(PYTHON, "-m", "pipeline.run", "prepare-mall", "--bench-src", checkout)
    print("\nBenchmarks ready. Run: python dev.py model --data-dir PATH_TO_CHALLENGE_DATA")


def model(data_dir):
    require_environment()
    bundle = (data_dir or Path(os.environ.get("MTAH_DATA_DIR") or ROOT / "data")).resolve()
    if bundle.name == "tactical":
        bundle = bundle.parent
    required = [bundle / "tactical" / name for name in (
        "BAR-ATH_tactical_atd.txt", "BAR-ATH_tactical_atd.xml",
        "BAR-ATH_tactical_smart_tagging.xml")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Challenge data is missing:\n  " + "\n  ".join(missing) +
                         "\nObtain the challenge bundle separately, then pass --data-dir PATH.")
    benchmark_dir = ROOT / "pipeline/cache/mall/bench"
    tags = ["bench1", "bench2", "bench3"] + [f"dfl{i}" for i in range(1, 8)]
    if any(not (benchmark_dir / f"{tag}_gt.csv").is_file() for tag in tags):
        raise ValueError("Training benchmarks are missing. Run: python dev.py prepare")
    run(PYTHON, "-m", "pipeline.run", "mall", "--mode", "model", "--data-dir", bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="Create an environment, install dependencies, and run tests")
    commands.add_parser("test", help="Run dependency checks and all pipeline unit tests; no match data needed")
    viewer = commands.add_parser("viewer", help="Serve the viewer; no dependency installation needed")
    viewer.add_argument("--port", type=int, default=8081)
    preparation = commands.add_parser("prepare", help="Download and prepare the ten public training games")
    preparation.add_argument("--bench-src", type=Path, help="Existing clean Metrica checkout at the documented revision")
    prediction = commands.add_parser("model", help="Train/reuse Mall models and export match predictions")
    prediction.add_argument("--data-dir", type=Path, help="Challenge bundle root or its tactical folder")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required; Python 3.10 is the tested baseline.")
    try:
        if args.command == "setup":
            setup()
        elif args.command == "test":
            test()
        elif args.command == "viewer":
            run(sys.executable, ROOT / "possession-viewer/server.py", "--port", args.port)
        elif args.command == "prepare":
            prepare(args.bench_src)
        elif args.command == "model":
            model(args.data_dir)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        print(f"Command failed (exit {error.returncode}). Fix the error above and retry.", file=sys.stderr)
        return error.returncode
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
