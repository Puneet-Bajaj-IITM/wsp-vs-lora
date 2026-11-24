"""Bundle the project and a list of shell commands into one self-contained, private Kaggle GPU kernel.

The kernel unpacks the sources, rebuilds the task data, runs the commands from the project root, and leaves
results/, runs/ and checkpoints/ under /kaggle/working/out for fetch.sh to download.

Kaggle caps a kernel's source at 1 MB, so the bundle holds only what a job needs: the library, the scripts,
the control edits, every run's ledger, the edit folder of each run its commands select (through
WSP_MODEL / WSP_TASK / WSP_RUN), any runs/ path the commands name, and with BUNDLE_WORK=<model>/<task>/<run>
that run's work folder.

Usage: KAGGLE_USERNAME=<you> python kaggle_runner/bundle.py <job> "<command>" ["<command>" ...]
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "kaggle_runner" / "build"
SLUG_PREFIX = "wsp"
MACHINE_SHAPE = os.environ.get("KAGGLE_MACHINE", "NvidiaTeslaT4")

ALWAYS_BUNDLED = ("src", "scripts", "controls")
BUNDLED_SUFFIXES = (".py", ".md", ".json", ".npy", ".pt")
LEDGER_GLOB = "runs/*/*/*/ledger.json"
DEFAULT_MODEL, DEFAULT_TASK = "qwen", "twin"
ENV_PATTERN = re.compile(r"\b(WSP_MODEL|WSP_TASK|WSP_RUN)=(\S*)")
RUNS_PATH_PATTERN = re.compile(r"(?<![\w/])runs/[\w./-]+")

# Kaggle's limit is 1 MB of source; the base64 archive must stay under it with some margin.
MAX_BUNDLED_BYTES = 950_000
MAX_FILE_BYTES = 200_000

RUNNER_TEMPLATE = '''import base64, io, os, pathlib, shutil, subprocess, sys, tarfile, uuid
ARCHIVE = __ARCHIVE__
COMMANDS = __COMMANDS__
WORK = pathlib.Path("/kaggle/tmp/proj")
OUT = pathlib.Path("/kaggle/working/out")
WORK.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARCHIVE)), mode="r:xz") as tar:
    # The filter argument exists only on newer Pythons.
    try:
        tar.extractall(WORK, filter="data")
    except TypeError:
        tar.extractall(WORK)
# Image adjustments used for every study job: torchao is removed (not used) and peft is installed.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft>=0.13"], check=False)
env = dict(os.environ, PYTHONPATH=str(WORK / "src"), PYTHONUNBUFFERED="1", WSP_JOB_ID=uuid.uuid4().hex[:12])
def sh(cmd):
    print(f"\\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=WORK, env=env).returncode
status = sh(f"{sys.executable} -m wsp.tasks.build")
for cmd in COMMANDS:
    if status != 0:
        print("aborting after failure", flush=True)
        break
    status = sh(cmd.replace("python ", f"{sys.executable} ", 1))
OUT.mkdir(parents=True, exist_ok=True)
for sub in ("results", "runs", "checkpoints"):
    if (WORK / sub).exists():
        shutil.copytree(WORK / sub, OUT / sub, dirs_exist_ok=True)
(OUT / "EXIT_STATUS").write_text(str(status))
print("EXIT_STATUS", status)
'''


def kaggle_user() -> str:
    user = os.environ.get("KAGGLE_USERNAME")
    if not user:
        raise SystemExit("set KAGGLE_USERNAME to the Kaggle account that owns the kernels")
    return user


def selected_dirs(commands: list[str]) -> list[Path]:
    """Directories a job needs beyond ALWAYS_BUNDLED: run edit folders, named runs/ paths, the work folder."""
    dirs = []
    for cmd in commands:
        env = {k: v.strip("'\"") for k, v in ENV_PATTERN.findall(cmd)}
        if env.get("WSP_RUN"):
            run = ROOT / "runs" / env.get("WSP_MODEL", DEFAULT_MODEL) / env.get("WSP_TASK", DEFAULT_TASK) / env["WSP_RUN"]
            dirs.append(run / "edit")
        # A named edit path has no .py suffix, so bundle its folder (the edit's constants live beside it).
        for named in RUNS_PATH_PATTERN.findall(cmd):
            path = ROOT / named.rstrip("/.")
            dirs.append(path if path.exists() else path.parent)
    if os.environ.get("BUNDLE_WORK"):
        dirs.append(ROOT / "runs" / os.environ["BUNDLE_WORK"] / "work")
    return dirs


def bundled_files(commands: list[str]) -> list[Path]:
    roots = [ROOT / d for d in ALWAYS_BUNDLED] + selected_dirs(commands)
    candidates = set()
    for root in roots:
        candidates |= {root} if root.is_file() else set(root.rglob("*"))
    files = sorted(p for p in candidates
                   if p.is_file() and p.suffix in BUNDLED_SUFFIXES and "__pycache__" not in p.parts)
    large = [p for p in files if p.stat().st_size > MAX_FILE_BYTES]
    if large:
        names = ", ".join(str(p.relative_to(ROOT)) for p in large[:3])
        print(f"bundle: skipped {len(large)} files over {MAX_FILE_BYTES} bytes ({names})", file=sys.stderr)
    return [p for p in files if p not in large] + sorted(ROOT.glob(LEDGER_GLOB))


def archive(files: list[Path]) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    encoded = base64.b64encode(buf.getvalue()).decode()
    if len(encoded) > MAX_BUNDLED_BYTES:
        raise ValueError(f"bundle too large for Kaggle's 1 MB source limit: {len(encoded)} bytes")
    return encoded


def build(job: str, commands: list[str]) -> Path:
    slug = f"{SLUG_PREFIX}-{job}"
    out_dir = BUILD_DIR / job
    out_dir.mkdir(parents=True, exist_ok=True)
    script = (RUNNER_TEMPLATE
              .replace("__ARCHIVE__", repr(archive(bundled_files(commands))))
              .replace("__COMMANDS__", repr(commands)))
    (out_dir / f"{slug}.py").write_text(script)
    meta = {
        "id": f"{kaggle_user()}/{slug}",
        "title": slug,
        "code_file": f"{slug}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [d for d in os.environ.get("DATASET_SOURCES", "").split(",") if d],
        "kernel_sources": [k for k in os.environ.get("KERNEL_SOURCES", "").split(",") if k],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": MACHINE_SHAPE,
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("job", help="kernel name suffix: the kernel is <user>/wsp-<job>")
    p.add_argument("commands", nargs="+", help="shell commands run from the project root")
    args = p.parse_args()
    print(build(args.job, args.commands))


if __name__ == "__main__":
    main()
