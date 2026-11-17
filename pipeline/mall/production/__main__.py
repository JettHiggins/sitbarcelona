"""Install the production Mall checkpoints into the model cache.

    python -m pipeline.mall.production            # verify hashes, copy into pipeline/cache/mall/models
    python -m pipeline.mall.production --check    # verify hashes only

After installing, `python -m pipeline.run mall --mode model` loads these checkpoints instead of
training, and does not need the ten benchmark games (see README.md in this folder).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

from ..common import CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(CACHE, "models")
INSTALLED = os.path.join(MODELS, "production.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify the bundled files only")
    args = parser.parse_args()
    with open(os.path.join(HERE, "MANIFEST.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    for name, info in manifest["models"].items():
        got = sha256(os.path.join(HERE, name))
        if got != info["sha256"]:
            sys.exit("{}: sha256 {} does not match the manifest ({})".format(name, got, info["sha256"]))
        print("ok", name)
    if args.check:
        return
    os.makedirs(MODELS, exist_ok=True)
    for name in manifest["models"]:
        dst = os.path.join(MODELS, name)
        if os.path.isfile(dst) and sha256(dst) != manifest["models"][name]["sha256"]:
            os.replace(dst, dst + ".replaced")
            print("kept the differing cached file as", os.path.relpath(dst + ".replaced"))
        shutil.copyfile(os.path.join(HERE, name), dst)
        print("installed", os.path.relpath(dst))
    with open(INSTALLED, "w", encoding="utf-8") as f:
        json.dump({"train_tags": manifest["train_tags"]}, f, indent=2)


if __name__ == "__main__":
    main()
