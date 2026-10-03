"""Fetch the corrected extrinsics and a sample of raw DROID episodes.

Both sources are public, no account needed:
  corrected extrinsics   https://huggingface.co/KarlP/droid
  raw episodes           gs://gresearch/robotics/droid_raw/1.0.1/  (read via the public HTTPS endpoint)

Usage
  python -m kintrace.droid download --out data/droid --episodes 50
  python -m kintrace.droid download --out data/droid --episodes 50 --video   # also the fixed-camera MP4s (~100 MB each)

Only standard library here so it runs anywhere.
"""
from __future__ import annotations

import json
import os
import random
import sys
import urllib.parse
import urllib.request

HF_BASE = "https://huggingface.co/KarlP/droid/resolve/main/"
HF_FILES = ["cam2base_extrinsics.json", "episode_id_to_path.json", "intrinsics.json"]
GCS_BUCKET = "gresearch"
GCS_PREFIX = "robotics/droid_raw/1.0.1/"
GCS_LIST = "https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={prefix}&fields=items(name,size),nextPageToken"
GCS_GET = "https://storage.googleapis.com/{bucket}/{name}"


def _fetch(url: str, dest: str, quiet: bool = False) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if not quiet and total:
                print(f"\r  {os.path.basename(dest)}  {got/1e6:.0f}/{total/1e6:.0f} MB", end="", file=sys.stderr)
    if not quiet and total:
        print(file=sys.stderr)
    os.replace(tmp, dest)
    return dest


def fetch_extrinsics(out: str) -> dict:
    paths = {}
    for name in HF_FILES:
        print(f"fetching {name}", file=sys.stderr)
        paths[name] = _fetch(HF_BASE + name, os.path.join(out, name))
    return paths


def list_episode_files(episode_path: str) -> list:
    prefix = GCS_PREFIX + episode_path.strip("/") + "/"
    items, token = [], None
    while True:
        url = GCS_LIST.format(bucket=GCS_BUCKET, prefix=urllib.parse.quote(prefix, safe=""))
        if token:
            url += "&pageToken=" + token
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        items += page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            break
    return items


def fetch_episode(episode_path: str, out: str, video: bool = False, fixed_serials=()) -> str:
    # episode folders are named like Sun_Jun__4_20:04:15_2023; ':' is not allowed on Windows
    dest_dir = os.path.join(out, "episodes", episode_path.strip("/").replace(":", "_"))
    files = list_episode_files(episode_path)
    if not files:
        raise FileNotFoundError(f"no files under gs://{GCS_BUCKET}/{GCS_PREFIX}{episode_path}")
    for it in files:
        name = it["name"]
        rel = name[len(GCS_PREFIX) + len(episode_path.strip("/")) + 1:]
        base = os.path.basename(rel)
        want = base == "trajectory.h5" or (base.startswith("metadata_") and base.endswith(".json"))
        if video and rel.startswith("recordings/MP4/") and base.endswith(".mp4"):
            serial = base[:-4]
            want = not fixed_serials or serial in fixed_serials
        if not want:
            continue
        url = GCS_GET.format(bucket=GCS_BUCKET, name=urllib.parse.quote(name, safe="/"))
        _fetch(url, os.path.join(dest_dir, rel))
    return dest_dir


def choose_episodes(out: str, n: int, lab: str | None = None, seed: int = 0,
                    source: str = "any", exclude=()) -> list:
    """Episode paths that have corrected extrinsics, so every one is labelled.

    source: "GT", "Pred" or "any", matched against the entry's source field.
    exclude: keys already downloaded; they are not chosen again.
    """
    with open(os.path.join(out, "cam2base_extrinsics.json"), encoding="utf-8") as f:
        corrected = json.load(f)
    id_to_path = {}
    p = os.path.join(out, "episode_id_to_path.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            id_to_path = json.load(f)
    paths = []
    exclude = set(exclude)
    for key, entry in corrected.items():
        if key in exclude:
            continue
        if source != "any" and (entry.get("source") if isinstance(entry, dict) else None) != source:
            continue
        path = id_to_path.get(key, key)
        if lab and lab.lower() not in path.lower():
            continue
        paths.append((key, path))
    random.Random(seed).shuffle(paths)
    return paths[:n]


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m kintrace.droid download")
    ap.add_argument("--out", default="data/droid")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--lab", default=None, help="only episodes whose path contains this (e.g. AUTOLab, ILIAD)")
    ap.add_argument("--video", action="store_true", help="also download the fixed-camera MP4s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=("GT", "Pred", "any"), default="any",
                    help="only entries DROID kept (GT) or re-solved (Pred)")
    a = ap.parse_args(argv)

    fetch_extrinsics(a.out)
    # append to an existing manifest; never re-download what is already there
    mpath = os.path.join(a.out, "manifest.json")
    manifest = []
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)
    have = {m["key"] for m in manifest}
    chosen = choose_episodes(a.out, a.episodes, a.lab, a.seed, a.source, exclude=have)
    print(f"{len(chosen)} new episodes with corrected extrinsics selected ({a.source}), {len(manifest)} already in the manifest", file=sys.stderr)
    for i, (key, path) in enumerate(chosen, 1):
        print(f"[{i}/{len(chosen)}] {path}", file=sys.stderr)
        try:
            d = fetch_episode(path, a.out)
            manifest.append({"key": key, "path": path, "dir": d})
        except Exception as e:  # keep going, report at the end
            print(f"  skipped: {e}", file=sys.stderr)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if a.video:
        # fixed cameras only, for every episode in the manifest; files already on disk are skipped
        for i, m in enumerate(manifest, 1):
            print(f"[video {i}/{len(manifest)}] {m['path']}", file=sys.stderr)
            try:
                fetch_episode(m["path"], a.out, video=True, fixed_serials=_fixed_serials(m["dir"]))
            except Exception as e:
                print(f"  video skipped: {e}", file=sys.stderr)
    print(f"wrote {mpath} ({len(manifest)} episodes)")


def _fixed_serials(episode_dir: str) -> tuple:
    import glob
    for p in glob.glob(os.path.join(episode_dir, "metadata_*.json")):
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
        return tuple(str(meta[k]) for k in ("ext1_cam_serial", "ext2_cam_serial") if k in meta)
    return ()


if __name__ == "__main__":
    main()
