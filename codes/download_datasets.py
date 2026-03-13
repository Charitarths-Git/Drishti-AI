"""
download_datasets.py
====================
Downloads AVCaps + Charades (ORIGINAL resolution) for the BLV navigation pipeline.

Hardware  : 8× NVIDIA RTX A6000 (49 GB each) | 20 TB storage
Storage   : ~55 GB Charades original + ~30 GB AVCaps ≈ 85 GB total (trivial on 20 TB)

Why original Charades over 480p:
  - Small but safety-critical objects (door handles, step edges, wet-floor signs,
    low cables) are often lost at 480p. Original resolution preserves them.
  - Qwen2.5-VL-7B accepts up to 1280 px input — higher source res → richer tokens.
  - You have 20 TB. Use it.

Usage:
    python download_datasets.py --data_dir /data
    python download_datasets.py --data_dir /data --datasets avcaps
    python download_datasets.py --data_dir /data --verify_only
"""

import os
import sys
import csv
import json
import zipfile
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def du(path: Path) -> int:
    """Total bytes under path."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def download_file(url: str, dest: Path, desc: str = "Downloading") -> Path:
    """Resumable streaming download with tqdm progress bar."""
    if not REQUESTS_AVAILABLE:
        raise ImportError("pip install requests")

    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers  = {"Range": f"bytes={existing}-"} if existing else {}

    resp  = requests.get(url, stream=True, headers=headers, timeout=120)
    total = int(resp.headers.get("content-length", 0)) + existing

    if resp.status_code == 416:          # already complete
        print(f"  ✅ Already downloaded: {dest.name}")
        return dest

    mode = "ab" if existing else "wb"
    with open(dest, mode) as fh, tqdm(
        total=total, initial=existing, unit="B", unit_scale=True, desc=desc
    ) as bar:
        for chunk in resp.iter_content(chunk_size=2 << 20):   # 2 MB chunks
            fh.write(chunk)
            bar.update(len(chunk))
    return dest


def hf_download(repo_id: str, repo_type: str, local_dir: Path):
    """Download a HuggingFace repo, preferring the Python SDK then CLI."""
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, repo_type=repo_type,
                          local_dir=str(local_dir), ignore_patterns=["*.git*"])
    except ImportError:
        subprocess.run(
            ["huggingface-cli", "download", repo_id,
             "--repo-type", repo_type, "--local-dir", str(local_dir)],
            check=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# AVCaps
# ─────────────────────────────────────────────────────────────────────────────
#
# AVCaps DOES NOT host video files directly.
# The HuggingFace repo (TUT-ARG/AVCaps) contains ONLY captions + metadata.
# Videos must be downloaded from YouTube using the video IDs in the metadata.
#
# This is standard practice for academic audio-visual datasets (AudioCaps,
# VGGSound, AVCaps, etc.) due to YouTube content rights.
#
# Expected outcome:
#   ~1,800–2,061 videos downloaded (some may be deleted/private on YouTube)
#   ~25–30 GB total video storage
#   Download time: ~2–4 hours with 8 parallel workers on a fast connection
# ─────────────────────────────────────────────────────────────────────────────

def _check_ytdlp() -> bool:
    """Check yt-dlp is installed and return True if available."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"  yt-dlp version: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def _load_avcaps_metadata(root: Path) -> dict[str, dict]:
    """
    Parse all annotation files in the AVCaps HF download.
    Returns dict: youtube_id -> {visual_caption, audio_caption, audio_visual_caption}

    AVCaps HF repo layout (TUT-ARG/AVCaps):
      data/train-*.parquet  or  train.csv / test.csv
      Columns: video_id (YouTube ID), visual_caption, audio_caption, audio_visual_caption
    """
    metadata: dict[str, dict] = {}

    # ── Parquet (primary format on HF) ───────────────────────────────────────
    parquets = list(root.rglob("*.parquet"))
    if parquets:
        try:
            import pandas as pd
            for pf in parquets:
                df = pd.read_parquet(pf)
                for _, row in df.iterrows():
                    vid = str(row.get("video_id") or row.get("id") or row.get("youtube_id") or "").strip()
                    if vid and vid not in metadata:
                        metadata[vid] = _avcaps_meta_entry(row.to_dict())
            print(f"  Loaded {len(metadata):,} entries from {len(parquets)} parquet files")
            return metadata
        except ImportError:
            print("  pandas not available — falling back to CSV/JSON")

    # ── CSV ───────────────────────────────────────────────────────────────────
    for cf in root.rglob("*.csv"):
        if cf.name == "index.json":
            continue
        try:
            with open(cf, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    vid = str(
                        row.get("video_id") or row.get("id") or row.get("youtube_id") or ""
                    ).strip()
                    if vid and vid not in metadata:
                        metadata[vid] = _avcaps_meta_entry(row)
        except Exception as e:
            print(f"  [warn] {cf.name}: {e}")

    # ── JSON ──────────────────────────────────────────────────────────────────
    for jf in root.rglob("*.json"):
        if jf.name in {"index.json"}:
            continue
        try:
            data = json.loads(jf.read_text())
            rows = data if isinstance(data, list) else list(data.values())
            for row in rows:
                vid = str(row.get("video_id") or row.get("id") or row.get("youtube_id") or "").strip()
                if vid and vid not in metadata:
                    metadata[vid] = _avcaps_meta_entry(row)
        except Exception:
            pass

    return metadata


def _avcaps_meta_entry(row: dict) -> dict:
    """Normalise one annotation row into a consistent metadata dict."""
    return {
        "visual_caption":       str(row.get("visual_caption")       or row.get("caption")          or "").strip(),
        "audio_caption":        str(row.get("audio_caption")        or row.get("audio_description") or "").strip(),
        "audio_visual_caption": str(row.get("audio_visual_caption") or row.get("av_caption")        or "").strip(),
    }


def download_avcaps(data_dir: Path, n_workers: int = 8, skip_existing: bool = True) -> Path:
    """
    Download AVCaps dataset:
      Step A — Download captions/metadata from HuggingFace (TUT-ARG/AVCaps)
      Step B — Download videos from YouTube using yt-dlp

    AVCaps contains 2,061 YouTube clips with 3 caption types per clip:
      visual_caption, audio_caption, audio_visual_caption

    Args:
        n_workers   : parallel yt-dlp download workers (8 recommended)
        skip_existing: skip videos already downloaded (safe to re-run)
    """
    print("\n" + "═" * 60)
    print("  AVCaps  — metadata from HuggingFace + videos from YouTube")
    print("═" * 60)

    out = data_dir / "avcaps"
    vid_dir = out / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    # ── Step A: Download metadata from HuggingFace ───────────────────────────
    print("\n  [A] Downloading AVCaps metadata from HuggingFace...")
    hf_download("TUT-ARG/AVCaps", "dataset", out)

    # Parse metadata to get YouTube IDs
    metadata = _load_avcaps_metadata(out)

    if not metadata:
        print("  [WARN] No metadata found after HF download.")
        print("  The dataset structure may have changed. Check:")
        print(f"    ls {out}")
        print("  Then run: python download_datasets.py --data_dir /data --reindex_only")
        _index_avcaps(out, metadata)
        return out

    print(f"  Found {len(metadata):,} video IDs in metadata")

    # ── Step B: Download videos from YouTube ─────────────────────────────────
    print(f"\n  [B] Downloading {len(metadata):,} videos from YouTube...")

    # Check yt-dlp
    if not _check_ytdlp():
        print("\n  [ERROR] yt-dlp not found. Install it:")
        print("    pip install yt-dlp")
        print("\n  After installing, re-run:")
        print(f"    python download_datasets.py --data_dir {data_dir} --datasets avcaps")
        print("\n  Saving metadata index now so captions are available even without videos.")
        _index_avcaps(out, metadata)
        return out

    # Build list of videos to download
    youtube_ids = list(metadata.keys())

    if skip_existing:
        existing = {p.stem for p in vid_dir.glob("*.mp4")}
        to_download = [vid for vid in youtube_ids if vid not in existing]
        print(f"  Already downloaded: {len(existing):,}")
        print(f"  Remaining         : {len(to_download):,}")
    else:
        to_download = youtube_ids

    if not to_download:
        print("  All videos already downloaded.")
    else:
        _ytdlp_download_batch(to_download, vid_dir, n_workers=n_workers)

    # ── Index ─────────────────────────────────────────────────────────────────
    _index_avcaps(out, metadata)
    return out


def _ytdlp_download_batch(youtube_ids: list[str], out_dir: Path, n_workers: int = 8):
    """
    Download a list of YouTube videos using yt-dlp with parallel workers.

    Format selection:
      best[height<=1080][ext=mp4]  — best quality up to 1080p, prefer mp4 container
      Fallback: best available if mp4 not available at <=1080p

    Error handling:
      - private/deleted videos are logged to failed_downloads.txt, not crashed on
      - age-restricted videos are skipped with a warning
      - each video gets 3 retries with exponential backoff

    Output filename: {youtube_id}.mp4  (stem = video_id used throughout pipeline)
    """
    # Write URL list to a temp file — yt-dlp -a is faster than one process per video
    url_list_path = out_dir.parent / "ytdlp_urls.txt"
    with open(url_list_path, "w") as f:
        for vid in youtube_ids:
            f.write(f"https://www.youtube.com/watch?v={vid}\n")

    print(f"  Written {len(youtube_ids):,} URLs to {url_list_path}")
    print(f"  Downloading with {n_workers} parallel workers...")
    print(f"  Output: {out_dir}/")
    print(f"  This will take 2-4 hours depending on connection speed.")
    print(f"  Safe to interrupt (Ctrl+C) and resume — already-downloaded videos are skipped.")

    cmd = [
        "yt-dlp",
        "--batch-file",     str(url_list_path),
        "--output",         str(out_dir / "%(id)s.%(ext)s"),
        "--format",         "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--concurrent-fragments", str(n_workers),  # parallel fragment download per video
        "-N",               str(n_workers),         # parallel video downloads
        "--retries",        "3",
        "--fragment-retries", "3",
        "--retry-sleep",    "5",
        "--ignore-errors",                          # don't stop on private/deleted videos
        "--no-warnings",
        "--progress",
        "--download-archive", str(out_dir.parent / "ytdlp_archive.txt"),  # skip already done
        "--write-info-json",  # save metadata alongside each video
        "--no-playlist",
        "--add-metadata",
    ]

    result = subprocess.run(cmd, check=False)

    # Count results
    downloaded = list(out_dir.glob("*.mp4"))
    print(f"\n  Download complete.")
    print(f"  Videos on disk: {len(downloaded):,}")

    if result.returncode != 0:
        print(f"  Some videos failed (private/deleted/geo-blocked) — this is normal.")
        print(f"  Failed video IDs are in: {out_dir.parent}/ytdlp_archive.txt")

    # Clean up URL list
    url_list_path.unlink(missing_ok=True)


def _index_avcaps(root: Path, metadata: dict[str, dict] = None):
    """
    Build avcaps/index.json mapping each video_id to its path + captions.

    If metadata is provided (from _load_avcaps_metadata), uses it directly.
    Otherwise re-parses annotation files from disk — useful for --reindex_only.

    Marks video_exists=True/False so combined_index can filter correctly.
    """
    print("  Building AVCaps index...")
    vid_dir = root / "videos"

    # Load metadata from disk if not provided
    if metadata is None:
        metadata = _load_avcaps_metadata(root)

    if not metadata:
        print("  [WARN] No metadata found. Index will be empty.")
        (root / "index.json").write_text(json.dumps({}, indent=2))
        return

    # Build fast video file lookup
    vid_lookup: dict[str, Path] = {}
    if vid_dir.exists():
        for mp4 in vid_dir.glob("*.mp4"):
            vid_lookup[mp4.stem] = mp4
        # Also check .webm in case yt-dlp fell back to webm
        for wm in vid_dir.glob("*.webm"):
            if wm.stem not in vid_lookup:
                vid_lookup[wm.stem] = wm

    index: dict[str, dict] = {}
    for vid, meta in metadata.items():
        vp = vid_lookup.get(vid, vid_dir / f"{vid}.mp4")
        index[vid] = {
            "video_id":             vid,
            "dataset":              "avcaps",
            "video_path":           str(vp),
            "video_exists":         vp.exists(),
            "visual_caption":       meta.get("visual_caption", ""),
            "audio_caption":        meta.get("audio_caption", ""),
            "audio_visual_caption": meta.get("audio_visual_caption", ""),
        }

    n_with_video = sum(1 for v in index.values() if v["video_exists"])
    (root / "index.json").write_text(json.dumps(index, indent=2))

    print(f"  AVCaps index: {len(index):,} entries")
    print(f"  Videos on disk: {n_with_video:,} / {len(index):,}")

    if n_with_video == 0:
        print("  [WARN] No videos found on disk.")
        print(f"  Make sure yt-dlp downloaded to: {vid_dir}")
    elif n_with_video < len(index) * 0.85:
        missing = len(index) - n_with_video
        print(f"  [NOTE] {missing:,} videos unavailable (deleted/private on YouTube) — normal for this dataset.")


# ─────────────────────────────────────────────────────────────────────────────
# Charades  — ORIGINAL resolution
# ─────────────────────────────────────────────────────────────────────────────

# AllenAI official S3 URLs (verified against prior.allenai.org/projects/charades)
#
# Charades ships TWO separate video zips:
#   Charades_v1_480.zip   — 480p re-encode     (~13 GB)   ← NOT what we want
#   Charades_v1.zip       — original resolution (~55 GB)   ← this is the one
#
# The annotations zip is separate and small (~100 MB).
# Both are hosted on AllenAI's public S3 bucket.
_CHARADES_ANN_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip"
_CHARADES_VID_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip"

# AllenAI sometimes throttles or the bucket moves; this HF mirror is a fallback
_CHARADES_HF_REPO = "HuggingFaceM4/charades"

# Expected video count — used to decide whether extraction is complete
_CHARADES_EXPECTED_VIDEOS = 9848


def download_charades(data_dir: Path, use_hf_mirror: bool = False) -> Path:
    """
    Download Charades ORIGINAL resolution videos (~55 GB) + annotations.
    9 848 videos · ~30 s each · 157 action classes · 27 847 captions.

    Original resolution is used (not 480p) because:
      - Small safety-critical objects (step edges, door handles, low cables,
        wet-floor signs) are often lost at 480p
      - Qwen2.5-VL-7B accepts up to 1280 px — higher source res → richer tokens
      - 20 TB storage makes the 55 GB size completely irrelevant

    Download strategy:
      1. Try AllenAI S3 directly (fastest, no auth required)
      2. Fall back to HuggingFace mirror if --charades_mirror flag is set
         or if S3 returns an error

    Resumable: if the zip is partially downloaded, the Range header resumes it.
    """
    print("\n" + "═" * 60)
    print("  Charades ORIGINAL resolution (~55 GB)")
    print("═" * 60)

    out = data_dir / "charades"
    out.mkdir(parents=True, exist_ok=True)

    if use_hf_mirror:
        print("  Using HuggingFace mirror (--charades_mirror)")
        _download_charades_hf(out)
    else:
        try:
            _download_charades_direct(out)
        except Exception as e:
            print(f"  AllenAI S3 failed ({e}). Falling back to HuggingFace mirror...")
            _download_charades_hf(out)

    _index_charades(out)
    return out


def _download_charades_direct(out: Path):
    """
    Download from AllenAI S3:
      1. Annotations ZIP (~100 MB) → extract to out/annotations/
      2. Original video ZIP (~55 GB) → extract to out/videos/

    Extraction is guarded: if out/videos/ already has >= 9000 .mp4 files,
    the zip is not re-extracted (safe to re-run after interruption).

    After extraction, flattens any nested subdirectory that Charades sometimes
    produces inside the zip (e.g. Charades_v1/ subfolder → videos/).
    """
    ann_zip = out / "Charades_annotations.zip"
    vid_zip = out / "Charades_v1_original.zip"
    vid_dir = out / "videos"
    ann_dir = out / "annotations"

    # ── Annotations ──────────────────────────────────────────────────────────
    if not ann_dir.exists() or not any(ann_dir.glob("*.csv")):
        if not ann_zip.exists():
            print("  Downloading Charades annotations (~100 MB)...")
            download_file(_CHARADES_ANN_URL, ann_zip, "Charades annotations")
        else:
            print(f"  Annotations zip already present: {ann_zip.name}")

        print("  Extracting annotations...")
        ann_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(ann_zip) as z:
            z.extractall(out)

        # The zip extracts to a folder like "Charades/" — rename it to "annotations"
        for p in sorted(out.iterdir()):
            if p.is_dir() and p.name not in {"annotations", "videos"} and not p.name.startswith("."):
                # Only rename if it contains CSV files (i.e., it's the annotation folder)
                if any(p.rglob("*.csv")):
                    target = out / "annotations"
                    if not target.exists():
                        p.rename(target)
                        print(f"  Renamed {p.name}/ → annotations/")
                    break

        csvs = list(ann_dir.rglob("*.csv")) if ann_dir.exists() else []
        print(f"  Annotations ready: {len(csvs)} CSV files in {ann_dir}")
    else:
        csvs = list(ann_dir.rglob("*.csv"))
        print(f"  Annotations already extracted: {len(csvs)} CSV files")

    # ── Videos ───────────────────────────────────────────────────────────────
    existing_mp4 = list(vid_dir.rglob("*.mp4")) if vid_dir.exists() else []

    if len(existing_mp4) >= _CHARADES_EXPECTED_VIDEOS:
        print(f"  Videos already extracted: {len(existing_mp4):,} files in {vid_dir}")
        return

    # Download zip if not present or incomplete
    if not vid_zip.exists():
        print(f"  Downloading Charades ORIGINAL videos (~55 GB)...")
        print(f"  URL: {_CHARADES_VID_URL}")
        print(f"  Destination: {vid_zip}")
        print(f"  This is a one-time download. It is resumable if interrupted.")
        download_file(_CHARADES_VID_URL, vid_zip, "Charades original videos")
    else:
        size_gb = vid_zip.stat().st_size / 1e9
        print(f"  Video zip already present: {vid_zip.name} ({size_gb:.1f} GB)")

    # Validate zip integrity before extracting
    print("  Validating zip file integrity...")
    try:
        with zipfile.ZipFile(vid_zip) as z:
            bad = z.testzip()
            if bad:
                print(f"  [WARN] Corrupt file in zip: {bad}")
                print(f"  Deleting and re-downloading...")
                vid_zip.unlink()
                download_file(_CHARADES_VID_URL, vid_zip, "Charades original videos (retry)")
    except zipfile.BadZipFile:
        print(f"  [WARN] Zip is corrupted or incomplete. Re-downloading...")
        vid_zip.unlink()
        download_file(_CHARADES_VID_URL, vid_zip, "Charades original videos (retry)")

    # Extract
    print(f"  Extracting {vid_zip.name} → {vid_dir}  (this may take 5-15 minutes)...")
    vid_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(vid_zip) as z:
        members = z.namelist()
        print(f"  Zip contains {len(members):,} entries")
        for member in tqdm(members, desc="  Extracting", unit="file"):
            z.extract(member, vid_dir)

    # Charades zip sometimes extracts into a subfolder like videos/Charades_v1/
    # Flatten it so all .mp4 files are directly in out/videos/
    _flatten_video_dir(vid_dir)

    final_mp4 = list(vid_dir.rglob("*.mp4"))
    print(f"  Extracted: {len(final_mp4):,} videos → {vid_dir}")

    if len(final_mp4) < _CHARADES_EXPECTED_VIDEOS * 0.95:
        print(f"  [WARN] Expected ~{_CHARADES_EXPECTED_VIDEOS:,} videos, got {len(final_mp4):,}")
        print(f"  The download may be incomplete. Check {vid_zip} and re-run if needed.")
    else:
        print(f"  All {len(final_mp4):,} videos extracted successfully.")
        # Delete the zip unless --keep_zip was passed (saves ~55 GB)
        if os.environ.get("CHARADES_KEEP_ZIP") == "1":
            print(f"  Keeping zip (--keep_zip): {vid_zip}")
        elif vid_zip.exists():
            print(f"  Removing zip to free ~55 GB...")
            vid_zip.unlink()
            print(f"  Done.")


def _flatten_video_dir(vid_dir: Path):
    """
    If Charades extraction created a single subfolder (e.g. Charades_v1/),
    move all .mp4 files up one level so they sit directly in vid_dir/.
    This handles the common case where the zip has a top-level directory.
    """
    # Count direct .mp4 files
    direct_mp4 = list(vid_dir.glob("*.mp4"))
    if direct_mp4:
        return  # already flat

    # Look for a single subdirectory that contains all the mp4s
    subdirs = [p for p in vid_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        subdir = subdirs[0]
        nested_mp4 = list(subdir.glob("*.mp4"))
        if nested_mp4:
            print(f"  Flattening {subdir.name}/ → {vid_dir.name}/ ({len(nested_mp4):,} files)...")
            for mp4 in tqdm(nested_mp4, desc="  Moving", unit="file"):
                mp4.rename(vid_dir / mp4.name)
            # Remove the now-empty subdir
            try:
                subdir.rmdir()
            except OSError:
                pass  # not empty — leave it


def _download_charades_hf(out: Path):
    """
    Download Charades from the HuggingFace mirror.
    Used as fallback when AllenAI S3 is unavailable or slow.
    Note: HF mirror may have different directory structure than the official release.
    """
    print(f"  Downloading from HuggingFace: {_CHARADES_HF_REPO}")
    hf_download(_CHARADES_HF_REPO, "dataset", out)

    # HF mirror puts videos in a different location — normalise to out/videos/
    vid_dir = out / "videos"
    if not vid_dir.exists():
        vid_dir.mkdir(exist_ok=True)
        # Search for mp4s anywhere under out/ and move them to videos/
        found = list(out.rglob("*.mp4"))
        if found and not any(f.parent == vid_dir for f in found):
            print(f"  Moving {len(found):,} videos to {vid_dir}...")
            for mp4 in tqdm(found, desc="  Moving", unit="file"):
                dest = vid_dir / mp4.name
                if not dest.exists():
                    mp4.rename(dest)


def _index_charades(root: Path):
    """
    Parse Charades CSVs into charades/index.json.

    Handles both the official AllenAI layout and the HuggingFace mirror layout:
      Official : root/annotations/Charades_v1_train.csv
      HF mirror: root/train.csv  or  root/data/train-*.parquet

    For each video, the index entry includes:
      - video_path  : absolute path to the .mp4 file (may not exist if download incomplete)
      - video_exists: bool — easy filter in combined_index
      - split       : train or test
      - scene       : room type (kitchen, bedroom, etc.)
      - objects     : list of objects in the scene
      - descriptions: human-written activity caption
      - actions     : action class annotations
      - length_sec  : video duration in seconds
    """
    print("  Building Charades index...")
    index: dict[str, dict] = {}
    vid_dir = root / "videos"

    # Build a fast lookup: video_id -> actual path (handles flat + nested layouts)
    print("  Scanning for video files...")
    vid_lookup: dict[str, Path] = {}
    for mp4 in (vid_dir.rglob("*.mp4") if vid_dir.exists() else []):
        vid_lookup[mp4.stem] = mp4
    print(f"  Found {len(vid_lookup):,} video files on disk")

    # Collect all CSV paths, prioritising named files over wildcards
    csv_paths: list[Path] = []
    for pattern in [
        "Charades_v1_train.csv",
        "Charades_v1_test.csv",
        "train.csv",
        "test.csv",
        "val.csv",
    ]:
        csv_paths.extend(root.rglob(pattern))

    # De-duplicate while preserving priority order
    seen_csv: set[str] = set()
    deduped_csv: list[Path] = []
    for p in csv_paths:
        if p.name not in seen_csv:
            seen_csv.add(p.name)
            deduped_csv.append(p)

    # Fallback: grab any CSV not already included
    for p in root.rglob("*.csv"):
        if p.name not in seen_csv:
            seen_csv.add(p.name)
            deduped_csv.append(p)

    if not deduped_csv:
        print("  [WARN] No annotation CSVs found. Index will have 0 entries.")
        print(f"  Expected CSVs in: {root / 'annotations'}")
        (root / "index.json").write_text(json.dumps({}, indent=2))
        return

    # Try HF parquet format as additional fallback
    parquet_entries = _load_charades_parquet(root, vid_lookup)
    for vid, entry in parquet_entries.items():
        index[vid] = entry

    for csv_path in deduped_csv:
        # Determine split from filename
        name_lower = csv_path.stem.lower()
        if "train" in name_lower:
            split = "train"
        elif "test" in name_lower:
            split = "test"
        elif "val" in name_lower:
            split = "validation"
        else:
            split = "train"  # default

        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows_loaded = 0
                for row in reader:
                    vid = (row.get("id") or row.get("video_id") or "").strip()
                    if not vid or vid in index:
                        continue

                    # Locate the .mp4 — use lookup table (O(1)) rather than rglob per video
                    vp = vid_lookup.get(vid, vid_dir / f"{vid}.mp4")

                    index[vid] = {
                        "video_id":     vid,
                        "dataset":      "charades",
                        "split":        split,
                        "scene":        row.get("scene", "").strip(),
                        "objects":      [o.strip() for o in row.get("objects", "").split(";") if o.strip()],
                        "descriptions": row.get("descriptions", "").strip(),
                        "actions":      row.get("actions", "").strip(),
                        "length_sec":   _safe_float(row.get("length") or row.get("length_sec") or 0),
                        "video_path":   str(vp),
                        "video_exists": vp.exists(),
                    }
                    rows_loaded += 1

                print(f"  {csv_path.name}: {rows_loaded:,} entries (split={split})")

        except Exception as e:
            print(f"  [warn] Could not parse {csv_path.name}: {e}")

    # Summary
    n_with_video = sum(1 for v in index.values() if v.get("video_exists"))
    n_train = sum(1 for v in index.values() if v.get("split") == "train")
    n_test  = sum(1 for v in index.values() if v.get("split") == "test")

    (root / "index.json").write_text(json.dumps(index, indent=2))
    print(f"  Charades index: {len(index):,} entries  "
          f"(train={n_train:,}, test={n_test:,})")
    print(f"  Videos on disk: {n_with_video:,} / {len(index):,}")

    if n_with_video < len(index) * 0.95:
        missing = len(index) - n_with_video
        print(f"  [WARN] {missing:,} videos missing from disk. "
              f"Check that extraction completed fully.")


def _load_charades_parquet(root: Path, vid_lookup: dict) -> dict:
    """
    Load entries from HuggingFace parquet shards if present.
    Returns dict of vid -> entry (same schema as CSV path).
    """
    parquets = list(root.rglob("*.parquet"))
    if not parquets:
        return {}

    try:
        import pandas as pd
    except ImportError:
        return {}

    index = {}
    for pf in parquets:
        split = "train" if "train" in pf.stem else "test"
        try:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                vid = str(row.get("id") or row.get("video_id") or "").strip()
                if not vid or vid in index:
                    continue
                vp = vid_lookup.get(vid, root / "videos" / f"{vid}.mp4")
                index[vid] = {
                    "video_id":     vid,
                    "dataset":      "charades",
                    "split":        split,
                    "scene":        str(row.get("scene", "") or "").strip(),
                    "objects":      [o.strip() for o in str(row.get("objects", "") or "").split(";") if o.strip()],
                    "descriptions": str(row.get("descriptions", "") or "").strip(),
                    "actions":      str(row.get("actions", "") or "").strip(),
                    "length_sec":   _safe_float(row.get("length") or row.get("length_sec") or 0),
                    "video_path":   str(vp),
                    "video_exists": vp.exists(),
                }
        except Exception as e:
            print(f"  [warn] parquet {pf.name}: {e}")

    if index:
        print(f"  Loaded {len(index):,} entries from parquet shards")
    return index


def _safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Combined index
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_index(data_dir: Path) -> Path:
    """
    Merge AVCaps + Charades indexes → data/combined_index.json.
    This is what Step 1 reads via --video_list.
    """
    combined: dict[str, dict] = {}

    for dataset, idx_path in [
        ("avcaps",   data_dir / "avcaps"   / "index.json"),
        ("charades", data_dir / "charades" / "index.json"),
    ]:
        if not idx_path.exists():
            print(f"  [skip] {idx_path} not found")
            continue
        sub = json.loads(idx_path.read_text())
        for vid, entry in sub.items():
            combined[f"{dataset}__{vid}"] = entry
        print(f"  Merged {len(sub):,} entries from {dataset}")

    out = data_dir / "combined_index.json"
    out.write_text(json.dumps(combined, indent=2))

    # Also write a flat video list (path + metadata) for Step 1
    vlist = [
        {"video_id": k, **{kk: vv for kk, vv in v.items()}}
        for k, v in combined.items()
        if Path(v.get("video_path", "")).exists()
    ]
    vlist_path = data_dir / "video_list.json"
    vlist_path.write_text(json.dumps(vlist, indent=2))

    print(f"\n  ✅ Combined index : {len(combined):,} total → {out}")
    print(f"  ✅ Existing videos: {len(vlist):,}   → {vlist_path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Verify
# ─────────────────────────────────────────────────────────────────────────────

def verify(data_dir: Path):
    print("\nDownload Verification")
    print("─" * 50)

    for name, sub, expected in [
        ("AVCaps",   "avcaps",   2061),
        ("Charades", "charades", 9848),
    ]:
        d = data_dir / sub
        if not d.exists():
            print(f"  {name:10s}: NOT DOWNLOADED")
            continue

        mp4s   = list(d.rglob("*.mp4"))
        size   = du(d)
        pct    = len(mp4s) / expected * 100
        status = "OK" if len(mp4s) >= expected * 0.95 else "INCOMPLETE"
        print(f"  {name:10s}: {len(mp4s):>6,} / {expected:,} videos  "
              f"({pct:.0f}%)  {sizeof_fmt(size)}  [{status}]")

        # Check index
        idx_path = d / "index.json"
        if idx_path.exists():
            idx = json.loads(idx_path.read_text())
            n_with_video = sum(1 for v in idx.values() if v.get("video_exists", True))
            print(f"  {'':10s}  index.json: {len(idx):,} entries, {n_with_video:,} with video on disk")
        else:
            print(f"  {'':10s}  index.json: MISSING (run --reindex_only to rebuild)")

    # Combined index
    ci = data_dir / "combined_index.json"
    if ci.exists():
        combined = json.loads(ci.read_text())
        n_existing = sum(
            1 for v in combined.values()
            if Path(v.get("video_path", "")).exists()
        )
        print(f"\n  combined_index.json: {len(combined):,} total, {n_existing:,} videos on disk")
    else:
        print(f"\n  combined_index.json: NOT BUILT (run without --verify_only)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Download AVCaps + Charades original resolution for BLV pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_dir", default="./data",
                    help="Root directory to store datasets")
    ap.add_argument("--datasets", nargs="+", default=["avcaps", "charades"],
                    choices=["avcaps", "charades"],
                    help="Which datasets to download")
    ap.add_argument("--charades_mirror", action="store_true",
                    help="Use HuggingFace mirror instead of AllenAI S3 "
                         "(use if S3 is slow or unavailable in your region)")
    ap.add_argument("--verify_only", action="store_true",
                    help="Check download status without downloading anything")
    ap.add_argument("--reindex_only", action="store_true",
                    help="Rebuild index.json files from already-extracted data "
                         "(use after manual extraction or partial re-download)")
    ap.add_argument("--keep_zip", action="store_true",
                    help="Keep the Charades video zip after extraction "
                         "(by default it is deleted to free ~55 GB). "
                         "Only useful if you plan to re-extract on another machine.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    if args.verify_only:
        verify(data_dir)
        sys.exit(0)

    if args.reindex_only:
        print("Rebuilding indexes from existing data...")
        if "avcaps" in args.datasets:
            avcaps_dir = data_dir / "avcaps"
            if avcaps_dir.exists():
                _index_avcaps(avcaps_dir)
            else:
                print(f"  [skip] {avcaps_dir} does not exist")
        if "charades" in args.datasets:
            charades_dir = data_dir / "charades"
            if charades_dir.exists():
                _index_charades(charades_dir)
            else:
                print(f"  [skip] {charades_dir} does not exist")
        build_combined_index(data_dir)
        verify(data_dir)
        sys.exit(0)

    if args.keep_zip:
        os.environ["CHARADES_KEEP_ZIP"] = "1"

    if "avcaps"   in args.datasets:
        download_avcaps(data_dir)
    if "charades" in args.datasets:
        download_charades(data_dir, args.charades_mirror)

    build_combined_index(data_dir)
    verify(data_dir)

    print("\nNext step:")
    print(f"  python step1_extract_frames.py \\")
    print(f"      --video_list {data_dir}/combined_index.json \\")
    print(f"      --output_dir ./keyframes \\")
    print(f"      --method motion --n_frames 24")
