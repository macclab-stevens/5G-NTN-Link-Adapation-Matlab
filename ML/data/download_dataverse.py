#!/usr/bin/env python3
"""
Harvard Dataverse downloader using pooch.

Supports:
- Direct URL via --url
- File ID via --file-id
- File persistent ID via --file-pid (e.g., doi:...)
- Dataset DOI + filename via --dataset-doi and --filename
- Download ALL files from a dataset DOI with --all, in parallel
  and optionally extract archives in parallel

By default, downloads into this `ML/data/` folder.

Examples:

  # Download by direct URL
  python ML/data/download_dataverse.py \
    --url "https://dataverse.harvard.edu/api/access/datafile/1234567?format=original" \
    --dest ML/data/

  # Download a specific file from the dataset by DOI and filename
  python ML/data/download_dataverse.py \
    --dataset-doi "doi:10.7910/DVN/BXBOTB" \
    --filename "Case9_MCS_ThroughputCalulation_BLERw50Tbler0.01_240601_124717.csv"

  # Download all files from the dataset (sequential download) and optionally extract archives
  python ML/data/download_dataverse.py \
    --dataset-doi "doi:10.7910/DVN/BXBOTB" \
    --all --extract --extract-to ML/data/

  # Download by file persistent ID
  python ML/data/download_dataverse.py \
    --file-pid "doi:10.7910/DVN/XXXXX/ABCDEFG" 

  # Download by file numeric ID
  python ML/data/download_dataverse.py --file-id 1234567

If you know the file hash, pass --hash (supports "md5:<hex>" or "sha256:<hex>").
If not provided, integrity check is skipped ("unverified")).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os
import shutil
import gzip
import bz2
import lzma
# No parallelism needed; downloads and extraction are sequential
from typing import Optional

import pooch

try:
    import requests
except Exception:
    requests = None  # Only needed for dataset lookup by filename


DEFAULT_BASE_URL = "https://dataverse.harvard.edu"


def build_download_url(
    *,
    base_url: str,
    url: Optional[str] = None,
    file_id: Optional[int] = None,
    file_pid: Optional[str] = None,
    dataset_doi: Optional[str] = None,
    filename: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Build a direct download URL for the Dataverse file.

    Returns (url, suggested_fname).
    """
    if url:
        return url, None

    if file_id is not None:
        return f"{base_url}/api/access/datafile/{file_id}?format=original", None

    if file_pid:
        from urllib.parse import quote

        quoted = quote(file_pid, safe="")
        return (
            f"{base_url}/api/access/datafile/:persistentId?persistentId={quoted}&format=original",
            None,
        )

    if dataset_doi and filename:
        if requests is None:
            raise RuntimeError(
                "requests is required for dataset lookups (pip install requests)"
            )
        from urllib.parse import quote

        # List files in the latest dataset version and match by label
        files_ep = (
            f"{base_url}/api/datasets/:persistentId/versions/:latest/files?persistentId="
            f"{quote(dataset_doi, safe='')}"
        )
        resp = requests.get(files_ep, timeout=60)
        try:
            resp.raise_for_status()
        except Exception:
            raise RuntimeError(
                f"Failed to list dataset files. HTTP {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        entries = data if isinstance(data, list) else data.get("data", [])
        if not entries:
            raise RuntimeError("No files found for dataset or unexpected API response.")

        match = None
        for e in entries:
            # Many Dataverse instances return a list of objects containing `label` and `dataFile`
            label = e.get("label") or e.get("filename")
            if label == filename:
                match = e
                break

        if match is None:
            # Try a case-insensitive match as a fallback
            for e in entries:
                label = e.get("label") or e.get("filename")
                if label and label.lower() == filename.lower():
                    match = e
                    break

        if match is None:
            available = ", ".join((e.get("label") or e.get("filename") or "?") for e in entries[:20])
            raise RuntimeError(
                f"Filename not found in dataset: {filename}. A few available: {available}"
            )

        df = match.get("dataFile") or {}
        fid = df.get("id")
        if not fid:
            # Some responses might use different keys
            fid = match.get("id")
        if not fid:
            raise RuntimeError("Could not determine file id for the selected filename.")

        return (
            f"{base_url}/api/access/datafile/{fid}?format=original",
            # Use the dataset-provided label as the local filename
            match.get("label") or filename,
        )

    raise ValueError(
        "Provide one of: --url, --file-id, --file-pid, or --dataset-doi + --filename"
    )


def _normalize_known_hash(value: Optional[str]) -> Optional[str]:
    """Map common skip markers to None for Pooch hash verification.

    Pooch treats known_hash=None as "don't verify" and logs the SHA256.
    Some environments may pass markers like 'unverified' which older Pooch
    versions don't recognize. Normalize those to None.
    """
    if value is None:
        return None
    marker = str(value).strip().lower()
    if marker in {"", "none", "unverified", "skip", "nohash", "false"}:
        return None
    return value


def list_dataset_files(base_url: str, dataset_doi: str) -> list[dict]:
    """Return list of file entries for the dataset latest version.

    Each entry typically has keys like 'label' and nested 'dataFile': {'id': ...}.
    """
    if requests is None:
        raise RuntimeError("requests is required for dataset lookups (pip install requests)")
    from urllib.parse import quote

    files_ep = (
        f"{base_url}/api/datasets/:persistentId/versions/:latest/files?persistentId="
        f"{quote(dataset_doi, safe='')}"
    )
    resp = requests.get(files_ep, timeout=120)
    try:
        resp.raise_for_status()
    except Exception:
        raise RuntimeError(
            f"Failed to list dataset files. HTTP {resp.status_code}: {resp.text[:300]}"
        )
    data = resp.json()
    entries = data if isinstance(data, list) else data.get("data", [])
    if not entries:
        raise RuntimeError("No files found for dataset or unexpected API response.")
    return entries


def sanitize_filename(name: str) -> str:
    # Avoid directory traversal and normalize
    return Path(name).name


def is_tar_or_zip(path: Path) -> bool:
    lower = path.name.lower()
    return (
        lower.endswith(".zip")
        or lower.endswith(".tar")
        or lower.endswith(".tar.gz")
        or lower.endswith(".tgz")
        or lower.endswith(".tar.bz2")
        or lower.endswith(".tbz2")
        or lower.endswith(".tar.xz")
        or lower.endswith(".txz")
    )


def is_single_compressed(path: Path) -> bool:
    lower = path.name.lower()
    # Single-file compression (not archives)
    return lower.endswith(".gz") or lower.endswith(".bz2") or lower.endswith(".xz")


def extract_archive(archive_path: Path, extract_to: Path) -> list[Path]:
    """Extract archive or compressed file to extract_to.

    Returns list of extracted file paths. For non-archives, returns [].
    """
    extracted: list[Path] = []
    extract_to.mkdir(parents=True, exist_ok=True)

    # First try known archive formats (zip/tar.*)
    if is_tar_or_zip(archive_path):
        try:
            shutil.unpack_archive(str(archive_path), str(extract_to))
            # Attempt to list extracted files (best-effort): walk immediate dir
            for root, _, files in os.walk(extract_to):
                for f in files:
                    extracted.append(Path(root) / f)
            return extracted
        except Exception as exc:
            raise RuntimeError(f"Failed to extract archive {archive_path.name}: {exc}")

    # Handle single compressed files (.gz/.bz2/.xz)
    if archive_path.suffix.lower() == ".gz" and not archive_path.name.lower().endswith(".tar.gz"):
        out_path = extract_to / archive_path.with_suffix("").name
        try:
            with gzip.open(archive_path, "rb") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out_path)
            return extracted
        except Exception as exc:
            raise RuntimeError(f"Failed to gunzip {archive_path.name}: {exc}")

    if archive_path.suffix.lower() == ".bz2" and not archive_path.name.lower().endswith(".tar.bz2"):
        out_path = extract_to / archive_path.with_suffix("").name
        try:
            with bz2.open(archive_path, "rb") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out_path)
            return extracted
        except Exception as exc:
            raise RuntimeError(f"Failed to bunzip2 {archive_path.name}: {exc}")

    if archive_path.suffix.lower() == ".xz" and not archive_path.name.lower().endswith(".tar.xz"):
        out_path = extract_to / archive_path.with_suffix("").name
        try:
            with lzma.open(archive_path, "rb") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out_path)
            return extracted
        except Exception as exc:
            raise RuntimeError(f"Failed to unxz {archive_path.name}: {exc}")

    return extracted


def _download_one(
    url: str,
    dest_dir: Path,
    fname: Optional[str],
    known_hash: str,
    show_progress: bool,
    extract: bool,
    extract_to: Path,
    keep_archives: bool,
) -> tuple[str, Optional[str]]:
    """Download a single file, optionally extract if an archive.

    Returns (local_path, error_message). error_message is None on success.
    """
    downloader = pooch.HTTPDownloader(progressbar=show_progress)
    try:
        local_path = pooch.retrieve(
            url=url,
            known_hash=_normalize_known_hash(known_hash),
            fname=fname,
            path=dest_dir,
            downloader=downloader,
        )
        local = Path(local_path)
        if extract:
            try:
                extracted = extract_archive(local, extract_to)
                if extracted:
                    if not keep_archives:
                        try:
                            local.unlink()
                        except Exception:
                            pass
            except Exception as exc:
                return str(local), f"extract failed: {exc}"
        return str(local), None
    except Exception as exc:
        return "", f"download failed: {exc}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = parser.add_argument_group("Source selectors")
    src.add_argument("--url", help="Direct Dataverse download URL (e.g., /api/access/datafile/..)")
    src.add_argument("--file-id", type=int, help="Dataverse numeric file id")
    src.add_argument("--file-pid", help="Dataverse file persistent id (e.g., doi:...)")
    src.add_argument("--dataset-doi", help="Dataset DOI (e.g., doi:10.7910/DVN/BXBOTB)")
    src.add_argument("--filename", help="Exact file label/name within the dataset")
    src.add_argument("--all", action="store_true", help="Download all files from the dataset (requires --dataset-doi)")

    out = parser.add_argument_group("Output & behavior")
    out.add_argument(
        "--dest",
        default=str(Path(__file__).parent),
        help="Destination directory (default: this ML/data/ folder)",
    )
    out.add_argument(
        "--hash",
        dest="known_hash",
        default="unverified",
        help="Known file hash ('md5:<hex>' or 'sha256:<hex>'), or 'unverified'",
    )
    out.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Dataverse base URL (default: {DEFAULT_BASE_URL})",
    )
    out.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite if destination file already exists",
    )
    out.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar during download",
    )
    out.add_argument(
        "--extract",
        action="store_true",
        help="Extract archives/compressed files after download",
    )
    out.add_argument(
        "--extract-to",
        default=None,
        help="Extraction directory (default: same as --dest)",
    )
    out.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep original archives after successful extraction",
    )

    args = parser.parse_args(argv)

    dest_dir = Path(args.dest).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    extract_to = Path(args.extract_to).expanduser().resolve() if args.extract_to else dest_dir
    if args.extract and not extract_to.exists():
        extract_to.mkdir(parents=True, exist_ok=True)

    # Multi-file path: --dataset-doi with --all
    if args.dataset_doi and args.all:
        try:
            entries = list_dataset_files(args.base_url, args.dataset_doi)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        tasks: list[tuple[str, str]] = []
        seen_names: set[str] = set()
        for e in entries:
            label = sanitize_filename(e.get("label") or e.get("filename") or str(e.get("id")))
            df = e.get("dataFile") or {}
            fid = df.get("id") or e.get("id")
            if not fid:
                print(f"Skipping entry without id: {label}")
                continue
            url = f"{args.base_url}/api/access/datafile/{fid}?format=original"

            # Skip if exists and not overwrite
            # Ensure unique destination name per task (avoid concurrent collisions)
            base_label = label
            if label in seen_names:
                stem = Path(label).stem
                suffix = Path(label).suffix
                label = f"{stem}-{fid}{suffix}"
            seen_names.add(label)

            target = dest_dir / label
            if target.exists() and not args.overwrite:
                print(f"Already exists (skipping): {target}")
                continue

            tasks.append((url, label))

        if not tasks:
            print("Nothing to download.")
            return 0

        print(f"Starting {len(tasks)} downloads (sequential)...")
        errors = 0
        
        # Download sequentially (and optionally extract per-file)
        for url, fname in tasks:
            local_path, err = _download_one(
                url,
                dest_dir,
                fname,
                args.known_hash,
                args.progress,
                args.extract,  # extract per-file if requested
                extract_to,
                args.keep_archives,
            )
            if err:
                errors += 1
                print(f"[ERROR] {fname}: {err}")
            else:
                print(f"Saved to: {local_path}")

        if errors:
            print(f"Completed with {errors} errors.", file=sys.stderr)
            return 1
        return 0

    # Single-file path (original behavior)
    try:
        url, suggested_fname = build_download_url(
            base_url=args.base_url,
            url=args.url,
            file_id=args.file_id,
            file_pid=args.file_pid,
            dataset_doi=args.dataset_doi,
            filename=args.filename,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    fname = suggested_fname
    if fname is not None:
        target = dest_dir / fname
        if target.exists() and not args.overwrite:
            print(f"Already exists (skipping): {target}")
            print(str(target))
            return 0

    local_path, err = _download_one(
        url,
        dest_dir,
        fname,
        _normalize_known_hash(args.known_hash),
        args.progress,
        args.extract,
        extract_to,
        args.keep_archives,
    )
    if err:
        print(f"Download failed: {err}", file=sys.stderr)
        return 1
    print(f"Saved to: {local_path}")
    print(local_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
