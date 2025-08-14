#!/usr/bin/env python3
import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

MASK_DIRS = {"Mask_TaskR1", "Mask_TaskR2", "Mask_TaskS1", "Mask_TaskS2"}
UNDERSAMPLE_DIRS = {"UnderSample_TaskR1", "UnderSample_TaskR2", "UnderSample_TaskS1", "UnderSample_TaskS2"}
TASK_DIRS = {"TaskR1", "TaskR2", "TaskS1", "TaskS2"}

# Map filename prefixes to canonical modality names
MODALITY_MAP = {
    "cine": "Cine",
    "blackblood": "BlackBlood",
    "flow2d": "Flow2d",
    "lge": "LGE",
    "mapping": "Mapping",
    "perfusion": "Perfusion",
    "t1rho": "T1rho",
    "t1w": "T1w",
    "t2w": "T2w",
}


def err(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def find_task(parts):
    # Return task name if present in path parts; else None
    for p in parts:
        if p in TASK_DIRS:
            return p
    return None


def find_anchor_index(parts):
    """
    Find the index of the anchor directory that precedes the center/scanner/patient triplet.
    Prefer UnderSample_Task* if present; otherwise fall back to TestSet if needed.
    Returns index of anchor part or None.
    """
    for i, p in enumerate(parts):
        if p in UNDERSAMPLE_DIRS or p == "TestSet":
            return i
    return None


def compose_flat_name(parts):
    """
    Given full path parts, return (task, flat_filename) or (None, None) if not applicable.
    Expected structure after anchor:
      <anchor> / <Center> / <Scanner> / <Patient> / <filename.mat>
    """
    task = find_task(parts)
    if task is None:
        return None, None

    # Ignore any path that passes through a mask directory
    if any(p in MASK_DIRS for p in parts):
        return None, None

    # Only .mat files
    filename = parts[-1]
    if not filename.lower().endswith(".mat"):
        return None, None

    anchor_idx = find_anchor_index(parts)
    if anchor_idx is None:
        return None, None

    # Expect at least 4 components after anchor (Center, Scanner, Patient, filename)
    if len(parts) < anchor_idx + 4 + 1:
        return None, None

    try:
        center = parts[anchor_idx + 1]
        scanner = parts[anchor_idx + 2]
        patient = parts[anchor_idx + 3]
    except IndexError:
        return None, None

    # Build flat filename
    flat_name = f"{center}_{scanner}_{patient}_{filename}"
    return task, flat_name


def safe_symlink(src: Path, dst: Path, verbose: bool = False):
    """
    Create a symlink dst -> src (absolute path). If dst exists, append a short hash for uniqueness.
    """
    dst_parent = dst.parent
    dst_parent.mkdir(parents=True, exist_ok=True)

    # Always link to an absolute, normalized path
    src_abs = src.resolve()

    def _link(target: Path, link_name: Path):
        # If link exists (file/dir or symlink), skip creating
        if link_name.exists() or link_name.is_symlink():
            return False
        link_name.symlink_to(target)
        return True

    if _link(src_abs, dst):
        if verbose:
            print(f"{dst} -> {src_abs}")
        return dst

    # Collision: append hash of absolute source path
    stem = dst.stem
    suffix = dst.suffix
    h = hashlib.sha1(str(src_abs).encode("utf-8")).hexdigest()[:8]
    new_dst = dst_parent / f"{stem}__{h}{suffix}"
    if _link(src_abs, new_dst):
        if verbose:
            print(f"{new_dst} -> {src_abs}")
        return new_dst

    # Fallback with numeric suffix
    i = 1
    while True:
        cand = dst_parent / f"{stem}__dup{i}{suffix}"
        if _link(src_abs, cand):
            if verbose:
                print(f"{cand} -> {src_abs}")
            return cand
        i += 1


_sampling_re = re.compile(r"kus_([A-Za-z]+)")


def parse_modality_and_sampling(filename: str):
    """
    Parse modality (Cine, LGE, Mapping, etc.) from the filename prefix before first underscore,
    and sampling type (Uniform, ktRadial, ktGaussian) from the token after 'kus_'.
    Returns (modality, sampling) or raises ValueError on failure.
    """
    base = Path(filename).name
    if not base.lower().endswith(".mat"):
        raise ValueError("Not a .mat file")

    # Modality: prefix before first underscore
    if "_" not in base:
        raise ValueError("Cannot find modality prefix")
    modality_key = base.split("_", 1)[0].lower()
    modality = MODALITY_MAP.get(modality_key)
    if modality is None:
        raise ValueError(f"Unknown modality prefix: {modality_key}")

    # Sampling: letters immediately after 'kus_'
    m = _sampling_re.search(base)
    if not m:
        raise ValueError("Cannot parse sampling after 'kus_'")
    sampling = m.group(1)

    return modality, sampling


def write_lists_for_task(task_dir: Path, lists_root: Path):
    """
    Scan task_dir (which should contain only flat .mat files) and group them
    by (modality, sampling). Write .lst files to lists_root/<Task>/Modality_Sampling.lst
    with absolute paths. Return a dict of counts and perform a consistency check.
    """
    task_name = task_dir.name
    out_dir = lists_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    all_files = []

    for p in sorted(task_dir.glob("*.mat")):
        all_files.append(p)
        try:
            modality, sampling = parse_modality_and_sampling(p.name)
        except ValueError as e:
            raise RuntimeError(f"[{task_name}] Could not categorize file '{p.name}': {e}") from e
        key = (modality, sampling)
        groups[key].append(str(p.resolve()))  # absolute paths

    # Write lists
    counts = {}
    for (modality, sampling), paths in sorted(groups.items()):
        lst_name = f"{modality}_{sampling}.lst"
        lst_path = out_dir / lst_name
        with open(lst_path, "w") as f:
            f.write("\n".join(paths) + ("\n" if paths else ""))
        counts[lst_name] = len(paths)

    # Consistency check
    total_listed = sum(counts.values())
    total_files = len(all_files)
    summary_path = out_dir / "SUMMARY.txt"
    with open(summary_path, "w") as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Total files discovered: {total_files}\n")
        f.write("Per-list counts:\n")
        for name, c in sorted(counts.items()):
            f.write(f"  {name}: {c}\n")
        f.write(f"\nSum across lists: {total_listed}\n")
        f.write("STATUS: OK\n" if total_listed == total_files else "STATUS: MISMATCH\n")

    if total_listed != total_files:
        raise RuntimeError(f"[{task_name}] Mismatch: {total_listed} listed vs {total_files} files")

    return counts, total_files


def main():
    parser = argparse.ArgumentParser(
        description="Flatten CMRxRecon dataset into a symbolic-link directory with absolute paths, and optionally emit per-task .lst files by modality and sampling type."
    )
    parser.add_argument("--input", type=str, required=True, help="ABSOLUTE path to the dataset root (e.g., /input).")
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="ABSOLUTE path to the output directory for flattened links (e.g., /output/flat).",
    )
    parser.add_argument(
        "--lists",
        type=str,
        required=False,
        help="ABSOLUTE path to write .lst files grouped by Modality_Sampling per task (e.g., /output/lists).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print created links.")
    args = parser.parse_args()

    if not os.path.isabs(args.input):
        err("--input must be an ABSOLUTE path (e.g., /input)")
    if not os.path.isabs(args.dest):
        err("--dest must be an ABSOLUTE path (e.g., /output/flat)")
    if args.lists is not None and not os.path.isabs(args.lists):
        err("--lists must be an ABSOLUTE path (e.g., /output/lists)")

    input_root = Path(args.input).resolve()
    dest_root = Path(args.dest).resolve()
    lists_root = Path(args.lists).resolve() if args.lists else None

    if not input_root.exists():
        err(f"Input root does not exist: {input_root}")

    created = 0
    skipped = 0

    # Step 1: Build the flat symlink view
    for root, dirs, files in os.walk(input_root):
        # Skip mask directories early
        dirs[:] = [d for d in dirs if d not in MASK_DIRS]

        for f in files:
            if not f.lower().endswith(".mat"):
                skipped += 1
                continue

            src = Path(root) / f
            parts = src.parts
            task, flat_name = compose_flat_name(parts)
            if task is None or flat_name is None:
                skipped += 1
                continue

            dst = dest_root / task / flat_name
            try:
                _ = safe_symlink(src, dst, verbose=args.verbose)
                created += 1
            except Exception as e:
                skipped += 1
                if args.verbose:
                    print(f"SKIP ({e}): {src}")

    print(f"Flattening done. Created {created} symlinks, skipped {skipped} items.")
    print(f"Flat output directory: {dest_root}")

    # Step 2: Optionally, write .lst files per task
    if lists_root is not None:
        lists_root.mkdir(parents=True, exist_ok=True)
        totals_ok = True
        report_lines = []
        # Only consider tasks that exist in the flat output
        present_tasks = [p.name for p in dest_root.iterdir() if p.is_dir() and p.name in TASK_DIRS]
        for task_name in sorted(present_tasks):
            task_dir = dest_root / task_name
            try:
                counts, total_files = write_lists_for_task(task_dir, lists_root)
                report_lines.append(
                    f"{task_name}: OK ({sum(counts.values())} == {total_files}) with {len(counts)} lists"
                )
            except RuntimeError as e:
                totals_ok = False
                report_lines.append(f"{task_name}: ERROR - {e}")

        report = "\n".join(report_lines) if report_lines else "No tasks present to list."
        print("List generation summary:")
        print(report)
        # Also write a top-level REPORT.txt
        with open(lists_root / "REPORT.txt", "w") as f:
            f.write(report + "\n")
        if not totals_ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
