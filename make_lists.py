#!/usr/bin/env python3
import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

TASK_DIRS = {"TaskR1", "TaskR2", "TaskS1", "TaskS2"}

# Normalize modality keys to the target lowercase names used in your example
# Accept a variety of aliases (including "mapping", "t1map", "t2map") but output "map"
MODALITY_ALIASES = {
    "cine": "cine",
    "lge": "lge",
    "blackblood": "blackblood",
    "flow2d": "flow2d",
    "perfusion": "perfusion",
    "t1rho": "t1rho",
    "t1w": "t1w",
    "t2w": "t2w",
    "mapping": "map",
    "t1map": "map",
    "t2map": "map",
    "t1mapping": "map",
    "t2mapping": "map",
    "map": "map",
}
# Sampling patterns normalized to lowercase
SAMP_ALIASES = {
    "uniform": "uniform",
    "ktradial": "ktradial",
    "ktgaussian": "ktgaussian",
}

_sampling_re = re.compile(r"kus_([A-Za-z]+)", re.IGNORECASE)


def err(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def parse_from_flat_name(name: str):
    """
    Given a flattened filename like:
      CenterXXX_ScannerYYY_P###_<original>.mat
    return (original_without_ext).
    """
    m = re.search(r"_P\d+_(.+)$", name, flags=re.IGNORECASE)
    if not m:
        raise ValueError("Cannot locate original segment after patient token")
    original = m.group(1)
    if original.lower().endswith(".mat"):
        original = original[:-4]
    return original


def parse_modality_and_sampling_from_original(original: str):
    """
    original: e.g., 'cine_lax_3ch_kus_Uniform16'
    Returns (modality_norm, sampling_norm)
    """
    # modality is prefix before first underscore
    if "_" not in original:
        raise ValueError("Original part lacks underscores to find modality")
    modality_key = original.split("_", 1)[0].lower()
    modality_norm = MODALITY_ALIASES.get(modality_key)
    if modality_norm is None:
        # fallback: scan for a known alias substring
        low = original.lower()
        for k, v in MODALITY_ALIASES.items():
            if k in low:
                modality_norm = v
                break
    if modality_norm is None:
        raise ValueError(f"Unknown modality: {modality_key}")

    m = _sampling_re.search(original)
    if not m:
        raise ValueError("Cannot parse sampling after 'kus_'")
    sampling_key = m.group(1).lower()
    sampling_norm = SAMP_ALIASES.get(sampling_key)
    if sampling_norm is None:
        # treat raw key as already normalized if it's one of expected lower cases
        if sampling_key in SAMP_ALIASES.values():
            sampling_norm = sampling_key
        else:
            # allow e.g. 'ktradial', 'ktgaussian', 'uniform' with any case
            if sampling_key.startswith("ktgaussian"):
                sampling_norm = "ktgaussian"
            elif sampling_key.startswith("ktradial"):
                sampling_norm = "ktradial"
            elif sampling_key.startswith("uniform"):
                sampling_norm = "uniform"
            else:
                raise ValueError(f"Unknown sampling: {sampling_key}")

    return modality_norm, sampling_norm


def write_lists_for_task(task_dir: Path, lists_root: Path, verbose: bool = False):
    """
    Group .mat files in task_dir into <modality>_<sampling>.lst based on filename parsing.
    """
    task_name = task_dir.name
    out_dir = lists_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    all_files = []

    for p in sorted(task_dir.glob("*.mat")):
        all_files.append(p)
        base = p.name
        original = parse_from_flat_name(base)
        modality, sampling = parse_modality_and_sampling_from_original(original)
        if verbose:
            print(f"  {p.name} -> {modality} {sampling}")
        groups[(modality, sampling)].append(str(p.resolve()))

    # Write .lst files
    counts = {}
    for (modality, sampling), paths in sorted(groups.items()):
        lst_name = f"{modality}_{sampling}.lst"
        with open(out_dir / lst_name, "w") as f:
            f.write("\n".join(paths) + ("\n" if paths else ""))
        counts[lst_name] = len(paths)

    if verbose:
        print(f"  {task_name} -> {len(groups)} lists")

    # Consistency check and summary
    total_listed = sum(counts.values())
    total_files = len(all_files)
    with open(out_dir / "SUMMARY.txt", "w") as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Total files discovered: {total_files}\n")
        f.write("Per-list counts:\n")
        for name, c in sorted(counts.items()):
            f.write(f"  {name}: {c}\n")
        f.write(f"\nSum across lists: {total_listed}\n")
        f.write("STATUS: OK\n" if total_listed == total_files else "STATUS: MISMATCH\n")

    if verbose:
        print("\n" * 2)
        print(f"Task: {task_name}\n")
        print(f"Total files discovered: {total_files}\n")
        print(f"Per-list counts:\n")
        for name, c in sorted(counts.items()):
            print(f"  {name}: {c}")
        print(f"\nSum across lists: {total_listed}\n")
        print("STATUS: OK\n" if total_listed == total_files else "STATUS: MISMATCH\n")

    return counts, total_files


def main():
    epilog = """
    Example:
      python make_lists.py --flat /output/flat --lists /output/lists
    """
    ap = argparse.ArgumentParser(
        description="Generate per-task modality/sampling list files from a flat symlink directory.", epilog=epilog
    )
    ap.add_argument("--flat", type=str, required=True, help="ABSOLUTE path to flat symlink root (e.g., /output/flat).")
    ap.add_argument("--lists", type=str, required=True, help="ABSOLUTE path to write lists (e.g., /output/lists).")
    ap.add_argument("--verbose", action="store_true", help="Print verbose output.")
    args = ap.parse_args()

    if not os.path.isabs(args.flat):
        err("--flat must be an ABSOLUTE path")
    if not os.path.isabs(args.lists):
        err("--lists must be an ABSOLUTE path")

    flat_root = Path(args.flat).resolve()
    lists_root = Path(args.lists).resolve()
    if not flat_root.exists():
        err(f"Flat root does not exist: {flat_root}")

    lists_root.mkdir(parents=True, exist_ok=True)

    report_lines = []
    ok = True
    # Only consider present tasks under flat_root
    for task_dir in sorted([p for p in flat_root.iterdir() if p.is_dir() and p.name in TASK_DIRS]):
        try:
            counts, total = write_lists_for_task(task_dir, lists_root, args.verbose)
            report_lines.append(f"{task_dir.name}: OK ({sum(counts.values())} == {total}) with {len(counts)} lists")
        except Exception as e:
            ok = False
            report_lines.append(f"{task_dir.name}: ERROR - {e}")

    report = "\n".join(report_lines) if report_lines else "No tasks found under flat root."
    print(report)
    with open(lists_root / "REPORT.txt", "w") as f:
        f.write(report + "\n")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
