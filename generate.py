"""
Generate a random subset of test cases from the three available datasets.

Usage:
    python3 generate.py <test_num> [--seed 42] [--output-dir generated_test]

Behavior:
    - Samples <test_num> unique cases across:
        * MultiHiertt/train_GRP.json
        * spreadsheet_dataset_research/QA_GRP.json
        * spreadsheet_dataset_research/QA_GRP_with_feedback.json
    - Writes the combined selection to <output-dir>/dataset_geneerated.json
    - Creates Task folders with leading zeros under <output-dir> (Task0001, ...):
        * case.json with the normalized task entry
        * spreadsheets/… with copied spreadsheet files when they exist
    - If a dataset provides no output list, the entry uses an empty
      `expected_output_file: []` as requested.
"""

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class DatasetConfig:
    name: str
    json_path: Path
    spreadsheet_kind: str  # "multihiertt" or "spreadsheet"


DATASETS: Sequence[DatasetConfig] = (
    DatasetConfig(
        name="MultiHiertt",
        json_path=Path("MultiHiertt/train_GRP.json"),
        spreadsheet_kind="multihiertt",
    ),
    # DatasetConfig(
    #     name="Spreadsheet",
    #     json_path=Path("spreadsheet_dataset_research/QA_GRP.json"),
    #     spreadsheet_kind="spreadsheet",
    # ),
    DatasetConfig(
        name="SpreadsheetWithFeedback",
        json_path=Path("spreadsheet_dataset_research/QA_GRP_with_feedback.json"),
        spreadsheet_kind="spreadsheet",
    ),
)


def load_dataset(cfg: DatasetConfig) -> List[Dict]:
    with cfg.json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{cfg.json_path} does not contain a list")
    return data


def ensure_list(val: Optional[Iterable]) -> List:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def normalize_entry(raw: Dict, cfg: DatasetConfig) -> Dict:
    """
    Standardize fields and keep track of the dataset source.
    Missing outputs become [] per user instruction.
    """
    expected_output = raw.get("expected_output_file")
    entry = {
        "task_id": str(raw.get("task_id", "")),
        "title": raw.get("title", ""),
        "spreadsheets": ensure_list(raw.get("spreadsheets")),
        "prompt": raw.get("prompt", ""),
        "answer": raw.get("answer", ""),
        "expected_output_file": expected_output if expected_output is not None else [],
        "feedback": raw.get("feedback", ""),
        "source_dataset": cfg.name,
    }
    return entry


def find_spreadsheet_file(kind: str, name: str) -> Optional[Path]:
    """
    Resolve a spreadsheet path on disk based on the dataset kind.
    Returns None when the file cannot be located.
    """
    root = Path(".")
    if kind == "multihiertt":
        candidate = root / "MultiHiertt" / name
        return candidate if candidate.exists() else None

    if kind == "spreadsheet":
        base = root / "spreadsheet_dataset_research" / "tables"
        # If the name already has an extension, use it directly.
        candidate = base / name
        if candidate.exists():
            return candidate
        # Otherwise try common extensions.
        for ext in (".xlsx", ".xls", ".csv", ".json"):
            candidate = base / f"{name}{ext}"
            if candidate.exists():
                return candidate
        # As a last resort, pick the first file that starts with the name.
        matches = sorted(base.glob(f"{name}*"))
        if matches:
            return matches[0]
    return None


def copy_spreadsheets(
    entry: Dict,
    cfg: DatasetConfig,
    task_dir: Path,
) -> List[str]:
    """
    Copy spreadsheet files for a task into its task_dir/spreadsheets folder.
    Returns the list of relative paths (from the output root) to the copied files.
    """
    resolved_paths: List[str] = []
    spreadsheets_dir = task_dir / "spreadsheets" / cfg.name
    for sheet in entry.get("spreadsheets", []):
        src = find_spreadsheet_file(cfg.spreadsheet_kind, sheet)
        if src is None:
            continue
        dest = spreadsheets_dir / Path(sheet)
        if dest.suffix == "" and src.suffix:
            dest = dest.with_suffix(src.suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        resolved_paths.append(str(dest))
    return resolved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random test cases.")
    parser.add_argument("test_num", type=int, help="Number of test cases to sample")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_test"),
        help="Directory to store generated tests",
    )
    args = parser.parse_args()

    if args.test_num <= 0:
        raise ValueError("test_num must be positive")

    if args.seed is not None:
        random.seed(args.seed)

    all_entries: List[Dict] = []
    dataset_by_name: Dict[str, DatasetConfig] = {cfg.name: cfg for cfg in DATASETS}
    for cfg in DATASETS:
        if not cfg.json_path.exists():
            raise FileNotFoundError(f"Dataset not found: {cfg.json_path}")
        raw_items = load_dataset(cfg)
        normalized = [normalize_entry(item, cfg) for item in raw_items]
        all_entries.extend(normalized)

    total_available = len(all_entries)
    if args.test_num > total_available:
        raise ValueError(
            f"Requested {args.test_num} cases but only {total_available} available"
        )

    indices = random.sample(range(total_available), args.test_num)
    width = max(3, len(str(args.test_num)))

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    generated_entries: List[Dict] = []
    for idx, selection in enumerate(indices, start=1):
        entry = all_entries[selection].copy()
        cfg = dataset_by_name[entry["source_dataset"]]
        task_name = f"Task{idx:0{width}d}"
        task_dir = output_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        copied_paths = copy_spreadsheets(entry, cfg, task_dir)
        entry["spreadsheets"] = copied_paths

        # Write individual task file
        case_path = task_dir / "case.json"
        with case_path.open("w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

        # Store path relative to the output root for the global file
        entry_for_global = entry.copy()
        entry_for_global["spreadsheets"] = [
            str(Path(p).relative_to(output_root)) for p in copied_paths
        ]
        entry_for_global["task_dir"] = task_name
        generated_entries.append(entry_for_global)

    out_file = output_root / "dataset_geneerated.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(generated_entries, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(generated_entries)} tasks to {out_file} in {output_root.resolve()}"
    )


if __name__ == "__main__":
    main()
