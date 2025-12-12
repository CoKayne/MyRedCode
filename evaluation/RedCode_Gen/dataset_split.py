"""
Dataset split configuration for train/test separation.

This module provides functions to determine which testcases belong to
training (reflection database population) vs testing (evaluation).

Split strategy: 50/50 by index
- Training: indices 1-10
- Testing: indices 11-20
"""

import os
import re
from typing import List, Tuple, Dict, Set

# Dataset categories
CATEGORIES = ["adware", "ddos", "others", "ransomware", "rootkit", "spyware", "trojan", "virus"]

# Default split configuration
TRAIN_INDICES = set(range(1, 11))   # 1-10 for training
TEST_INDICES = set(range(11, 21))   # 11-20 for testing


def get_file_index(file_name: str) -> int:
    """
    Extract the numeric index from a filename.

    Examples:
        'ddos_1_syn_flood.py' -> 1
        'ransomware_15_fake_antivirus.py' -> 15

    Args:
        file_name: The filename to parse

    Returns:
        The extracted index, or -1 if not found
    """
    match = re.search(r'_(\d+)_', file_name)
    if match:
        return int(match.group(1))
    return -1


def is_train_file(file_name: str) -> bool:
    """
    Check if a file belongs to the training set.

    Args:
        file_name: The filename to check

    Returns:
        True if the file index is in training range (1-10)
    """
    idx = get_file_index(file_name)
    return idx in TRAIN_INDICES


def is_test_file(file_name: str) -> bool:
    """
    Check if a file belongs to the test set.

    Args:
        file_name: The filename to check

    Returns:
        True if the file index is in test range (11-20)
    """
    idx = get_file_index(file_name)
    return idx in TEST_INDICES


def get_train_files(base_dir: str) -> List[Tuple[str, str]]:
    """
    Get list of (family, filename) tuples for training set.

    Args:
        base_dir: Path to the dataset directory (e.g., ./dataset/RedCode-Gen)

    Returns:
        List of (family, filename) tuples for training files
    """
    train_files = []
    for family in CATEGORIES:
        family_dir = os.path.join(base_dir, family)
        if os.path.isdir(family_dir):
            for file_name in sorted(os.listdir(family_dir)):
                if file_name.endswith(".py") and is_train_file(file_name):
                    train_files.append((family, file_name))
    return train_files


def get_test_files(base_dir: str) -> List[Tuple[str, str]]:
    """
    Get list of (family, filename) tuples for test set.

    Args:
        base_dir: Path to the dataset directory (e.g., ./dataset/RedCode-Gen)

    Returns:
        List of (family, filename) tuples for test files
    """
    test_files = []
    for family in CATEGORIES:
        family_dir = os.path.join(base_dir, family)
        if os.path.isdir(family_dir):
            for file_name in sorted(os.listdir(family_dir)):
                if file_name.endswith(".py") and is_test_file(file_name):
                    test_files.append((family, file_name))
    return test_files


def get_split_stats(base_dir: str) -> Dict[str, Dict[str, int]]:
    """
    Get statistics about the train/test split.

    Args:
        base_dir: Path to the dataset directory

    Returns:
        Dictionary with counts per category and totals
    """
    stats = {"by_category": {}, "totals": {"train": 0, "test": 0, "total": 0}}

    for family in CATEGORIES:
        family_dir = os.path.join(base_dir, family)
        if os.path.isdir(family_dir):
            train_count = 0
            test_count = 0
            for file_name in os.listdir(family_dir):
                if file_name.endswith(".py"):
                    if is_train_file(file_name):
                        train_count += 1
                    elif is_test_file(file_name):
                        test_count += 1

            stats["by_category"][family] = {
                "train": train_count,
                "test": test_count,
                "total": train_count + test_count
            }
            stats["totals"]["train"] += train_count
            stats["totals"]["test"] += test_count
            stats["totals"]["total"] += train_count + test_count

    return stats


def print_split_summary(base_dir: str) -> None:
    """Print a summary of the train/test split."""
    stats = get_split_stats(base_dir)

    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"Training indices: 1-10")
    print(f"Testing indices: 11-20")
    print("-" * 60)
    print(f"{'Category':<15} {'Train':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)

    for family in CATEGORIES:
        if family in stats["by_category"]:
            cat_stats = stats["by_category"][family]
            print(f"{family:<15} {cat_stats['train']:<10} {cat_stats['test']:<10} {cat_stats['total']:<10}")

    print("-" * 60)
    totals = stats["totals"]
    print(f"{'TOTAL':<15} {totals['train']:<10} {totals['test']:<10} {totals['total']:<10}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test the module
    import sys

    base_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset/RedCode-Gen"

    print_split_summary(base_dir)

    print("Sample training files:")
    train_files = get_train_files(base_dir)
    for family, fname in train_files[:5]:
        print(f"  {family}/{fname}")

    print("\nSample test files:")
    test_files = get_test_files(base_dir)
    for family, fname in test_files[:5]:
        print(f"  {family}/{fname}")
