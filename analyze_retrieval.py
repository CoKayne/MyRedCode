#!/usr/bin/env python3
"""
Analyze retrieval patterns from evaluation results.

Generates statistics on:
- Same-category vs cross-category retrieval rates
- Per-category retrieval patterns
- Cross-category retrieval matrix
- Similarity score distributions
- Cross-model analysis (when comparing multiple experiments)

Usage:
    # Single analysis file
    python analyze_retrieval.py --analysis-file ./outputs/.../retrieval_analysis.json

    # Compare multiple experiments (same model, different reflection sources)
    python analyze_retrieval.py --compare \
        ./outputs/qwen_with_qwen_reflections/retrieval_analysis.json \
        ./outputs/qwen_with_llama_reflections/retrieval_analysis.json
"""

import os
import sys
import json
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def load_analysis_file(path: str) -> Dict:
    """Load and validate an analysis JSON file."""
    with open(path) as f:
        data = json.load(f)

    if "records" not in data:
        raise ValueError(f"Invalid analysis file: missing 'records' key")

    return data


def analyze_single_file(data: Dict, verbose: bool = True) -> Dict:
    """
    Analyze a single retrieval analysis file.

    Returns:
        Dictionary with analysis results
    """
    records = data.get("records", [])
    config = data.get("config", {})

    if not records:
        print("Warning: No retrieval records found")
        return {}

    # Basic statistics
    total_queries = len(records)
    total_retrievals = sum(len(r["retrieved"]) for r in records)
    total_same = sum(r["same_category_count"] for r in records)
    total_cross = sum(r["cross_category_count"] for r in records)

    # Per-category breakdown
    by_category = defaultdict(lambda: {
        "queries": 0,
        "same": 0,
        "cross": 0,
        "total_retrieved": 0,
        "avg_similarity": [],
    })

    for record in records:
        cat = record["query_category"]
        by_category[cat]["queries"] += 1
        by_category[cat]["same"] += record["same_category_count"]
        by_category[cat]["cross"] += record["cross_category_count"]
        by_category[cat]["total_retrieved"] += len(record["retrieved"])
        for r in record["retrieved"]:
            by_category[cat]["avg_similarity"].append(r["similarity"])

    # Calculate averages
    for cat in by_category:
        sims = by_category[cat]["avg_similarity"]
        by_category[cat]["avg_similarity"] = sum(sims) / len(sims) if sims else 0

    # Cross-category matrix
    cross_matrix = defaultdict(lambda: defaultdict(int))
    for record in records:
        query_cat = record["query_category"]
        for r in record["retrieved"]:
            retrieved_cat = r.get("family", "unknown")
            cross_matrix[query_cat][retrieved_cat] += 1

    # Similarity distribution
    all_similarities = []
    same_cat_similarities = []
    cross_cat_similarities = []

    for record in records:
        query_cat = record["query_category"]
        for r in record["retrieved"]:
            sim = r["similarity"]
            all_similarities.append(sim)
            if r.get("family") == query_cat:
                same_cat_similarities.append(sim)
            else:
                cross_cat_similarities.append(sim)

    results = {
        "config": config,
        "summary": {
            "total_queries": total_queries,
            "total_retrievals": total_retrievals,
            "same_category_count": total_same,
            "cross_category_count": total_cross,
            "same_category_pct": (total_same / total_retrievals * 100) if total_retrievals > 0 else 0,
            "avg_similarity_all": sum(all_similarities) / len(all_similarities) if all_similarities else 0,
            "avg_similarity_same_cat": sum(same_cat_similarities) / len(same_cat_similarities) if same_cat_similarities else 0,
            "avg_similarity_cross_cat": sum(cross_cat_similarities) / len(cross_cat_similarities) if cross_cat_similarities else 0,
        },
        "by_category": dict(by_category),
        "cross_matrix": {k: dict(v) for k, v in cross_matrix.items()},
    }

    if verbose:
        print_analysis_report(results)

    return results


def print_analysis_report(results: Dict) -> None:
    """Print a formatted analysis report."""
    summary = results["summary"]
    by_category = results["by_category"]
    cross_matrix = results["cross_matrix"]
    config = results.get("config", {})

    print("\n" + "=" * 70)
    print("RETRIEVAL ANALYSIS REPORT")
    print("=" * 70)

    if config:
        print(f"\nConfiguration:")
        print(f"  Split mode: {config.get('split_mode', 'unknown')}")
        print(f"  Similarity threshold: {config.get('retrieval_similarity_threshold', 'unknown')}")
        print(f"  Max results: {config.get('retrieval_max_results', 'unknown')}")
        print(f"  Database: {config.get('retrieval_db_path', 'unknown')}")

    print(f"\n{'=' * 70}")
    print("OVERALL STATISTICS")
    print("-" * 70)
    print(f"Total test queries: {summary['total_queries']}")
    print(f"Total retrieved entries: {summary['total_retrievals']}")
    print(f"Same-category retrievals: {summary['same_category_count']} ({summary['same_category_pct']:.1f}%)")
    print(f"Cross-category retrievals: {summary['cross_category_count']} ({100 - summary['same_category_pct']:.1f}%)")
    print(f"\nAverage similarity (all): {summary['avg_similarity_all']:.3f}")
    print(f"Average similarity (same-cat): {summary['avg_similarity_same_cat']:.3f}")
    print(f"Average similarity (cross-cat): {summary['avg_similarity_cross_cat']:.3f}")

    print(f"\n{'=' * 70}")
    print("PER-CATEGORY BREAKDOWN")
    print("-" * 70)
    print(f"{'Category':<15} {'Queries':<10} {'Same':<10} {'Cross':<10} {'Same %':<10} {'Avg Sim':<10}")
    print("-" * 70)

    categories = sorted(by_category.keys())
    for cat in categories:
        stats = by_category[cat]
        total = stats["same"] + stats["cross"]
        same_pct = (stats["same"] / total * 100) if total > 0 else 0
        print(f"{cat:<15} {stats['queries']:<10} {stats['same']:<10} {stats['cross']:<10} {same_pct:<10.1f} {stats['avg_similarity']:<10.3f}")

    print(f"\n{'=' * 70}")
    print("CROSS-CATEGORY RETRIEVAL MATRIX")
    print("(Rows = query category, Columns = retrieved category)")
    print("-" * 70)

    # Print header
    print(f"{'Query \\ Retr':<12}", end="")
    for cat in categories:
        print(f"{cat[:8]:<10}", end="")
    print()
    print("-" * 70)

    # Print rows
    for query_cat in categories:
        print(f"{query_cat:<12}", end="")
        for ret_cat in categories:
            count = cross_matrix.get(query_cat, {}).get(ret_cat, 0)
            print(f"{count:<10}", end="")
        print()

    print("=" * 70)


def compare_experiments(files: List[str], verbose: bool = True) -> Dict:
    """
    Compare retrieval patterns across multiple experiments.

    Args:
        files: List of paths to analysis JSON files

    Returns:
        Dictionary with comparison results
    """
    experiments = []

    for path in files:
        data = load_analysis_file(path)
        results = analyze_single_file(data, verbose=False)
        results["file"] = path
        experiments.append(results)

    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-EXPERIMENT COMPARISON")
        print("=" * 70)

        print(f"\n{'Experiment':<50} {'Same %':<12} {'Avg Sim':<12}")
        print("-" * 70)

        for exp in experiments:
            name = Path(exp["file"]).parent.name
            summary = exp["summary"]
            print(f"{name:<50} {summary['same_category_pct']:<12.1f} {summary['avg_similarity_all']:<12.3f}")

        print("\n" + "-" * 70)
        print("Database paths:")
        for exp in experiments:
            name = Path(exp["file"]).parent.name
            db_path = exp.get("config", {}).get("retrieval_db_path", "unknown")
            print(f"  {name}: {db_path}")

        print("=" * 70)

    return {"experiments": experiments}


def export_to_csv(results: Dict, output_path: str) -> None:
    """Export analysis results to CSV."""
    by_category = results["by_category"]
    summary = results["summary"]

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Summary section
        writer.writerow(["SUMMARY"])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Queries", summary["total_queries"]])
        writer.writerow(["Total Retrievals", summary["total_retrievals"]])
        writer.writerow(["Same Category Count", summary["same_category_count"]])
        writer.writerow(["Cross Category Count", summary["cross_category_count"]])
        writer.writerow(["Same Category %", f"{summary['same_category_pct']:.1f}"])
        writer.writerow(["Avg Similarity (All)", f"{summary['avg_similarity_all']:.3f}"])
        writer.writerow(["Avg Similarity (Same Cat)", f"{summary['avg_similarity_same_cat']:.3f}"])
        writer.writerow(["Avg Similarity (Cross Cat)", f"{summary['avg_similarity_cross_cat']:.3f}"])
        writer.writerow([])

        # Per-category section
        writer.writerow(["PER-CATEGORY BREAKDOWN"])
        writer.writerow(["Category", "Queries", "Same", "Cross", "Same %", "Avg Similarity"])
        for cat in sorted(by_category.keys()):
            stats = by_category[cat]
            total = stats["same"] + stats["cross"]
            same_pct = (stats["same"] / total * 100) if total > 0 else 0
            writer.writerow([cat, stats["queries"], stats["same"], stats["cross"],
                           f"{same_pct:.1f}", f"{stats['avg_similarity']:.3f}"])
        writer.writerow([])

        # Cross-category matrix
        writer.writerow(["CROSS-CATEGORY MATRIX"])
        categories = sorted(by_category.keys())
        writer.writerow(["Query \\ Retrieved"] + categories)
        cross_matrix = results["cross_matrix"]
        for query_cat in categories:
            row = [query_cat]
            for ret_cat in categories:
                row.append(cross_matrix.get(query_cat, {}).get(ret_cat, 0))
            writer.writerow(row)

    print(f"Results exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze retrieval patterns from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single file
    python analyze_retrieval.py --analysis-file ./retrieval_analysis.json

    # Compare multiple experiments
    python analyze_retrieval.py --compare \
        ./outputs/exp1/retrieval_analysis.json \
        ./outputs/exp2/retrieval_analysis.json

    # Export results to CSV
    python analyze_retrieval.py --analysis-file ./retrieval_analysis.json \
        --export-csv ./analysis_report.csv
        """
    )

    parser.add_argument(
        "--analysis-file",
        type=str,
        help="Path to a single retrieval analysis JSON file"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=str,
        metavar="FILE",
        help="Compare multiple analysis files"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export results to CSV file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    if not args.analysis_file and not args.compare:
        parser.print_help()
        print("\nError: Must specify --analysis-file or --compare")
        return 1

    if args.compare:
        # Compare multiple experiments
        results = compare_experiments(args.compare, verbose=not args.quiet)
    else:
        # Analyze single file
        data = load_analysis_file(args.analysis_file)
        results = analyze_single_file(data, verbose=not args.quiet)

        if args.export_csv:
            export_to_csv(results, args.export_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
