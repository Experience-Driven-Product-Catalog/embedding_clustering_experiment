"""CLI for static catalog reports and submitted-review decision proposals."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .dynamic_report import (
    DynamicReportSettings,
    build_dynamic_review_decision_report,
    build_submission_from_catalog_reviews,
    render_dynamic_review_decision_markdown,
)
from .reporting import (
    ReportSettings,
    atomic_write_text,
    build_static_catalog_report,
    load_report_inputs,
    render_static_markdown,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Immutable experiment_results/YYYYMMDD-HHMMSS directory.",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=PROJECT_ROOT / "dataset/monitor_reviews.parquet",
        help="Review table containing idx, review, and productName/product_name.",
    )
    parser.add_argument(
        "--opinion-input",
        type=Path,
        default=PROJECT_ROOT / "dataset/monitor_opinion_units.parquet",
        help="Raw Opinion Unit input used to account for excluded general experience rows.",
    )
    parser.add_argument(
        "--human-results-dir",
        type=Path,
        default=None,
        help=(
            "Completed evaluator Parquet directory. By default, auto-discover it under "
            "RUN_DIR/evaluation/human_results."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for deterministic JSON and Markdown artifacts.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM-free report contracts for an AI shopping agent."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    static_parser = subparsers.add_parser(
        "static",
        help="Generate a static catalog analysis for one exact product name.",
    )
    _add_common_arguments(static_parser)
    static_parser.add_argument("--product-name", required=True)

    proposal_parser = subparsers.add_parser(
        "proposal",
        help="Generate a dynamic proposal from submitted, normalized reviews.",
    )
    _add_common_arguments(proposal_parser)
    source = proposal_parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--review-idx",
        action="append",
        dest="review_indices",
        type=int,
        help="Existing catalog review index for a demo submission; repeat for multiple reviews.",
    )
    source.add_argument(
        "--submission-json",
        type=Path,
        help="Structured review submission matching docs/agent_report_contract.md.",
    )
    proposal_parser.add_argument(
        "--submitted-at",
        help="Optional YYYYMMDD-HHMMSS submission time for --review-idx demos.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    inputs = load_report_inputs(
        args.run_dir,
        args.reviews,
        args.opinion_input,
        args.human_results_dir,
    )
    output_dir = args.output_dir.expanduser().resolve()
    if args.command == "static":
        report = build_static_catalog_report(
            inputs,
            args.product_name,
            settings=ReportSettings(),
        )
        json_path = output_dir / "static_catalog_report.json"
        markdown_path = output_dir / "static_catalog_report.md"
        write_json(report, json_path)
        atomic_write_text(render_static_markdown(report), markdown_path)
        receipt = {
            "report_type": report["report_type"],
            "report_id": report["report_id"],
            "json": str(json_path),
            "markdown": str(markdown_path),
        }
    else:
        if args.review_indices is not None:
            submission = build_submission_from_catalog_reviews(
                inputs,
                args.review_indices,
                submitted_at_local=args.submitted_at,
            )
        else:
            submission_path = args.submission_json.expanduser().resolve()
            submission = json.loads(submission_path.read_text(encoding="utf-8"))
        proposal = build_dynamic_review_decision_report(
            inputs,
            submission,
            settings=DynamicReportSettings(),
        )
        json_path = output_dir / "dynamic_decision_proposal.json"
        markdown_path = output_dir / "dynamic_decision_proposal.md"
        write_json(proposal, json_path)
        atomic_write_text(render_dynamic_review_decision_markdown(proposal), markdown_path)
        receipt = {
            "proposal_type": proposal["proposal_type"],
            "proposal_id": proposal["proposal_id"],
            "alternative_recommendation_status": proposal["alternative_recommendations"]["status"],
            "json": str(json_path),
            "markdown": str(markdown_path),
        }
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
