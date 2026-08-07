"""Single-file Streamlit evaluation UI; no separate API or frontend build."""

from __future__ import annotations

import argparse
import html
import json
import random
import threading
import uuid
from collections.abc import Mapping, MutableMapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from .artifacts import atomic_write_parquet

TASK_COLUMNS = ("review_idx", "result_a", "result_b", "result_c", "result_d")
CLUSTER_TASK_COLUMNS = (
    "cluster_validation_id",
    "stage",
    "cluster_id",
    "parent_aspect_label",
    "canonical_label",
    "members",
)
EXPERIMENTS = ("A", "B", "C", "D")
SCORE_COLUMNS = (
    "factual_faithfulness",
    "core_explanatory_power",
    "information_coverage",
)
CLUSTER_SCORE_COLUMNS = ("cohesion", "canonical_label_fit")
SCORE_LABELS = {
    "factual_faithfulness": (
        "사실 충실성",
        "리뷰에서 실제 언급된 내용인가?",
    ),
    "core_explanatory_power": (
        "핵심 설명력",
        "핵심 구매 판단 요소를 설명하는가?",
    ),
    "information_coverage": (
        "정보 포괄성",
        "중요 정보를 충분히 포괄하는가?",
    ),
}
CLUSTER_SCORE_LABELS = {
    "cohesion": (
        "군집 응집성",
        "모든 구성원이 하나의 동일한 개념인가?",
    ),
    "canonical_label_fit": (
        "대표명 적합성",
        "대표명이 구성원 전체를 잘 설명하는가?",
    ),
}
_PARQUET_READ_LOCK = threading.Lock()


def _configure_arrow_memory_pool() -> None:
    """Avoid mimalloc crashes when Arrow allocates from Streamlit worker threads."""
    if pa.default_memory_pool().backend_name == "mimalloc":
        pa.set_memory_pool(pa.system_memory_pool())


_configure_arrow_memory_pool()


def _detach_arrow_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialize Arrow-backed strings as Python objects before session caching.

    Pandas can rebuild an all-string row as a new Arrow array during ``iloc``.
    Streamlit performs that operation in a worker thread, which triggered the
    native allocator crash observed at the review-to-cluster transition.
    """
    detached = frame.copy()
    for column, dtype in detached.dtypes.items():
        if isinstance(dtype, pd.StringDtype) and dtype.storage == "pyarrow":
            detached[column] = detached[column].astype(object)
    return detached


def _task_record_at(frame: pd.DataFrame, position: int) -> dict[str, Any]:
    """Read one task without constructing a homogeneous Pandas row Series."""
    if not 0 <= position < len(frame):
        raise IndexError(f"Task position {position} is outside 0..{len(frame) - 1}.")
    return {column: frame[column].iat[position] for column in frame.columns}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate configured review and cluster task cohorts with Streamlit."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--cluster-tasks", type=Path)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    return args


def load_evaluation_inputs(tasks_path: Path, reviews_path: Path) -> pd.DataFrame:
    """Join review text at runtime by review_idx without persisting it to task artifacts."""
    tasks = _detach_arrow_string_columns(
        pd.read_parquet(tasks_path, engine="pyarrow", use_threads=False)
    )
    if tuple(tasks.columns) != TASK_COLUMNS:
        raise ValueError(f"Task columns must be exactly {list(TASK_COLUMNS)}.")
    if tasks.empty or tasks["review_idx"].isna().any() or not tasks["review_idx"].is_unique:
        raise ValueError("Tasks must contain unique, non-null review_idx values.")
    for column in TASK_COLUMNS[1:]:
        for value in tasks[column]:
            try:
                decoded = json.loads(str(value))
            except json.JSONDecodeError as error:
                raise ValueError(f"{column} must contain JSON arrays.") from error
            if (
                not isinstance(decoded, list)
                or not decoded
                or not all(isinstance(item, str) and item.strip() for item in decoded)
            ):
                raise ValueError(f"{column} must contain non-empty JSON arrays of strings.")

    reviews = _detach_arrow_string_columns(
        pd.read_parquet(reviews_path, engine="pyarrow", use_threads=False)
    )
    if "review_idx" in reviews:
        review_key = "review_idx"
    elif "idx" in reviews:
        review_key = "idx"
    else:
        raise ValueError("Review table must contain review_idx or idx.")
    if "review" not in reviews:
        raise ValueError("Review table must contain review text in a review column.")
    if reviews[review_key].isna().any() or not reviews[review_key].is_unique:
        raise ValueError(f"Review table {review_key} must be unique and non-null.")

    display_columns = [review_key, "review"]
    if "productName" in reviews:
        display_columns.append("productName")
    elif "product_name" in reviews:
        display_columns.append("product_name")
    display = reviews[display_columns].rename(columns={review_key: "review_idx"})
    joined = tasks.merge(display, on="review_idx", how="left", validate="one_to_one")
    if joined["review"].isna().any():
        missing = joined.loc[joined["review"].isna(), "review_idx"].head(10).tolist()
        raise ValueError(f"Review text is missing for task review_idx values: {missing}")
    return _detach_arrow_string_columns(joined)


def load_cluster_evaluation_tasks(cluster_tasks_path: Path) -> pd.DataFrame:
    tasks = _detach_arrow_string_columns(
        pd.read_parquet(cluster_tasks_path, engine="pyarrow", use_threads=False)
    )
    if tuple(tasks.columns) != CLUSTER_TASK_COLUMNS:
        raise ValueError(
            f"Cluster task columns must be exactly {list(CLUSTER_TASK_COLUMNS)}."
        )
    if (
        tasks.empty
        or tasks["cluster_validation_id"].isna().any()
        or not tasks["cluster_validation_id"].is_unique
    ):
        raise ValueError("Cluster tasks must contain unique, non-null validation IDs.")
    if tasks[["stage", "cluster_id", "canonical_label"]].isna().any().any():
        raise ValueError("Every cluster task must identify its stage, cluster, and label.")
    for column in ("cluster_validation_id", "stage", "cluster_id", "canonical_label"):
        if tasks[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Every cluster task must have a non-empty {column}.")
    for value in tasks["members"]:
        try:
            decoded = json.loads(str(value))
        except json.JSONDecodeError as error:
            raise ValueError("Cluster members must contain JSON arrays.") from error
        if (
            not isinstance(decoded, list)
            or len(decoded) <= 1
            or not all(isinstance(member, str) and member.strip() for member in decoded)
        ):
            raise ValueError("Every cluster task must have at least two named members.")
    return tasks


def save_completed_evaluation(
    review_responses: pd.DataFrame,
    cluster_responses: pd.DataFrame,
    *,
    expected_review_ids: set[int],
    expected_cluster_ids: set[str],
    output_dir: Path,
    compression: str = "zstd",
) -> Path:
    """Atomically persist one completed evaluator; incomplete sessions stay in memory."""
    if not expected_review_ids or len(expected_review_ids) != len(expected_cluster_ids):
        raise ValueError("Review and cluster completion cohorts must contain the same N.")
    required = {"evaluator_uuid", "review_idx", "experiment", *SCORE_COLUMNS, "preferred"}
    missing = sorted(required - set(review_responses.columns))
    if missing:
        raise ValueError(f"Completed review responses are missing columns: {missing}")
    cluster_required = {
        "evaluator_uuid",
        "cluster_validation_id",
        "stage",
        "cluster_id",
        *CLUSTER_SCORE_COLUMNS,
    }
    cluster_missing = sorted(cluster_required - set(cluster_responses.columns))
    if cluster_missing:
        raise ValueError(
            f"Completed cluster responses are missing columns: {cluster_missing}"
        )
    evaluator_ids = set(review_responses["evaluator_uuid"].astype(str)) | set(
        cluster_responses["evaluator_uuid"].astype(str)
    )
    if len(evaluator_ids) != 1:
        raise ValueError("Completed responses must belong to one evaluator UUID.")
    if set(review_responses["review_idx"]) != expected_review_ids:
        raise ValueError("Completed responses do not cover the full review cohort.")
    expected_rows = len(expected_review_ids) * len(EXPERIMENTS)
    if len(review_responses) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} review ratings; found {len(review_responses)}."
        )
    if review_responses.duplicated(["review_idx", "experiment"]).any():
        raise ValueError("Each review and experiment must have exactly one rating.")
    if set(review_responses["experiment"]) != set(EXPERIMENTS):
        raise ValueError("Completed responses must cover experiments A-D.")
    for column in SCORE_COLUMNS:
        if not review_responses[column].between(1, 5).all():
            raise ValueError(f"{column} ratings must be in [1, 5].")
    preferred_counts = review_responses.groupby("review_idx", observed=True)["preferred"].sum()
    if preferred_counts.lt(1).any():
        raise ValueError("Each review must have at least one preferred experiment.")

    if set(cluster_responses["cluster_validation_id"].astype(str)) != expected_cluster_ids:
        raise ValueError("Completed responses do not cover the full cluster cohort.")
    if len(cluster_responses) != len(expected_cluster_ids):
        raise ValueError(
            f"Expected {len(expected_cluster_ids)} cluster ratings; "
            f"found {len(cluster_responses)}."
        )
    if cluster_responses["cluster_validation_id"].duplicated().any():
        raise ValueError("Each cluster must have exactly one rating.")
    for column in CLUSTER_SCORE_COLUMNS:
        if not cluster_responses[column].between(1, 5).all():
            raise ValueError(f"{column} ratings must be in [1, 5].")

    review_completed = review_responses.copy()
    review_completed.insert(1, "evaluation_level", "review")
    cluster_completed = cluster_responses.copy()
    cluster_completed.insert(1, "evaluation_level", "cluster")
    completed = pd.concat([review_completed, cluster_completed], ignore_index=True, sort=False)
    evaluator_uuid = evaluator_ids.pop()
    destination = output_dir / "completed_evaluators" / f"{evaluator_uuid}.parquet"
    if destination.exists():
        raise FileExistsError(f"Evaluator result already exists: {destination}")
    atomic_write_parquet(completed, destination, compression=compression)
    return destination


def _response_rows(
    evaluator_uuid: str,
    review_idx: int,
    ratings: dict[str, dict[str, int]],
    preferred: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "evaluator_uuid": evaluator_uuid,
            "review_idx": review_idx,
            "experiment": experiment,
            **ratings[experiment],
            "preferred": experiment in preferred,
        }
        for experiment in EXPERIMENTS
    ]


def _cluster_response_row(
    evaluator_uuid: str,
    task: Mapping[str, Any],
    ratings: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "evaluator_uuid": evaluator_uuid,
        "cluster_validation_id": str(task["cluster_validation_id"]),
        "stage": str(task["stage"]),
        "cluster_id": str(task["cluster_id"]),
        **ratings,
    }


def decode_task_results(task: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Decode the four task result arrays while preserving their display order."""
    return {
        experiment: tuple(json.loads(str(task[f"result_{experiment.lower()}"])))
        for experiment in EXPERIMENTS
    }


def matching_attribute_groups(task: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    """Return experiments whose exact attribute-name sets match.

    Attribute order and duplicate occurrences are not meaningful for matching, but
    the strings themselves are compared exactly.
    """
    grouped: dict[tuple[str, ...], list[str]] = {}
    for experiment, values in decode_task_results(task).items():
        signature = tuple(sorted(set(values)))
        grouped.setdefault(signature, []).append(experiment)
    return tuple(tuple(group) for group in grouped.values() if len(group) > 1)


def rating_state_key(review_idx: int, experiment: str, score_column: str) -> str:
    return f"rating::{review_idx}::{experiment}::{score_column}"


def preference_state_key(review_idx: int, experiment: str) -> str:
    return f"preferred::{review_idx}::{experiment}"


def review_card_order(evaluator_uuid: str, review_idx: int) -> tuple[str, ...]:
    """Return a stable, blinded A-D card permutation for one review.

    The evaluator UUID supplies per-session entropy.  Deriving the permutation
    from it (rather than shuffling on every Streamlit rerun) keeps a card in the
    same location while the evaluator is scoring it.
    """
    order = list(EXPERIMENTS)
    random.Random(f"review-card-order-v1::{evaluator_uuid}::{review_idx}").shuffle(order)
    return tuple(order)


def cluster_rating_state_key(cluster_validation_id: str, score_column: str) -> str:
    return f"cluster-rating::{cluster_validation_id}::{score_column}"


def _sync_state_key(review_idx: int, group: Sequence[str], score_column: str) -> str:
    return f"rating-sync::{review_idx}::{','.join(group)}::{score_column}"


def apply_first_rating_sync(
    state: MutableMapping[str, Any],
    *,
    review_idx: int,
    source_experiment: str,
    score_column: str,
    matching_group: Sequence[str],
) -> tuple[str, ...]:
    """Copy a group's first selected score once, without propagating later edits."""
    group = tuple(matching_group)
    if source_experiment not in group:
        raise ValueError(f"{source_experiment} is not in matching group {group}.")

    marker = _sync_state_key(review_idx, group, score_column)
    source_key = rating_state_key(review_idx, source_experiment, score_column)
    source_value = state.get(source_key)
    if state.get(marker, False) or source_value is None:
        return ()

    copied: list[str] = []
    for experiment in group:
        if experiment == source_experiment:
            continue
        target_key = rating_state_key(review_idx, experiment, score_column)
        if state.get(target_key) is None:
            state[target_key] = source_value
            copied.append(experiment)
    state[marker] = True
    return tuple(copied)


def _select_rating(
    review_idx: int,
    source_experiment: str,
    score_column: str,
    score: int,
    matching_group: Sequence[str],
) -> None:
    import streamlit as st

    st.session_state[rating_state_key(review_idx, source_experiment, score_column)] = score
    if matching_group:
        apply_first_rating_sync(
            st.session_state,
            review_idx=review_idx,
            source_experiment=source_experiment,
            score_column=score_column,
            matching_group=matching_group,
        )


def _select_cluster_rating(
    cluster_validation_id: str,
    score_column: str,
    score: int,
) -> None:
    import streamlit as st

    st.session_state[
        cluster_rating_state_key(cluster_validation_id, score_column)
    ] = score


def _render_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
        :root {
            --evaluation-green: #3f6f55;
            --evaluation-ink: #17251c;
            --evaluation-paper: #f6f1e7;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 70% 10%, #eef1e8 0, transparent 38%),
                        var(--evaluation-paper);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container {
            max-width: 1500px;
            padding-top: 2.6rem;
            padding-bottom: 4rem;
        }
        .evaluation-kicker {
            color: var(--evaluation-green);
            font-size: .75rem;
            font-weight: 800;
            letter-spacing: .18em;
            margin-bottom: -.55rem;
        }
        .attribute-chip {
            display: inline-block;
            margin: 0 .35rem .45rem 0;
            padding: .32rem .62rem;
            border-radius: 999px;
            background: #edf2ed;
            color: var(--evaluation-green);
            font-size: .83rem;
            font-weight: 700;
        }
        .score-heading {
            color: var(--evaluation-ink);
            font-size: .9rem;
            font-weight: 800;
            margin-top: .55rem;
        }
        .score-help {
            color: #7a7e79;
            font-size: .75rem;
            margin: -.25rem 0 .2rem;
        }
        .canonical-label-card {
            margin: .8rem 0 1.1rem;
            padding: 1.05rem 1.3rem;
            border-radius: 1rem;
            background: var(--evaluation-green);
            color: white;
        }
        .canonical-label-caption {
            display: block;
            margin-bottom: .25rem;
            font-size: .75rem;
            font-weight: 700;
            opacity: .76;
        }
        .canonical-label-value {
            font-size: 1.7rem;
            font-weight: 800;
        }
        div[class*="score_button_"] button {
            width: 100%;
            min-width: 0;
            aspect-ratio: 1;
            border-radius: 999px !important;
            justify-content: center;
            padding: 0 !important;
        }
        div[class*="score_button_"] button[kind="primary"] {
            background: var(--evaluation-green) !important;
            border-color: var(--evaluation-green) !important;
            color: white !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, .94);
            border-color: rgba(23, 37, 28, .09);
            border-radius: 1.35rem;
            box-shadow: 0 14px 36px rgba(29, 45, 34, .055);
        }
        div[data-testid="stProgress"] > div > div > div {
            background-color: var(--evaluation-green);
        }
        .stButton > button[kind="primary"] {
            border-radius: 999px;
            background: var(--evaluation-ink);
            border-color: var(--evaluation-ink);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@lru_cache(maxsize=4)
def _load_process_cached_inputs(
    tasks_path: str,
    reviews_path: str,
    cluster_tasks_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        load_evaluation_inputs(Path(tasks_path), Path(reviews_path)),
        load_cluster_evaluation_tasks(Path(cluster_tasks_path)),
    )


def _load_tasks_once(
    st: Any,
    tasks_path: Path,
    reviews_path: Path,
    cluster_tasks_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_signature = (str(tasks_path), str(reviews_path), str(cluster_tasks_path))
    if st.session_state.get("input_signature") != input_signature:
        # Streamlit sessions run in separate script threads. Serialize the first
        # native Parquet read and reuse it process-wide for all evaluator sessions.
        with _PARQUET_READ_LOCK:
            review_tasks, cluster_tasks = _load_process_cached_inputs(*input_signature)
        if len(review_tasks) != len(cluster_tasks):
            raise ValueError(
                "Review and cluster evaluation task files must contain the same configured N."
            )
        st.session_state.input_review_tasks = review_tasks
        st.session_state.input_cluster_tasks = cluster_tasks
        st.session_state.input_signature = input_signature
    return st.session_state.input_review_tasks, st.session_state.input_cluster_tasks


def _initialize_evaluation_session(st: Any) -> None:
    input_signature = st.session_state.input_signature
    if st.session_state.get("evaluation_session_signature") == input_signature:
        return

    for key in list(st.session_state):
        if str(key).startswith(
            ("rating::", "preferred::", "rating-sync::", "cluster-rating::")
        ):
            del st.session_state[key]
    st.session_state.evaluator_uuid = str(uuid.uuid4())
    st.session_state.phase = "review"
    st.session_state.position = 0
    st.session_state.cluster_position = 0
    st.session_state.responses = {}
    st.session_state.cluster_responses = {}
    st.session_state.completed_path = None
    st.session_state.evaluation_session_signature = input_signature


def _review_response_frame(st: Any) -> pd.DataFrame:
    rows = [
        row
        for review_rows in st.session_state.responses.values()
        for row in review_rows
    ]
    return pd.DataFrame(rows)


def _cluster_response_frame(st: Any) -> pd.DataFrame:
    return pd.DataFrame(list(st.session_state.cluster_responses.values()))


def _render_cluster_evaluation(
    st: Any,
    *,
    review_tasks: pd.DataFrame,
    cluster_tasks: pd.DataFrame,
    output_dir: Path,
) -> None:
    position = int(st.session_state.cluster_position)
    with st.container(border=True):
        st.markdown("**군집 수준 평가**")
        st.progress(
            position / len(cluster_tasks),
            text=f"{position:,} / {len(cluster_tasks):,} 군집 완료",
        )

    task = _task_record_at(cluster_tasks, position)
    cluster_validation_id = str(task["cluster_validation_id"])
    members = tuple(json.loads(str(task["members"])))
    for score_column in CLUSTER_SCORE_COLUMNS:
        st.session_state.setdefault(
            cluster_rating_state_key(cluster_validation_id, score_column), None
        )

    with st.container(border=True):
        st.caption(f"군집 결과 {position + 1:,} / {len(cluster_tasks):,}")
        if pd.notna(task["parent_aspect_label"]):
            st.markdown(
                '<span class="attribute-chip">상위 속성 · '
                f'{html.escape(str(task["parent_aspect_label"]))}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="canonical-label-card">'
            '<span class="canonical-label-caption">대표명</span>'
            f'<span class="canonical-label-value">'
            f'{html.escape(str(task["canonical_label"]))}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**구성원**")
        member_chips = "".join(
            f'<span class="attribute-chip">{html.escape(member)}</span>'
            for member in members
        )
        st.markdown(member_chips, unsafe_allow_html=True)
        st.divider()

        criterion_columns = st.columns(len(CLUSTER_SCORE_COLUMNS))
        ratings: dict[str, int | None] = {}
        for criterion_column, score_column in zip(
            criterion_columns, CLUSTER_SCORE_COLUMNS, strict=True
        ):
            with criterion_column:
                label, description = CLUSTER_SCORE_LABELS[score_column]
                st.markdown(
                    f'<div class="score-heading">{label}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="score-help">{description}</div>',
                    unsafe_allow_html=True,
                )
                state_key = cluster_rating_state_key(
                    cluster_validation_id, score_column
                )
                score_buttons = st.columns(5, gap="small")
                for score_button, score in zip(
                    score_buttons, range(1, 6), strict=True
                ):
                    with score_button:
                        st.button(
                            str(score),
                            key=(
                                "cluster_score_button_"
                                f"{cluster_validation_id}_{score_column}_{score}"
                            ),
                            type=(
                                "primary"
                                if st.session_state[state_key] == score
                                else "secondary"
                            ),
                            on_click=_select_cluster_rating,
                            args=(cluster_validation_id, score_column, score),
                            help=f"{label} {score}점",
                            use_container_width=True,
                        )
                ratings[score_column] = st.session_state[state_key]

    all_scores_selected = all(value is not None for value in ratings.values())
    if not all_scores_selected:
        st.caption("두 평가 점수를 모두 선택해 주세요.")

    is_last = position + 1 == len(cluster_tasks)
    _, next_column = st.columns([6, 1])
    with next_column:
        next_cluster = st.button(
            "평가 완료 및 저장" if is_last else "다음 군집",
            key=f"next_cluster_{cluster_validation_id}",
            type="primary",
            disabled=not all_scores_selected,
            use_container_width=True,
        )
    if not next_cluster:
        return

    st.session_state.cluster_responses[cluster_validation_id] = _cluster_response_row(
        st.session_state.evaluator_uuid,
        task,
        {score_column: int(ratings[score_column]) for score_column in CLUSTER_SCORE_COLUMNS},
    )
    if not is_last:
        st.session_state.cluster_position = position + 1
        st.rerun()
        return

    try:
        destination = save_completed_evaluation(
            _review_response_frame(st),
            _cluster_response_frame(st),
            expected_review_ids=set(map(int, review_tasks["review_idx"])),
            expected_cluster_ids=set(
                map(str, cluster_tasks["cluster_validation_id"])
            ),
            output_dir=output_dir,
        )
    except Exception as error:  # noqa: BLE001 - surface an atomic-save failure in the UI.
        st.error("완료 응답을 저장하지 못했습니다.")
        st.exception(error)
        return
    st.session_state.completed_path = str(destination)
    st.session_state.phase = "completed"
    st.rerun()


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="상품 속성 사용자 평가", layout="wide")
    _render_styles(st)
    try:
        args = parse_args()
    except SystemExit:
        st.error(
            "입력 경로가 누락되었습니다. 아래 형식으로 앱을 실행해 주세요."
        )
        st.code(
            "uv run --extra user-evaluation user_evaluation.py "
            "--tasks <tasks.parquet> --reviews <reviews.parquet> "
            "--output-dir <directory> [--cluster-tasks <cluster-tasks.parquet>]"
        )
        return

    cluster_tasks_path = (
        args.cluster_tasks
        if args.cluster_tasks is not None
        else args.tasks.with_name("cluster_evaluation_tasks.parquet")
    )
    try:
        review_tasks, cluster_tasks = _load_tasks_once(
            st,
            args.tasks.resolve(),
            args.reviews.resolve(),
            cluster_tasks_path.resolve(),
        )
    except Exception as error:  # noqa: BLE001 - keep serving a visible input diagnostic.
        st.error("평가 입력 데이터를 불러오지 못했습니다.")
        st.exception(error)
        return

    st.markdown(
        '<div class="evaluation-kicker">BLIND ATTRIBUTE EVALUATION</div>',
        unsafe_allow_html=True,
    )
    st.title("상품 속성 사용자 평가")
    st.caption("완료 전 응답은 파일로 저장되지 않습니다.")

    _initialize_evaluation_session(st)
    if st.session_state.completed_path:
        st.success(f"평가가 저장되었습니다: {st.session_state.completed_path}")
        return
    if st.session_state.phase == "cluster":
        try:
            _render_cluster_evaluation(
                st,
                review_tasks=review_tasks,
                cluster_tasks=cluster_tasks,
                output_dir=args.output_dir.resolve(),
            )
        except Exception as error:  # noqa: BLE001 - keep the server available for diagnosis.
            st.error("군집 평가 화면을 표시하지 못했습니다.")
            st.exception(error)
        return

    position = int(st.session_state.position)
    with st.container(border=True):
        st.markdown("**리뷰 수준 평가**")
        st.progress(
            position / len(review_tasks),
            text=f"{position:,} / {len(review_tasks):,} 리뷰 완료",
        )
    if position >= len(review_tasks):
        st.session_state.phase = "cluster"
        st.rerun()
        return

    task = _task_record_at(review_tasks, position)
    review_idx = int(task["review_idx"])
    decoded_results = decode_task_results(task)
    matching_groups = matching_attribute_groups(task)
    group_by_experiment = {
        experiment: group
        for group in matching_groups
        for experiment in group
    }

    with st.container(border=True):
        st.caption(
            f"리뷰 {position + 1:,} / {len(review_tasks):,} · review_idx={review_idx}"
        )
        product_column = "productName" if "productName" in task else "product_name"
        if product_column in task and pd.notna(task[product_column]):
            st.caption(str(task[product_column]))
        st.markdown(f"### {html.escape(str(task['review']))}")

    st.caption(
        "동일한 속성명으로 구성된 결과는 평가 기준별 최초 선택만 같은 점수로 "
        "자동 반영됩니다. "
        "이후 각 결과의 점수는 독립적으로 수정할 수 있습니다."
    )

    for experiment in EXPERIMENTS:
        for score_column in SCORE_COLUMNS:
            st.session_state.setdefault(
                rating_state_key(review_idx, experiment, score_column), None
            )
        st.session_state.setdefault(preference_state_key(review_idx, experiment), False)

    display_order = review_card_order(st.session_state.evaluator_uuid, review_idx)
    ratings: dict[str, dict[str, int | None]] = {}
    columns = st.columns(4)
    for result_number, (column, experiment) in enumerate(
        zip(columns, display_order, strict=True), start=1
    ):
        with column, st.container(border=True):
            title_column, preference_column = st.columns([1.1, 1])
            with title_column:
                st.markdown(f"#### 결과 {result_number}")
            with preference_column:
                st.checkbox(
                    "최종 선호",
                    key=preference_state_key(review_idx, experiment),
                )

            chips = "".join(
                f'<span class="attribute-chip">{html.escape(value)}</span>'
                for value in decoded_results[experiment]
            )
            st.markdown(
                chips or '<span class="attribute-chip">속성 없음</span>',
                unsafe_allow_html=True,
            )
            st.divider()

            ratings[experiment] = {}
            for score_column in SCORE_COLUMNS:
                label, description = SCORE_LABELS[score_column]
                st.markdown(f'<div class="score-heading">{label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="score-help">{description}</div>', unsafe_allow_html=True)
                state_key = rating_state_key(review_idx, experiment, score_column)
                score_buttons = st.columns(5, gap="small")
                for score_button, score in zip(score_buttons, range(1, 6), strict=True):
                    with score_button:
                        st.button(
                            str(score),
                            key=(f"score_button_{review_idx}_{experiment}_{score_column}_{score}"),
                            type=(
                                "primary" if st.session_state[state_key] == score else "secondary"
                            ),
                            on_click=_select_rating,
                            args=(
                                review_idx,
                                experiment,
                                score_column,
                                score,
                                group_by_experiment.get(experiment, ()),
                            ),
                            help=f"결과 {result_number} {label} {score}점",
                            use_container_width=True,
                        )
                ratings[experiment][score_column] = st.session_state[state_key]

    preferred = [
        experiment
        for experiment in EXPERIMENTS
        if st.session_state[preference_state_key(review_idx, experiment)]
    ]
    all_scores_selected = all(
        ratings[experiment][score_column] is not None
        for experiment in EXPERIMENTS
        for score_column in SCORE_COLUMNS
    )
    if not all_scores_selected:
        st.caption("각 결과의 세 평가 점수를 모두 선택해 주세요.")
    if not preferred:
        st.caption(
            "최종 선호 결과를 하나 이상 선택해 주세요. "
            "동률이면 복수 선택할 수 있습니다."
        )

    _, next_column = st.columns([6, 1])
    is_last = position + 1 == len(review_tasks)
    with next_column:
        next_review = st.button(
            "군집 평가로 이동" if is_last else "다음 리뷰",
            key=f"next_review_{review_idx}",
            type="primary",
            disabled=not (all_scores_selected and preferred),
            use_container_width=True,
        )
    if next_review:
        st.session_state.responses[review_idx] = _response_rows(
            st.session_state.evaluator_uuid,
            review_idx,
            {
                experiment: {
                    score_column: int(ratings[experiment][score_column])
                    for score_column in SCORE_COLUMNS
                }
                for experiment in EXPERIMENTS
            },
            preferred,
        )
        st.session_state.position = position + 1
        if is_last:
            st.session_state.phase = "cluster"
        st.rerun()
