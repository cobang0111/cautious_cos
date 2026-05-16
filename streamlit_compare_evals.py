#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


SYSTEM_LABELS = {
    "base": "base",
    "icl": "icl",
    "icl_rag": "icl-rag",
    "cos": "cos",
    "cos_history": "cos-history",
    "steer_distill": "cautious-cos",
    "cautious-cos": "cautious-cos",
    "cautious_cos": "cautious-cos",
    "lora_sft": "lora-sft",
}

CAUTIOUS_SYSTEMS = {"steer_distill", "cautious-cos", "cautious_cos"}

GENERATION_METRICS = [
    "bertscore_f1",
    "rouge1_f1",
    "rougeL_f1",
    "gen_time_sec",
    "mean_gen_len",
    "empty_rate",
    "drift_rate",
]

POLICY_METRICS = [
    "policy_preference_acc",
]

MODEL_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?B\b", re.IGNORECASE)


def display_system(system: str) -> str:
    return SYSTEM_LABELS.get(system, system.replace("_", "-"))


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_dataset(summary: Dict[str, Any], run_dir: Path) -> str:
    text = " ".join(
        [
            run_dir.name.lower(),
            str(summary.get("support_jsonl", "")).lower(),
            str(summary.get("query_jsonl", "")).lower(),
        ]
    )
    if "personalllm" in text or "personal" in text:
        return "personalllm"
    if "ultrafeedback" in text or "p_4" in text or "survey_16" in text:
        return "ultrafeedback"
    if "psoups" in text:
        return "psoups"
    if "tldr" in text:
        return "tldr"
    if "prism" in text:
        return "prism"
    return run_dir.name


def strip_suffix(value: str, suffix: str) -> str:
    return value[: -len(suffix)] if value.endswith(suffix) else value


def find_model_token(parts: Sequence[str]) -> Optional[int]:
    for idx, part in enumerate(parts):
        if MODEL_TOKEN_RE.search(part):
            return idx
    return None


def parse_model_version_from_name(name: str, dataset: str) -> Tuple[str, str, str]:
    stem = strip_suffix(name, "_steer_distill")
    stem = strip_suffix(stem, "_cautious_cos")

    prefixes = [
        f"all_eval_{dataset}_",
        "all_eval_",
        "prism_cautious_context_steering_distill_",
    ]
    for prefix in prefixes:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break

    parts = [part for part in stem.split("_") if part]
    if parts and parts[0] == dataset:
        parts = parts[1:]

    model_idx = find_model_token(parts)
    if model_idx is None:
        return name, "", ""

    eval_config = "_".join(parts[:model_idx])
    model_name = parts[model_idx]
    version_name = "_".join(parts[model_idx + 1 :])
    return model_name, version_name, eval_config


def infer_run_metadata(summary: Dict[str, Any], run_dir: Path, dataset: str) -> Dict[str, str]:
    model_name, version_name, eval_config = parse_model_version_from_name(run_dir.name, dataset)

    if not version_name:
        ckpt = normalize_text(summary.get("steering_checkpoint", ""))
        if ckpt:
            ckpt_path = Path(ckpt)
            ckpt_dir = ckpt_path.parent if ckpt_path.suffix else ckpt_path
            if ckpt_dir.name == "last":
                ckpt_dir = ckpt_dir.parent
            ckpt_model, ckpt_version, _ = parse_model_version_from_name(ckpt_dir.name, dataset)
            if ckpt_version:
                model_name, version_name = ckpt_model, ckpt_version

    run_label = model_name
    if version_name:
        run_label = f"{model_name} / {version_name}"
    if eval_config:
        run_label = f"{run_label} ({eval_config})"

    return {
        "model_name": model_name or "unknown",
        "version_name": version_name or "unknown",
        "eval_config": eval_config,
        "run_label": run_label,
    }


def discover_summaries(runs_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    if not runs_dir.exists():
        return pd.DataFrame(), summaries

    for summary_path in sorted(runs_dir.rglob("summary.json")):
        run_dir = summary_path.parent
        try:
            summary = read_json(summary_path)
        except Exception as exc:
            st.warning(f"Could not read {summary_path}: {exc}")
            continue

        run_id = str(run_dir)
        summaries[run_id] = summary
        dataset = infer_dataset(summary, run_dir)
        run_meta = infer_run_metadata(summary, run_dir, dataset)
        budgets = summary.get("budgets", {}) or {}
        for budget, budget_info in budgets.items():
            systems = (budget_info or {}).get("systems", {}) or {}
            for system, metrics in systems.items():
                row = {
                    "dataset": dataset,
                    "budget": str(budget),
                    "system": system,
                    "system_label": display_system(system),
                    "run_dir": run_id,
                    "run_name": run_dir.name,
                    **run_meta,
                }
                for key, value in (metrics or {}).items():
                    if isinstance(value, (int, float)):
                        row[key] = float(value)
                if "gen_time_sec" not in row and "gen_wall_time_mean_per_sample_sec" in row:
                    row["gen_time_sec"] = row["gen_wall_time_mean_per_sample_sec"]
                rows.append(row)

    return pd.DataFrame(rows), summaries


def prediction_path(run_dir: Path, budget: str, system: str) -> Path:
    return run_dir / f"predictions_budget{budget}_{system}.jsonl"


def sample_key(row: Dict[str, Any]) -> str:
    user_id = normalize_text(row.get("user_id", "unknown"))
    conversation_id = normalize_text(row.get("conversation_id", ""))
    turn = normalize_text(row.get("turn", ""))
    chosen_hash = hashlib.sha1(normalize_text(row.get("chosen", "")).encode("utf-8")).hexdigest()[:12]
    if conversation_id:
        return f"{user_id}|{conversation_id}|{turn}|{chosen_hash}"
    return f"{user_id}|{turn}|{chosen_hash}"


def load_predictions(run_dir: Path, budget: str, systems: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for system in systems:
        path = prediction_path(run_dir, budget, system)
        if path.exists():
            out[system] = read_jsonl(path)
    return out


def merge_predictions(predictions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    merged: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for system, rows in predictions.items():
        for row in rows:
            merged.setdefault(sample_key(row), {})[system] = row
    return merged


def pick_common_row(system_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if "base" in system_rows:
        return system_rows["base"]
    return next(iter(system_rows.values()))


def pick_history(system_rows: Dict[str, Dict[str, Any]]) -> str:
    for row in system_rows.values():
        history = normalize_text(row.get("user_history_text", ""))
        if history:
            return history

    for row in system_rows.values():
        pairs = row.get("user_history_pairs", [])
        if isinstance(pairs, list) and pairs:
            lines: List[str] = []
            for idx, pair in enumerate(pairs, start=1):
                prompt = normalize_text(pair.get("prompt", ""))
                chosen = normalize_text(pair.get("chosen", ""))
                rejected = normalize_text(pair.get("rejected", ""))
                if prompt:
                    lines.append(f"Support {idx} prompt: {prompt}")
                if chosen:
                    lines.append(f"Preferred: {chosen}")
                if rejected:
                    lines.append(f"Rejected: {rejected}")
            return "\n".join(lines)
    return ""


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def is_cautious_system(system: str) -> bool:
    return system in CAUTIOUS_SYSTEMS or display_system(system) == "cautious-cos"


def best_cautious_score(system_rows: Dict[str, Dict[str, Any]], metric: str) -> Optional[float]:
    return max(
        (
            score
            for system, row in system_rows.items()
            if is_cautious_system(system) and (score := to_float(row.get(metric))) is not None
        ),
        default=None,
    )


def baseline_scores_for_metric(
    system_rows: Dict[str, Dict[str, Any]],
    baselines: Sequence[str],
    metric: str,
) -> Optional[List[float]]:
    scores: List[float] = []
    for baseline in baselines:
        row = system_rows.get(baseline)
        score = to_float(row.get(metric)) if row is not None else None
        if score is None:
            return None
        scores.append(score)
    return scores


def beats_all_baselines(
    cautious_score: Optional[float],
    baseline_scores: Optional[Sequence[float]],
) -> bool:
    return cautious_score is not None and baseline_scores is not None and bool(baseline_scores) and cautious_score > max(baseline_scores)


def select_ranked_example_keys(
    merged: Dict[str, Dict[str, Dict[str, Any]]],
    systems: Sequence[str],
    max_examples: int,
) -> List[str]:
    selected_baselines = [system for system in systems if not is_cautious_system(system)]
    ranked: List[Tuple[int, float, float, str]] = []

    for key, system_rows in merged.items():
        cautious_rouge = best_cautious_score(system_rows, "rouge1_f1")
        if cautious_rouge is None:
            continue

        cautious_bert = best_cautious_score(system_rows, "bertscore_f1")
        rouge_wins = beats_all_baselines(
            cautious_rouge,
            baseline_scores_for_metric(system_rows, selected_baselines, "rouge1_f1"),
        )
        bert_wins = beats_all_baselines(
            cautious_bert,
            baseline_scores_for_metric(system_rows, selected_baselines, "bertscore_f1"),
        )

        if rouge_wins and bert_wins:
            tier = 0
        elif bert_wins:
            tier = 1
        else:
            tier = 2

        ranked.append((tier, cautious_rouge, cautious_bert if cautious_bert is not None else -1.0, key))

    ranked.sort(key=lambda item: (item[0], -item[1], -item[2], item[3]))
    return [key for _, _, _, key in ranked[:max_examples]]


def short_value(value: Any, max_len: int = 24) -> str:
    text = normalize_text(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def example_title(idx: int, key: str, row: Dict[str, Any], dataset: str) -> str:
    user_id = short_value(row.get("user_id", "unknown"))
    parts = [f"user={user_id}"]
    conversation_id = short_value(row.get("conversation_id", ""))
    turn = row.get("turn", "")

    if dataset == "prism" and turn not in ("", None):
        parts.append(f"turn={turn}")
    elif conversation_id:
        parts.append(f"conversation={conversation_id}")
    else:
        sample_hash = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        parts.append(f"sample={sample_hash}")

    return f"Example {idx}: " + ", ".join(parts)


def metric_table(system_rows: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for system, row in system_rows.items():
        rows.append(
            {
                "system": display_system(system),
                "rouge1_f1": row.get("rouge1_f1"),
                "rougeL_f1": row.get("rougeL_f1"),
                "bertscore_f1": row.get("bertscore_f1"),
                "gen_time_sec": row.get("gen_time_sec", row.get("gen_wall_time_sec")),
            }
        )
    return pd.DataFrame(rows)


def render_metric_bars(df: pd.DataFrame, metrics: Sequence[str]) -> None:
    plot_df = df.copy()
    plot_df["dataset_budget"] = plot_df["dataset"] + " / k=" + plot_df["budget"].astype(str)
    for metric in metrics:
        if metric not in plot_df.columns:
            continue
        metric_df = plot_df.dropna(subset=[metric])
        if metric_df.empty:
            continue
        fig = px.bar(
            metric_df,
            x="dataset_budget",
            y=metric,
            color="system_label",
            barmode="group",
            hover_data=["run_name", "model_name", "version_name", "system"],
            title=metric,
        )
        fig.update_layout(xaxis_title="dataset / support budget", yaxis_title=metric, legend_title="system")
        st.plotly_chart(fig, use_container_width=True)


def render_examples(df: pd.DataFrame) -> None:
    st.subheader("Generation examples")
    st.caption("Examples prioritize cautious-cos wins on both rouge1_f1 and bertscore_f1, then bertscore_f1 wins, then cautious-cos rouge1_f1.")

    datasets = sorted(df["dataset"].dropna().unique().tolist())
    dataset = st.selectbox("Example dataset", datasets)
    dataset_df = df[df["dataset"] == dataset]
    run_options = sorted(dataset_df["run_dir"].unique().tolist())
    run_labels = dataset_df.drop_duplicates("run_dir").set_index("run_dir")["run_label"].to_dict()
    run_dir_str = st.selectbox(
        "Example result directory",
        run_options,
        format_func=lambda x: f"{run_labels.get(x, Path(x).name)} | {Path(x).name}",
    )
    budget = st.selectbox("Example support budget", sorted(dataset_df[dataset_df["run_dir"] == run_dir_str]["budget"].unique().tolist()))

    available_systems = sorted(dataset_df[(dataset_df["run_dir"] == run_dir_str) & (dataset_df["budget"] == budget)]["system"].unique().tolist())
    default_systems = [s for s in ["base", "icl", "cos", "steer_distill"] if s in available_systems]
    systems = st.multiselect(
        "Systems to compare in examples",
        available_systems,
        default=default_systems or available_systems,
        format_func=display_system,
    )
    max_examples = st.slider("Number of examples", min_value=1, max_value=20, value=5)

    predictions = load_predictions(Path(run_dir_str), str(budget), systems)
    if not predictions:
        st.info("No prediction JSONL files found for this selection.")
        return

    missing = [display_system(system) for system in systems if system not in predictions]
    if missing:
        st.warning("Missing prediction files for: " + ", ".join(missing))

    merged = merge_predictions(predictions)
    keys = select_ranked_example_keys(merged, systems, max_examples)
    if not keys:
        st.info("No cautious-cos examples with rouge1_f1 were found for this selection.")
        return

    for idx, key in enumerate(keys, start=1):
        system_rows = merged[key]
        common = pick_common_row(system_rows)
        with st.expander(example_title(idx, key, common, dataset), expanded=(idx == 1)):
            history = pick_history(system_rows)
            if history:
                st.markdown("**Preference history**")
                st.text_area("history", history, height=180, label_visibility="collapsed", key=f"history-{idx}-{key}")
            else:
                st.info("Preference history was not saved in these predictions. Re-run evaluation with the current code to populate it.")

            st.markdown("**Prompt**")
            st.text_area("prompt", common.get("prompt_text", ""), height=140, label_visibility="collapsed", key=f"prompt-{idx}-{key}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Preferred / reference response**")
                st.write(common.get("chosen", ""))
            with col_b:
                st.markdown("**Rejected response**")
                st.write(common.get("rejected", ""))

            st.markdown("**Per-system generations**")
            st.dataframe(metric_table(system_rows), use_container_width=True, hide_index=True)
            for system in systems:
                row = system_rows.get(system)
                if row is None:
                    continue
                st.markdown(f"**{display_system(system)}**")
                st.write(row.get("generated", ""))


def main() -> None:
    st.set_page_config(page_title="Cautious CoS Eval Compare", layout="wide")
    st.title("Cautious CoS Evaluation Comparison")
    st.caption("Compare base, ICL, CoS, and cautious-cos results across all evaluated datasets.")

    with st.sidebar:
        runs_dir = Path(st.text_input("Runs directory", value="runs")).expanduser()
        st.markdown("Run with multiple systems, e.g. `SYSTEMS=\"base icl cos cautious-cos\" bash run_all_cautious_context_steering_distill_evals.sh ...`")

    df, _ = discover_summaries(runs_dir)
    if df.empty:
        st.warning(f"No summary.json files found under {runs_dir}.")
        return

    with st.sidebar:
        models = sorted(df["model_name"].unique().tolist())
        selected_model = st.selectbox("Model", models)

        model_df = df[df["model_name"] == selected_model]
        versions = sorted(model_df["version_name"].unique().tolist())
        selected_version = st.selectbox("Version", versions)

        version_df = model_df[model_df["version_name"] == selected_version]
        run_options = sorted(version_df["run_dir"].unique().tolist())
        run_labels = version_df.drop_duplicates("run_dir").set_index("run_dir")["run_label"].to_dict()
        selected_run_dir = st.selectbox(
            "Run / eval config",
            run_options,
            format_func=lambda x: f"{run_labels.get(x, Path(x).name)} | {Path(x).name}",
        )

        run_df = version_df[version_df["run_dir"] == selected_run_dir]
        datasets = sorted(run_df["dataset"].unique().tolist())
        systems = sorted(run_df["system"].unique().tolist())
        budgets = sorted(run_df["budget"].unique().tolist())
        selected_datasets = st.multiselect("Datasets", datasets, default=datasets)
        selected_systems = st.multiselect(
            "Systems",
            systems,
            default=[s for s in ["base", "icl", "cos", "steer_distill"] if s in systems] or systems,
            format_func=display_system,
        )
        selected_budgets = st.multiselect("Support budgets", budgets, default=budgets)

    filtered = run_df[
        run_df["dataset"].isin(selected_datasets)
        & run_df["system"].isin(selected_systems)
        & run_df["budget"].isin(selected_budgets)
    ].copy()

    if filtered.empty:
        st.info("No rows match the current filters.")
        return

    st.subheader("Metric overview")
    metric_options = [m for m in GENERATION_METRICS + POLICY_METRICS if m in filtered.columns]
    default_metrics = [m for m in ["rouge1_f1", "rougeL_f1", "bertscore_f1", "gen_time_sec", "policy_preference_acc"] if m in metric_options]
    selected_metrics = st.multiselect("Metrics", metric_options, default=default_metrics or metric_options[:4])

    render_metric_bars(filtered, selected_metrics)

    st.subheader("Metric table")
    table_cols = ["dataset", "budget", "model_name", "version_name", "system_label", "run_name"] + selected_metrics
    st.dataframe(
        filtered[table_cols].sort_values(["dataset", "model_name", "version_name", "budget", "system_label", "run_name"]),
        use_container_width=True,
        hide_index=True,
    )

    render_examples(filtered)


if __name__ == "__main__":
    main()
