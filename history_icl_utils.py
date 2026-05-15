from __future__ import annotations

from typing import Any, Dict, List, Sequence


def stringify_messages(messages: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "unknown")).strip()
        content = msg.get("content", "")
        if isinstance(content, list):
            rendered = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        rendered.append(str(item["text"]))
                    else:
                        rendered.append(str(item))
                else:
                    rendered.append(str(item))
            content = "\n".join(rendered)
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def normalize_ws(value: Any) -> str:
    return " ".join(str(value or "").split())


def clip_text(value: Any, max_chars: int) -> str:
    text = normalize_ws(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 3)] + "..."


def normalize_history_text(value: Any) -> str:
    if isinstance(value, list):
        value = stringify_messages(value)
    return normalize_ws(value)


def extract_history_pairs(record: Dict[str, Any]) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for raw_pair in list(record.get("user_history_pairs", []) or []):
        if not isinstance(raw_pair, dict):
            continue
        prompt = normalize_history_text(raw_pair.get("prompt", ""))
        chosen = normalize_history_text(raw_pair.get("chosen", ""))
        rejected = normalize_history_text(raw_pair.get("rejected", ""))
        if not (prompt or chosen):
            continue
        pair = {"prompt": prompt, "chosen": chosen}
        if rejected:
            pair["rejected"] = rejected
        pairs.append(pair)
    return pairs


def extract_last_user_prompt(row: Dict[str, Any]) -> str:
    messages = row.get("messages", []) or []
    for msg in reversed(messages):
        if str(msg.get("role", "")).lower() == "user":
            return normalize_ws(msg.get("content", ""))
    if "prompt" in row:
        return normalize_ws(row.get("prompt", ""))
    if "prompt_text" in row:
        return normalize_ws(row.get("prompt_text", ""))
    return ""


def build_history_pairs_from_support(
    support_rows: Sequence[Dict[str, Any]],
    max_items: int,
    max_chars: int,
) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    limit = len(support_rows) if max_items <= 0 else min(max_items, len(support_rows))
    for row in support_rows[:limit]:
        prompt = clip_text(extract_last_user_prompt(row), max_chars)
        chosen = clip_text(row.get("chosen", ""), max_chars)
        rejected = clip_text(row.get("rejected", ""), max_chars)
        if not chosen:
            continue
        pair = {"prompt": prompt, "chosen": chosen}
        if rejected:
            pair["rejected"] = rejected
        pairs.append(pair)
    return pairs


def build_history_exemplar_pair_text(
    pair: Dict[str, str],
    index: int,
    include_prompt: bool,
    exemplar_mode: str,
    max_chars: int,
) -> str:
    prompt = clip_text(pair.get("prompt", ""), max_chars)
    chosen = clip_text(pair.get("chosen", ""), max_chars)
    rejected = clip_text(pair.get("rejected", ""), max_chars)

    lines = [f"Example {index}"]
    if include_prompt and prompt:
        lines.append(f"User request: {prompt}")
    lines.append(f"Preferred assistant response: {chosen}")
    if exemplar_mode == "pairwise" and rejected:
        lines.append(f"Less preferred assistant response: {rejected}")
    return "\n".join(lines)


def build_history_exemplar_prefix(
    history_pairs: Sequence[Dict[str, str]],
    intro: str,
    exemplar_mode: str,
    include_prompt: bool,
    include_user_profile: bool,
    user_profile: Any,
    max_chars: int,
    max_items: int = 0,
) -> str:
    intro_text = normalize_ws(intro)
    blocks: List[str] = [intro_text] if intro_text else []

    if include_user_profile:
        profile = clip_text(normalize_history_text(user_profile), max_chars * 2)
        if profile:
            blocks.append("User profile:\n" + profile)

    pairs = list(history_pairs)
    if max_items > 0:
        pairs = pairs[:max_items]
    for i, pair in enumerate(pairs, start=1):
        blocks.append(
            build_history_exemplar_pair_text(
                pair=pair,
                index=i,
                include_prompt=include_prompt,
                exemplar_mode=exemplar_mode,
                max_chars=max_chars,
            )
        )
    return "\n\n".join(block for block in blocks if block.strip())


def build_history_exemplar_prompt(
    base_prompt: str,
    history_pairs: Sequence[Dict[str, str]],
    intro: str,
    exemplar_mode: str,
    include_prompt: bool,
    include_user_profile: bool,
    user_profile: Any,
    max_chars: int,
    max_items: int = 0,
) -> str:
    prefix = build_history_exemplar_prefix(
        history_pairs=history_pairs,
        intro=intro,
        exemplar_mode=exemplar_mode,
        include_prompt=include_prompt,
        include_user_profile=include_user_profile,
        user_profile=user_profile,
        max_chars=max_chars,
        max_items=max_items,
    )
    if not prefix:
        return str(base_prompt).strip()
    return (prefix + "\n\nCurrent interaction:\n" + str(base_prompt)).strip()
