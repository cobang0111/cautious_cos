#!/usr/bin/env python3
"""
Build PRISM -> Pengram JSONL splits with user-disjoint, chronological protocols.

Outputs:
- seen_train.jsonl: train users, earlier conversations only
- seen_valid.jsonl: same users, later conversations and unseen prompts
- calib_unseen.jsonl: unseen users, support interactions for adaptation/calibration
- test_unseen.jsonl: unseen users, later interactions for evaluation

Each JSONL row is compatible with train_pengram_last_layer_prism.py and contains:
{
  "messages": [...],
  "chosen": str,
  "rejected": str,
  "user_profile_text": str,
  "user_history_text": str,
  "pair_weight": float,
  "meta": {...}
}

Expected input files from the PRISM release:
- survey.jsonl
- conversations.jsonl

This script intentionally uses only released PRISM fields and avoids synthetic augmentation.
"""
from __future__ import annotations

import argparse, json, math, random, statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def normalize_ws(s: Any) -> str:
    return ' '.join(str(s or '').split())


def maybe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {'1','true','yes','y','t'}


def score_to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def clip_text(s: str, max_chars: int) -> str:
    s = normalize_ws(s)
    return s if len(s) <= max_chars else s[: max_chars - 3] + '...'


def survey_profile_text(row: Dict[str, Any], include_demographics: bool = False) -> str:
    parts = []
    self_desc = normalize_ws(row.get('self_description', ''))
    system_string = normalize_ws(row.get('system_string', ''))
    stated_prefs = row.get('stated_prefs', {}) or {}
    if self_desc:
        parts.append(f"Self-description: {self_desc}")
    if stated_prefs:
        order = ['values','creativity','fluency','factuality','diversity','safety','personalisation','helpfulness']
        pref_bits = []
        for k in order:
            if k in stated_prefs and stated_prefs[k] is not None:
                pref_bits.append(f"{k}={stated_prefs[k]}")
        for k,v in stated_prefs.items():
            if k not in order and v is not None:
                pref_bits.append(f"{k}={v}")
        if pref_bits:
            parts.append('Stated preferences: ' + ', '.join(pref_bits))
    if system_string:
        parts.append(f"Preferred default instructions: {system_string}")
    if include_demographics:
        demo_keys = ['country','age','gender','employment_status','education','english_proficiency']
        demo = []
        for k in demo_keys:
            v = row.get(k)
            if v not in (None, '', 'NA'):
                demo.append(f"{k}={v}")
        if demo:
            parts.append('Demographics: ' + ', '.join(demo))
    return '\n'.join(parts) if parts else 'Unknown user profile.'


def collect_history_pairs(
    convs: List[Dict[str, Any]],
    max_items: int,
    max_chars: int,
) -> List[Dict[str, str]]:
    # Note: convs[-0:] is the whole list in Python (-0 == 0), so guard max_items <= 0.
    if max_items <= 0:
        return []
    pairs = []
    for conv in convs[-max_items:]:
        turns = parse_turn_candidates(conv)
        for turn, payload in sorted(turns.items()):
            prompt = clip_text(payload.get("user_prompt", ""), max_chars)
            cands = sorted(payload.get("candidates", []), key=lambda c: c.score, reverse=True)
            if not prompt or len(cands) < 2:
                continue
            best = cands[0]
            neg = cands[1]
            pairs.append({
                "prompt": prompt,
                "chosen": clip_text(best.content, max_chars),
                "rejected": clip_text(neg.content, max_chars),
            })
    return pairs[-max_items:]


def summarize_history(
    convs: List[Dict[str, Any]],
    max_items: int,
    max_chars: int,
    include_prompt: bool = False,
) -> str:
    if max_items <= 0:
        return 'No prior interaction history available.'
    lines = []
    for conv in convs[-max_items:]:
        #fb = normalize_ws(conv.get('open_feedback', ''))
        #if fb:
        #    lines.append(f"Past open feedback: {clip_text(fb, max_chars)}")

        chosen_examples = conv.get('_chosen_examples', [])
        for ex in chosen_examples[:2]:
            if isinstance(ex, dict):
                prompt = clip_text(ex.get("prompt", ""), max_chars)
                chosen = clip_text(ex.get("chosen", ""), max_chars)
            else:
                prompt = ""
                chosen = clip_text(str(ex), max_chars)

            if include_prompt and prompt:
                lines.append(f"Previously preferred interaction: User request: {prompt}")
            if chosen:
                lines.append(f"Previously preferred answer style example: {chosen}")

    return '\n'.join(lines) if lines else 'No prior interaction history available.'

@dataclass
class Candidate:
    content: str
    score: float
    if_chosen: bool
    model_name: str
    within_turn_id: str


def parse_turn_candidates(conv: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    history = conv.get('conversation_history', []) or []
    turns: Dict[int, Dict[str, Any]] = defaultdict(lambda: {'user_prompt': None, 'candidates': []})
    for msg in history:
        role = str(msg.get('role', '')).lower()
        turn = int(msg.get('turn', 0) or 0)
        if role == 'user':
            content = normalize_ws(msg.get('content', ''))
            if content:
                turns[turn]['user_prompt'] = content
        elif role in {'assistant', 'model'}:
            content = normalize_ws(msg.get('content', ''))
            score = score_to_float(msg.get('score'))
            if not content or score is None:
                continue
            turns[turn]['candidates'].append(Candidate(
                content=content,
                score=score,
                if_chosen=maybe_bool(msg.get('if_chosen')),
                model_name=str(msg.get('model_name', 'unknown')),
                within_turn_id=str(msg.get('within_turn_id', '')),
            ))
    return turns


def build_prefix_messages(conv: Dict[str, Any], current_turn: int) -> List[Dict[str, str]]:
    history = conv.get('conversation_history', []) or []
    messages: List[Dict[str, str]] = []
    for msg in history:
        turn = int(msg.get('turn', 0) or 0)
        role = str(msg.get('role', '')).lower()
        if turn > current_turn:
            continue
        if role == 'user':
            content = normalize_ws(msg.get('content', ''))
            if content and turn < current_turn:
                messages.append({'role': 'user', 'content': content})
        elif role in {'assistant', 'model'}:
            if turn >= current_turn:
                continue
            if maybe_bool(msg.get('if_chosen')):
                content = normalize_ws(msg.get('content', ''))
                if content:
                    messages.append({'role': 'assistant', 'content': content})
    # append current user prompt last
    turns = parse_turn_candidates(conv)
    prompt = normalize_ws(turns.get(current_turn, {}).get('user_prompt', ''))
    if prompt:
        messages.append({'role': 'user', 'content': prompt})
    return messages


def valid_conv(conv: Dict[str, Any], only_english: bool, drop_flagged: bool) -> bool:
    if only_english and not maybe_bool(conv.get('en_flag', True)):
        return False
    if drop_flagged:
        if maybe_bool(conv.get('pii_positive')) or maybe_bool(conv.get('pii_manual_flag')):
            return False
        if maybe_bool(conv.get('moderation_flag')):
            return False
    return True


def pair_weight(score_gap: float, turn: int) -> float:
    # Gap-driven confidence; same-model later turns are more trustworthy than first-turn cross-model comparisons.
    gap_term = min(max(score_gap / 20.0, 0.25), 2.0)
    turn_term = 0.55 if turn == 0 else 1.0
    return gap_term * turn_term


def build_pairs_for_conv(
    conv: Dict[str, Any],
    survey_by_user: Dict[str, Dict[str, Any]],
    prior_convs_by_user: Dict[str, List[Dict[str, Any]]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    user_id = str(conv.get('user_id'))
    survey = survey_by_user.get(user_id, {})
    profile_text = survey_profile_text(survey, include_demographics=args.include_demographics)
    history_text = summarize_history(prior_convs_by_user.get(user_id, []), max_items=args.history_conversations, max_chars=args.history_max_chars, include_prompt=args.history_include_prompt)
    history_pairs = collect_history_pairs(prior_convs_by_user.get(user_id, []), max_items=args.history_conversations, max_chars=args.history_max_chars)
    conversation_id = str(conv.get('conversation_id'))
    conversation_type = str(conv.get('conversation_type', 'unknown'))
    created = conv.get('generated_datetime', '')

    turns = parse_turn_candidates(conv)
    rows: List[Dict[str, Any]] = []
    chosen_examples = []
    for turn, payload in sorted(turns.items()):
        prompt = payload['user_prompt']
        cands: List[Candidate] = payload['candidates']
        if not prompt or len(cands) < 2:
            continue
        # drop empties/duplicates
        uniq = {}
        for c in cands:
            if c.content and c.content.upper() != 'EMPTY STRING':
                uniq[(c.content, c.model_name, c.within_turn_id)] = c
        cands = list(uniq.values())
        if len(cands) < 2:
            continue
        cands = sorted(cands, key=lambda c: c.score, reverse=True)
        best = cands[0]
        for neg in cands[1:]:
            gap = best.score - neg.score
            if gap < args.min_score_gap:
                continue
            messages = build_prefix_messages(conv, turn)
            rows.append({
                'messages': messages,
                'chosen': best.content,
                'rejected': neg.content,
                'user_profile_text': profile_text,
                'user_history_text': history_text,
                'user_history_pairs': history_pairs,
                'pair_weight': pair_weight(gap, turn),
                'meta': {
                    'dataset': 'prism',
                    'user_id': user_id,
                    'conversation_id': conversation_id,
                    'generated_datetime': created,
                    'conversation_type': conversation_type,
                    'turn': turn,
                    'score_gap': gap,
                    'chosen_score': best.score,
                    'rejected_score': neg.score,
                    'chosen_model_name': best.model_name,
                    'rejected_model_name': neg.model_name,
                }
            })
        # collect preferred answer snippets for future dynamic history.
        chosen_examples.append({
            "prompt": prompt,
            "chosen": best.content,
        })
    conv['_chosen_examples'] = chosen_examples[:2]
    return rows


def split_users(user_ids: List[str], seed: int, unseen_user_frac: float) -> Tuple[set, set]:
    rng = random.Random(seed)
    ids = sorted(set(user_ids))
    rng.shuffle(ids)
    n_unseen = max(1, int(round(len(ids) * unseen_user_frac)))
    unseen = set(ids[:n_unseen])
    seen = set(ids[n_unseen:])
    return seen, unseen


def temporal_split_for_user(convs: List[Dict[str, Any]], support_frac: float, min_support: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    convs = sorted(convs, key=lambda x: (str(x.get('generated_datetime', '')), str(x.get('conversation_id', ''))))
    if len(convs) <= min_support:
        return convs[:-1], convs[-1:]
    n_support = max(min_support, int(math.floor(len(convs) * support_frac)))
    n_support = min(n_support, len(convs) - 1)
    return convs[:n_support], convs[n_support:]


def clear_eval_history(row: Dict[str, Any], output_split: str) -> Dict[str, Any]:
    out = dict(row)
    out['user_history_text'] = ''
    out['user_history_pairs'] = []
    meta = dict(out.get('meta', {}) or {})
    meta['output_split'] = output_split
    meta['history_len'] = 0
    out['meta'] = meta
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--survey_jsonl', type=Path, required=True)
    ap.add_argument('--conversations_jsonl', type=Path, required=True)
    ap.add_argument('--out_dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--unseen_user_frac', type=float, default=0.2)
    ap.add_argument('--seen_valid_frac', type=float, default=0.15)
    ap.add_argument('--unseen_support_frac', type=float, default=0.5)
    ap.add_argument('--min_support_conversations', type=int, default=2)
    ap.add_argument('--min_score_gap', type=float, default=8.0)
    ap.add_argument('--history_conversations', type=int, default=8)
    ap.add_argument('--history_max_chars', type=int, default=256)
    ap.add_argument('--only_english', action='store_true')
    ap.add_argument('--drop_flagged', action='store_true')
    ap.add_argument('--include_demographics', action='store_true')
    ap.add_argument('--history_include_prompt', action='store_true')
    args = ap.parse_args()

    surveys = read_jsonl(args.survey_jsonl)
    conversations = read_jsonl(args.conversations_jsonl)
    survey_by_user = {str(r['user_id']): r for r in surveys if 'user_id' in r}

    conversations = [c for c in conversations if 'user_id' in c and 'conversation_id' in c and valid_conv(c, args.only_english, args.drop_flagged)]

    convs_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in conversations:
        convs_by_user[str(c['user_id'])].append(c)

    seen_users, unseen_users = split_users(list(convs_by_user.keys()), seed=args.seed, unseen_user_frac=args.unseen_user_frac)

    seen_train, seen_valid, calib_unseen, test_unseen = [], [], [], []

    # Seen users: chronological train/valid split
    for user_id in sorted(seen_users):
        user_convs = sorted(convs_by_user[user_id], key=lambda x: (str(x.get('generated_datetime', '')), str(x.get('conversation_id', ''))))
        n_valid = max(1, int(math.floor(len(user_convs) * args.seen_valid_frac))) if len(user_convs) > 1 else 0
        train_convs = user_convs[:-n_valid] if n_valid > 0 else user_convs
        valid_convs = user_convs[-n_valid:] if n_valid > 0 else []
        prior = []
        for conv in train_convs:
            seen_train.extend(build_pairs_for_conv(conv, survey_by_user, {user_id: prior}, args))
            prior = prior + [conv]
        for conv in valid_convs:
            seen_valid.extend(build_pairs_for_conv(conv, survey_by_user, {user_id: prior}, args))
            prior = prior + [conv]

    # Unseen users: support/calibration and test split
    for user_id in sorted(unseen_users):
        support_convs, test_convs = temporal_split_for_user(convs_by_user[user_id], args.unseen_support_frac, args.min_support_conversations)
        prior = []
        for conv in support_convs:
            calib_unseen.extend(
                clear_eval_history(row, 'calib_unseen')
                for row in build_pairs_for_conv(conv, survey_by_user, {user_id: prior}, args)
            )
            prior = prior + [conv]
        for conv in test_convs:
            test_unseen.extend(
                clear_eval_history(row, 'test_unseen')
                for row in build_pairs_for_conv(conv, survey_by_user, {user_id: prior}, args)
            )
            prior = prior + [conv]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / 'seen_train.jsonl', seen_train)
    write_jsonl(args.out_dir / 'seen_valid.jsonl', seen_valid)
    write_jsonl(args.out_dir / 'calib_unseen.jsonl', calib_unseen)
    write_jsonl(args.out_dir / 'test_unseen.jsonl', test_unseen)

    summary = {
        'n_users_total': len(convs_by_user),
        'n_users_seen': len(seen_users),
        'n_users_unseen': len(unseen_users),
        'rows_seen_train': len(seen_train),
        'rows_seen_valid': len(seen_valid),
        'rows_calib_unseen': len(calib_unseen),
        'rows_test_unseen': len(test_unseen),
        'config': vars(args),
    }
    (args.out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
