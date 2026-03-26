import argparse
import json
import re
import sys
import time
import types
from pathlib import Path


def _prepare_streamlit_stub() -> None:
    """Allow importing app.py in CLI tests without requiring streamlit runtime."""
    try:
        import streamlit  # type: ignore
        return
    except Exception:
        pass

    st_mod = types.ModuleType("streamlit")
    st_mod.set_page_config = lambda *args, **kwargs: None
    st_mod.markdown = lambda *args, **kwargs: None
    st_mod.error = lambda *args, **kwargs: None

    components_mod = types.ModuleType("streamlit.components")
    components_v1_mod = types.ModuleType("streamlit.components.v1")
    components_v1_mod.html = lambda *args, **kwargs: None

    components_mod.v1 = components_v1_mod
    st_mod.components = components_mod

    sys.modules["streamlit"] = st_mod
    sys.modules["streamlit.components"] = components_mod
    sys.modules["streamlit.components.v1"] = components_v1_mod


_prepare_streamlit_stub()

import app
from chatbot_prompt_sets import load_prompt_set


FOCUS_PROMPTS = load_prompt_set("focus_calorie")

PROMPT_LEAK_MARKERS = [
    "h\u00e3y tr\u1ea3 l\u1eddi",
    "hay tra loi",
    "ng\u01b0\u1eddi d\u00f9ng",
    "nguoi dung",
    "ng\u1eef c\u1ea3nh",
    "ngu canh",
    "assistant:",
    "duyet:",
]


def has_garbled_prefix(text: str) -> bool:
    if not text:
        return True

    lower = text.lower().strip()
    if lower.startswith("ột ") or lower.startswith("ot ") or lower.startswith("t mốc"):
        return True

    if "có thể bạn" in lower and not lower.startswith("có thể bạn"):
        return True

    # Starts with a short broken token then punctuation: "ột mốc. ..."
    if re.match(r"^[^A-Za-zÀ-ỹ0-9]{0,2}[A-Za-zÀ-ỹ]{1,3}\s+[A-Za-zÀ-ỹ]{1,5}\.", text):
        return True

    return False


def is_focus_answer_good(question: str, answer: str) -> tuple[bool, str]:
    text = (answer or "").strip()
    if not text:
        return False, "empty"

    lower = text.lower()
    if any(m in lower for m in PROMPT_LEAK_MARKERS):
        return False, "prompt_leak"

    if "http://" in lower or "https://" in lower:
        return False, "url"

    if has_garbled_prefix(text):
        return False, "garbled_prefix"

    if not re.search(r"\d", text):
        return False, "missing_numbers"

    if all(k not in lower for k in ["calo", "kcal"]):
        return False, "missing_calorie_keyword"

    if len(text.split()) < 10:
        return False, "too_short"

    return True, "ok"


def run_once(prompt: str) -> dict:
    started = time.perf_counter()
    try:
        result = app.run_local_chat_query(prompt)
    except Exception as exc:
        result = {"ok": False, "blocked": False, "answer": f"runtime_error: {exc}", "source": "exception"}
    elapsed = time.perf_counter() - started

    answer = str(result.get("answer", "")).strip()
    ok = bool(result.get("ok"))
    blocked = bool(result.get("blocked"))
    source = str(result.get("source", ""))

    quality_ok, reason = is_focus_answer_good(prompt, answer)
    record = {
        "question": prompt,
        "ok": ok,
        "blocked": blocked,
        "source": source,
        "latency_sec": round(elapsed, 4),
        "quality_ok": bool(ok and not blocked and quality_ok),
        "fail_reason": reason if not (ok and not blocked and quality_ok) else "",
        "answer": answer,
        "answer_preview": answer[:180],
    }
    return record


def run_with_recovery(prompt: str) -> tuple[dict, dict | None]:
    primary = run_once(prompt)
    if primary["quality_ok"]:
        return primary, None

    repair_prompt = (
        "Trả lời ngắn gọn bằng tiếng Việt có dấu về mục tiêu calo hôm nay cho người giảm mỡ. "
        "Bắt buộc có số kcal đề xuất và 1 câu hướng dẫn chia bữa."
    )
    repaired = run_once(repair_prompt)
    return primary, repaired


def append_feedback_jsonl(items: list[dict]) -> None:
    out = Path("data") / "chat_feedback.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for item in items:
            payload = {
                "timestamp": int(time.time()),
                "question": item["question"],
                "answer": item["answer"],
                "rating": "up" if item["quality_ok"] else "down",
                "source": "feedbackloop_focus_stress_vi",
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def summarize(records: list[dict], recovered: int, total_failed: int) -> dict:
    total = len(records)
    latencies = [r["latency_sec"] for r in records] if records else [0.0]

    fail_buckets = {}
    for r in records:
        if r["quality_ok"]:
            continue
        key = r["fail_reason"] or "unknown"
        fail_buckets[key] = fail_buckets.get(key, 0) + 1

    quality_ok = sum(1 for r in records if r["quality_ok"])
    blocked = sum(1 for r in records if r["blocked"])

    return {
        "total": total,
        "quality_rate": round(quality_ok / total, 4) if total else 0.0,
        "blocked_rate": round(blocked / total, 4) if total else 0.0,
        "latency_avg_sec": round(sum(latencies) / len(latencies), 4),
        "latency_fastest_sec": round(min(latencies), 4),
        "latency_slowest_sec": round(max(latencies), 4),
        "failures_before_recovery": total_failed,
        "recovered_by_feedback_loop": recovered,
        "recovery_rate": round(recovered / total_failed, 4) if total_failed else 1.0,
        "fail_buckets": fail_buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused heavy stress test for Vietnamese calorie-consult responses")
    parser.add_argument("--rounds", type=int, default=12, help="Stress rounds over focused prompts")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests")
    args = parser.parse_args()

    app.CONFIG["hf_max_new_tokens"] = 96
    app.CONFIG["top_k"] = 2

    app.ensure_models_loaded()

    all_primary = []
    all_logged = []
    recovered = 0
    failed = 0

    for _ in range(args.rounds):
        for p in FOCUS_PROMPTS:
            primary, retry = run_with_recovery(p)
            all_primary.append(primary)
            all_logged.append(primary)

            if not primary["quality_ok"]:
                failed += 1
                if retry is not None:
                    all_logged.append(retry)
                    if retry["quality_ok"]:
                        recovered += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

    append_feedback_jsonl(all_logged)

    summary = summarize(all_primary, recovered, failed)

    report = {
        "created_at": int(time.time()),
        "config": {
            "rounds": args.rounds,
            "sleep": args.sleep,
            "focus": "daily_calorie_consult",
            "prompt_count": len(FOCUS_PROMPTS),
        },
        "summary": summary,
        "records": all_primary,
    }

    out_path = Path("data") / "feedbackloop_focus_stress_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== FOCUS STRESS TEST (DAILY CALORIE) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
