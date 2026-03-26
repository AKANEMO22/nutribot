import argparse
import json
import re
import sys
import time
import types
from pathlib import Path


def _prepare_streamlit_stub() -> None:
    """Allow importing app.py in non-UI CLI tests without streamlit installed."""
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


VI_PROMPTS = load_prompt_set("fast_vi")


def to_seconds(value: float) -> float:
    return round(value, 3)


def is_good_vi_answer(question: str, answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return False

    lower = text.lower()
    if "http://" in lower or "https://" in lower:
        return False

    prompt_leak_markers = [
        "tra loi ngan gon",
        "ưu tiên dùng tiếng việt",
        "uu tien dung tieng viet",
        "neu nguoi dung chi chao hoi",
        "nguoi dung:",
        "hãy trả lời",
        "hay tra loi",
        "duyet:",
        "assistant:",
    ]
    if any(marker in lower for marker in prompt_leak_markers):
        return False

    # Avoid repetitive/noisy outputs.
    chunks = [c.strip().lower() for c in re.split(r"[.!?\n]+", lower) if c.strip()]
    if len(chunks) >= 3 and len(set(chunks)) <= len(chunks) - 1:
        return False

    # Greeting should be short and natural.
    question_lower = question.lower()
    if ("chao" in question_lower) and len(text) > 220:
        return False

    if "xin chao" in question_lower or question_lower.strip().startswith("chao"):
        if ("chào" not in lower) and ("xin chào" not in lower):
            return False

    if "giam can" in question_lower or "giam mo" in question_lower:
        if all(token not in lower for token in ["kcal", "calo", "protein", "thực đơn", "thuc don"]):
            return False

    if ("uc ga" in question_lower) or ("protein" in question_lower) or ("calo" in question_lower):
        if ("protein" not in lower) or (("calo" not in lower) and ("kcal" not in lower)):
            return False
        if not re.search(r"\d", text):
            return False

    # Basic sanity: avoid suspiciously short non-blocked output.
    if len(text.split()) < 3:
        return False

    return True


def run_query(prompt: str) -> dict:
    started = time.perf_counter()
    result = app.run_local_chat_query(prompt)
    elapsed = time.perf_counter() - started

    answer = str(result.get("answer", "")).strip()
    ok = bool(result.get("ok"))
    blocked = bool(result.get("blocked"))

    quality_ok = ok and (not blocked) and is_good_vi_answer(prompt, answer)
    auto_rating = "up" if quality_ok else "down"

    return {
        "question": prompt,
        "ok": ok,
        "blocked": blocked,
        "quality_ok": quality_ok,
        "rating": auto_rating,
        "latency_sec": to_seconds(elapsed),
        "answer": answer,
        "answer_preview": answer[:180],
    }


def append_feedback_log(items: list[dict]) -> None:
    path = Path("data") / "chat_feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for item in items:
            record = {
                "timestamp": int(time.time()),
                "question": item["question"],
                "answer": item["answer"],
                "rating": item["rating"],
                "source": "feedbackloop_fast_vi_test",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict]) -> dict:
    if not records:
        return {}

    latencies = [r["latency_sec"] for r in records]
    total = len(records)
    ok_count = sum(1 for r in records if r["ok"])
    blocked_count = sum(1 for r in records if r["blocked"])
    quality_count = sum(1 for r in records if r["quality_ok"])

    fastest = min(latencies)
    slowest = max(latencies)
    avg = sum(latencies) / total

    return {
        "total": total,
        "ok_rate": round(ok_count / total, 4),
        "blocked_rate": round(blocked_count / total, 4),
        "quality_rate": round(quality_count / total, 4),
        "latency_fastest_sec": to_seconds(fastest),
        "latency_avg_sec": to_seconds(avg),
        "latency_slowest_sec": to_seconds(slowest),
        "ready_for_demo": (quality_count == total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Feedback loop test nhanh cho chatbot (Tieng Viet)")
    parser.add_argument("--rounds", type=int, default=2, help="So vong test. De mac dinh 2 de feedback loop manh hon.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Nghi giua 2 cau hoi (giay)")
    args = parser.parse_args()

    if args.rounds < 1:
        raise ValueError("--rounds phai >= 1")

    Path("data").mkdir(parents=True, exist_ok=True)

    # Speed-first mode for local test: reduce generation length and retrieval load.
    app.CONFIG["hf_max_new_tokens"] = 96
    app.CONFIG["top_k"] = 2

    # Warmup once so measured latencies are closer to real serving speed.
    app.ensure_models_loaded()

    results = []
    for _ in range(args.rounds):
        for prompt in VI_PROMPTS:
            rec = run_query(prompt)
            results.append(rec)
            if args.sleep > 0:
                time.sleep(args.sleep)

    append_feedback_log(results)
    summary = summarize(results)

    report = {
        "created_at": int(time.time()),
        "config": {
            "rounds": args.rounds,
            "sleep": args.sleep,
            "prompts": VI_PROMPTS,
        },
        "summary": summary,
        "records": results,
    }

    out_path = Path("data") / "feedbackloop_fast_vi_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== FEEDBACK LOOP TEST (VI) ===")
    for idx, r in enumerate(results, start=1):
        print(f"[{idx}] Q: {r['question']}")
        print(f"    ok={r['ok']} blocked={r['blocked']} quality_ok={r['quality_ok']} latency={r['latency_sec']}s")
        print(f"    A: {r['answer_preview']}")

    print("\n=== TONG KET ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDa luu report: {out_path}")


if __name__ == "__main__":
    main()
