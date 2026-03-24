import argparse
import json
import statistics
import time
from pathlib import Path

import app


DEFAULT_PROMPTS = [
    "hi",
    "lap ke hoach giam can trong 1 thang",
    "100g uc ga co bao nhieu protein",
]

TIMEOUT_LIKE_MARKERS = [
    "phan hoi cham",
    "lỗi kết nối",
    "loi ket noi",
    "chua san sang",
]


def normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def is_timeout_like(answer: str) -> bool:
    t = normalize_text(answer)
    return any(marker in t for marker in TIMEOUT_LIKE_MARKERS)


def is_low_quality(question: str, answer: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return True

    lower = a.lower()
    if "http://" in lower or "https://" in lower:
        return True

    words = [w for w in a.replace("\n", " ").split(" ") if w]
    if len(words) >= 12:
        short_ratio = sum(1 for w in words if len(w) <= 2) / len(words)
        if short_ratio > 0.55:
            return True

    if normalize_text(question) == "hi" and len(a) > 180:
        return True

    return False


def run_once(prompt: str) -> dict:
    started = time.perf_counter()
    result = app.run_local_chat_query(prompt)
    elapsed = time.perf_counter() - started

    answer = str(result.get("answer", ""))
    ok = bool(result.get("ok"))
    blocked = bool(result.get("blocked"))
    timeout_like = is_timeout_like(answer)
    low_quality = is_low_quality(prompt, answer)

    return {
        "prompt": prompt,
        "ok": ok,
        "blocked": blocked,
        "timeout_like": timeout_like,
        "low_quality": low_quality,
        "latency_sec": round(elapsed, 3),
        "answer_preview": answer[:180],
    }


def summarize(records: list[dict]) -> dict:
    if not records:
        return {}

    latencies = [r["latency_sec"] for r in records]
    ok_count = sum(1 for r in records if r["ok"])
    blocked_count = sum(1 for r in records if r["blocked"])
    timeout_like_count = sum(1 for r in records if r["timeout_like"])
    low_quality_count = sum(1 for r in records if r.get("low_quality"))

    hi_records = [r for r in records if normalize_text(r["prompt"]) == "hi"]
    hi_ok = all(
        r["ok"] and not r["blocked"] and not r["timeout_like"] and not r.get("low_quality")
        for r in hi_records
    ) if hi_records else False

    p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 2 else latencies[0]

    summary = {
        "total": len(records),
        "ok_rate": round(ok_count / len(records), 4),
        "blocked_rate": round(blocked_count / len(records), 4),
        "timeout_like_rate": round(timeout_like_count / len(records), 4),
        "low_quality_rate": round(low_quality_count / len(records), 4),
        "latency_avg_sec": round(sum(latencies) / len(latencies), 3),
        "latency_p95_sec": round(p95, 3),
        "hi_is_stable": hi_ok,
    }

    is_stable = (
        summary["ok_rate"] >= 0.9
        and summary["timeout_like_rate"] == 0
        and summary["low_quality_rate"] <= 0.1
        and summary["hi_is_stable"]
    )
    summary["stable"] = is_stable
    return summary


def main():
    parser = argparse.ArgumentParser(description="Feedback loop test for local AI model path")
    parser.add_argument("--rounds", type=int, default=3, help="How many rounds over prompt set")
    parser.add_argument("--max-cycles", type=int, default=3, help="Repeat cycle until stable or max cycles")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep between requests (sec)")
    args = parser.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)

    # Warmup once before measuring.
    app.ensure_models_loaded()

    all_cycles = []
    stable = False

    for cycle in range(1, args.max_cycles + 1):
        records = []
        for _ in range(args.rounds):
            for prompt in DEFAULT_PROMPTS:
                rec = run_once(prompt)
                rec["cycle"] = cycle
                records.append(rec)
                time.sleep(args.sleep)

        summary = summarize(records)
        all_cycles.append({"cycle": cycle, "summary": summary, "records": records})

        print(
            f"[Cycle {cycle}] stable={summary.get('stable')} ok_rate={summary.get('ok_rate')} "
            f"timeout_rate={summary.get('timeout_like_rate')} low_quality={summary.get('low_quality_rate')} "
            f"hi_ok={summary.get('hi_is_stable')} p95={summary.get('latency_p95_sec')}s"
        )

        if summary.get("stable"):
            stable = True
            break

    report = {
        "created_at": int(time.time()),
        "rounds": args.rounds,
        "max_cycles": args.max_cycles,
        "stable": stable,
        "cycles": all_cycles,
    }

    out_path = Path("data") / "feedbackloop_model_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written: {out_path}")


if __name__ == "__main__":
    main()
