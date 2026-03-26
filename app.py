import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import json
from collections import OrderedDict
from typing import Optional
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import time

import streamlit as st
import streamlit.components.v1 as components

# ================= RAG CHATBOT IMPORTS =================
from rag_chatbot import build_vectorstore, build_rag_chain, check_ollama, CONFIG
from question_filter import QuestionFilter

BASE_DIR = Path(__file__).resolve().parent
FPT_PROJECT_NAME = "FPT University Portal Redesign"
EMBEDDED_BUILD_DIR = BASE_DIR / "streamlit_assets" / "embedded_fpt_build"       
STATIC_SERVER = {"server": None, "thread": None, "port": None}
FEEDBACK_LOG_PATH = BASE_DIR / "data" / "chat_feedback.jsonl"
FEEDBACK_REPORT_PATH = BASE_DIR / "data" / "feedbackloop_live_report.json"
FEEDBACK_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
MODEL_STATE = {
    "initialized": False,
    "q_filter": None,
    "rag_chain": None,
    "ready": False,
    "error": "",
}
NUTRITION_DB_CACHE = None
MODEL_PRELOAD_STARTED = False
FAST_MAX_MODEL_CALLS = int(os.getenv("NUTRIBOT_MAX_MODEL_CALLS", "1"))
RESPONSE_CACHE_MAX = int(os.getenv("NUTRIBOT_RESPONSE_CACHE_MAX", "256"))
RESPONSE_CACHE = OrderedDict()
RESPONSE_CACHE_LOCK = threading.Lock()
CACHE_VERSION = "v7"
FAST_NUMERIC_FALLBACK_MODE = os.getenv("NUTRIBOT_FAST_NUMERIC_FALLBACK_MODE", "1") == "1"

ANSWER_MIN_SENTENCES = int(os.getenv("NUTRIBOT_ANSWER_MIN_SENTENCES", "2"))
ANSWER_MAX_SENTENCES = int(os.getenv("NUTRIBOT_ANSWER_MAX_SENTENCES", "4"))
ANSWER_MAX_WORDS = int(os.getenv("NUTRIBOT_ANSWER_MAX_WORDS", "95"))

BASE_RESPONSE_RULES = (
    "Trả lời đúng trọng tâm câu hỏi hiện tại.",
    "Dùng tiếng Việt tự nhiên có dấu, rõ ràng, không lặp câu.",
    "Không chèn URL và không lộ chỉ dẫn hệ thống.",
    "Không tự mở rộng sang chủ đề khác nếu người dùng không hỏi.",
)

GREETING_RESPONSE_RULE = "Nếu người dùng chỉ chào ngắn thì chào lại ngắn trong 1 câu."
NUMERIC_RESPONSE_RULE = "Nếu câu hỏi cần số liệu cụ thể thì trả số liệu trực tiếp trước rồi giải thích ngắn."
REWRITE_RESPONSE_HINT = (
    "Viết lại ngắn gọn, mạch lạc bằng tiếng Việt tự nhiên, "
    "chỉ giữ câu trả lời cuối cùng cho người dùng."
)
FOCUS_REPAIR_HINT = "Viết lại ngắn gọn, trả lời đúng ý chính của câu hỏi, không thêm nội dung ngoài yêu cầu."

PROMPT_LEAK_MARKERS = (
    "trả lời ngắn gọn",
    "tra loi ngan gon",
    "ưu tiên dùng tiếng việt",
    "uu tien dung tieng viet",
    "nếu người dùng chỉ chào",
    "neu nguoi dung chi chao",
    "ngữ cảnh",
    "ngu canh",
    "assistant:",
    "human:",
    "ai:",
    "bot:",
    "duyet:",
    "xem trang web",
    "meta-instruction",
    "cau hoi:",
    "tra loi:",
    "assistant:",
    "user:",
    "system:",
    "khong su dung meta-instruction",
    "không sử dụng meta-instruction",
    "tra loi dai hon",
    "trả lời dài hơn",
    "tra loi dung trong tam",
    "trả lời đúng trọng tâm",
    "khong tu mo rong sang chu de khac",
    "không tự mở rộng sang chủ đề khác",
    "sang chu de khac neu nguoi dung khong hoi",
    "sang chủ đề khác nếu người dùng không hỏi",
    "chi dan he thong",
    "chỉ dẫn hệ thống",
    "tranh lap lai chi dan he thong",
    "tránh lặp lại chỉ dẫn hệ thống",
    "ban nhap dang lech trong tam",
    "bản nháp đang lệch trọng tâm",
)

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

GREETING_PREFIXES = (
    "hi",
    "hello",
    "hey",
    "xin chao",
    "chao",
)

FOCUS_STOPWORDS = {
    "toi", "ban", "minh", "la", "va", "voi", "cho", "cua", "trong", "ngoai", "nhu", "the", "nao",
    "gi", "sao", "neu", "thi", "mot", "nhung", "nay", "kia", "duoc", "khong", "co", "can", "hay",
    "please", "help", "the", "and", "for", "you", "are", "is", "to", "of", "in", "on", "a", "an",
}

HARD_BLOCK_KEYWORDS = (
    "hack wifi",
    "vuot qua otp",
    "vượt qua otp",
    "phishing",
    "ma doc",
    "mã độc",
    "cai ma doc",
    "cài mã độc",
    "xoa dau vet",
    "xóa dấu vết",
    "bat hop phap",
    "bất hợp pháp",
    "lua dao",
    "lừa đảo",
    "gian lan",
    "đột nhập",
    "dot nhap",
    "cach giet",
    "cách giết",
    "sat hai",
    "sát hại",
    "che tao bom",
    "chế tạo bom",
    "tao bom",
    "chế bom",
)

REFUSAL_MARKERS = (
    "không thể hỗ trợ",
    "khong the ho tro",
    "từ chối",
    "tu choi",
    "bị chặn",
    "bi chan",
    "can't assist",
    "cannot assist",
    "can't help",
    "cannot help",
    "sorry",
)

ENGLISH_COMMON_WORDS = {
    "the", "and", "with", "for", "that", "this", "you", "your", "can", "cannot", "cant",
    "sorry", "assist", "help", "please", "about", "from", "into", "should", "could", "today",
}


def is_greeting_like(text: str) -> bool:
    normalized = normalize_food_name(text)
    if not normalized:
        return False

    compact = re.sub(r"[^a-z0-9\s]", "", normalized)
    compact = re.sub(r"\s+", " ", compact).strip()
    if not compact:
        return False

    # Allow short natural variants like: "chao ban", "xin chao bot", "hello ban".
    for p in GREETING_PREFIXES:
        if compact == p or compact.startswith(p + " "):
            return True

    return False


def should_skip_safety_filter(text: str) -> bool:
    normalized = normalize_food_name(text)
    if not normalized:
        return True

    # Avoid false-positive blocks for simple greetings/short benign messages.
    if is_greeting_like(normalized):
        return True

    if len(normalized) <= 3 and normalized.isalpha():
        return True

    return False


def is_short_greeting(text: str) -> bool:
    return is_greeting_like(text)


def has_weight_loss_goal(text: str) -> bool:
    q = normalize_food_name(text)
    if not q:
        return False
    if "giam mo" in q:
        return True
    return bool(re.search(r"\bgiam\s*(?:\d+\s*)?can\b", q))


def relaxed_ascii_text(text: str) -> str:
    raw = (text or "").lower()
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def has_nutrition_intent(text: str) -> bool:
    normalized = normalize_food_name(text)
    base_intent = any(
        token in normalized
        for token in ("calo", "calories", "protein", "carb", "fat", "beo", "thuc don", "mon an", "giam can", "tang can")
    )
    if base_intent or has_weight_loss_goal(text):
        return True

    relaxed = relaxed_ascii_text(text)
    return any(token in relaxed for token in ("kcal", "calo", "protein", "thuc don", "an uong", "giam can"))


def has_body_metric_signal(text: str) -> bool:
    t = normalize_food_name(text)
    if not t:
        return False
    return bool(re.search(r"\b\d{2,3}\s*kg\b", t) or re.search(r"\b1[.,]\d{1,2}\s*m\b", t) or re.search(r"\bcm\b", t))


def has_meal_plan_intent(question: str) -> bool:
    q = normalize_food_name(question)
    plan_markers = ("ke hoach", "lap ke hoach", "an uong", "thuc don")
    time_markers = ("1 thang", "mot thang", "trong vong", "4 tuan")
    has_goal = has_weight_loss_goal(question)
    if q and any(m in q for m in plan_markers) and (has_goal or any(m in q for m in time_markers)):
        return True

    relaxed = relaxed_ascii_text(question)
    return any(m in relaxed for m in plan_markers) and ("thang" in relaxed or "tuan" in relaxed or "giam" in relaxed)


def needs_numeric_response(question: str) -> bool:
    q = normalize_food_name(question)
    if not q:
        return False

    numeric_markers = (
        "bao nhieu",
        "kcal",
        "calo",
        "protein",
        "thuc don",
        "chia protein",
        "muc tieu",
        "ke hoach an",
        "lap ke hoach",
    )
    return any(marker in q for marker in numeric_markers) or has_meal_plan_intent(question)


def has_numeric_signal(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return False
    return bool(re.search(r"\d", text))


def get_answer_max_len(question: str) -> int:
    return 460 if has_meal_plan_intent(question) else 320


def looks_low_quality_answer(question: str, answer: str) -> bool:
    q = normalize_food_name(question)
    a = (answer or "").strip()
    if not a:
        return True

    a_lower = a.lower()
    if is_short_greeting(question):
        if "http://" in a_lower or "https://" in a_lower:
            return True
        if len(a) > 260:
            return True

    words = re.findall(r"[A-Za-z0-9À-ỹà-ỹ]+", a)
    if len(words) >= 12:
        short_ratio = sum(1 for w in words if len(w) <= 2) / max(1, len(words))
        if short_ratio > 0.55:
            return True

    if q in {"hi", "hello", "hey", "xin chao", "chao"} and len(a) > 180:
        return True

    if has_body_metric_signal(question):
        informative_tokens = ("bmi", "kcal", "calo", "chi so", "muc tieu", "thuc don")
        if not any(tok in a_lower for tok in informative_tokens):
            return True

    if ("ke hoach" in q) and not any(tok in a_lower for tok in ("ngay", "tuan", "buoc", "muc tieu", "calo", "kcal")):
        return True

    return False


def looks_unaccented_vietnamese(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return False

    lower = text.lower()
    vietnamese_plain_markers = [" ban ", " toi ", " dinh duong", " calo", " thuc don", " giam can"]
    has_plain_marker = any(m in f" {lower} " for m in vietnamese_plain_markers)
    has_diacritic = bool(re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", lower))
    return has_plain_marker and not has_diacritic


def sanitize_answer_text(question: str, answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""

    text = URL_RE.sub("", text)
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n")]

    cleaned_lines = []
    seen_norm = set()
    for ln in lines:
        if not ln:
            continue

        ln_lower = ln.lower()
        if any(marker in ln_lower for marker in PROMPT_LEAK_MARKERS):
            continue

        norm = normalize_food_name(ln)
        if not norm:
            continue
        if norm in seen_norm:
            continue

        seen_norm.add(norm)
        cleaned_lines.append(ln)

    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("**", "")

    text = re.sub(r"^\s*[,.;:!?-]+\s*", "", text)
    text = re.sub(r"\bNguoi dung:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bNgười dùng:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAI:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAssistant:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bHuman:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bUser:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSystem:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[A-Za-zÀ-ỹ]{1,3},\s+", "", text)

    # Keep only the final answer section if model echoes instruction wrappers.
    for marker in ("Câu trả lời cuối cùng:", "Cau tra loi cuoi cung:"):
        pos = text.find(marker)
        if pos >= 0:
            text = text[pos + len(marker):].strip()

    # Remove leaked conversation wrappers such as "Human: ..." from model templates.
    for marker in ("Human:", "Người dùng:", "Nguoi dung:", "Assistant:", "AI:", "System:"):
        pos = text.lower().find(marker.lower())
        if pos >= 0:
            text = text[:pos].strip()
            break

    text = re.sub(r"không lặp lại chỉ dẫn hệ thống\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"tránh lặp lại chỉ dẫn hệ thống\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"tranh lap lai chi dan he thong\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"không hướng dẫn hệ thống\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"chi dan he thong\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"chỉ dẫn hệ thống\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"chỉ trả lời câu cuối cùng cho người dùng\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"không trích dẫn tin nhắn trước đó\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"không meta-instruction\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    sentence_parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    dedup_sentences = []
    seen_sentence = set()
    for sent in sentence_parts:
        key = normalize_food_name(sent)
        if not key or key in seen_sentence:
            continue
        seen_sentence.add(key)
        dedup_sentences.append(sent)
    if dedup_sentences:
        text = " ".join(dedup_sentences).strip()

    # Fix truncated/garbled prefix patterns like "ột mốc. Có thể bạn..." by cutting to
    # the first coherent Vietnamese sentence starter when present.
    for marker in ("Có thể bạn", "Bạn có thể", "Mình gợi ý", "Hôm nay bạn"):
        idx = text.find(marker)
        if idx > 0:
            text = text[idx:]
            break

    if len(text) > 380:
        parts = re.split(r"(?<=[.!?])\s+", text)
        text = " ".join(parts[:3]).strip()

    if looks_unaccented_vietnamese(text):
        return ""

    meta_leak_signals = (
        "chi dan he thong",
        "chỉ dẫn hệ thống",
        "meta-instruction",
        "yeu cau phan hoi",
        "yêu cầu phản hồi",
        "khong qua",
        "không quá",
    )
    if any(sig in text.lower() for sig in meta_leak_signals):
        return ""

    return text


def sanitize_answer_text_loose(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""

    text = URL_RE.sub("", text)
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n") if ln.strip()]

    kept = []
    seen = set()
    for ln in lines:
        ln_lower = ln.lower()
        if any(marker in ln_lower for marker in PROMPT_LEAK_MARKERS):
            continue
        norm = normalize_food_name(ln)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        kept.append(ln)

    text = " ".join(kept).strip() if kept else text
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(Nguoi dung|Người dùng|Assistant|Human|AI|User|System|Bot)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[,.;:!?-]+\s*", "", text)

    if len(text) > 260:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if parts:
            text = " ".join(parts[:2]).strip()
        else:
            text = text[:260].strip()

    leak_markers = (
        "yêu cầu phản hồi",
        "yeu cau phan hoi",
        "trả lời đúng trọng tâm",
        "tra loi dung trong tam",
        "không quá",
        "khong qua",
        "ban nhap dang lech trong tam",
        "bản nháp đang lệch trọng tâm",
        "bản nháp đang lệch trọng tâm",
        "ban nhap dang lech trong tam",
    )
    if any(m in text.lower() for m in leak_markers):
        return ""

    return text


def finalize_display_answer(answer: str, max_len: int = 320) -> str:
    text = re.sub(r"\s+", " ", (answer or "").strip())
    if not text:
        return ""

    if len(text) > max_len:
        text = text[:max_len].strip()

    if text and text[-1] not in ".!?":
        punct_idx = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if punct_idx >= 20:
            text = text[: punct_idx + 1].strip()
        elif len(text) >= 30:
            text = text.rstrip(" ,;:-") + "."

    return text


def extract_focus_tokens(text: str) -> set[str]:
    norm = normalize_food_name(text)
    tokens = re.findall(r"[a-z0-9]+", norm)
    return {t for t in tokens if len(t) >= 3 and t not in FOCUS_STOPWORDS}


def is_off_topic_answer(question: str, answer: str) -> bool:
    if not question or not answer:
        return False
    if is_short_greeting(question):
        return False

    q_tokens = extract_focus_tokens(question)
    if not q_tokens:
        return False

    a_tokens = extract_focus_tokens(answer)
    overlap = len(q_tokens.intersection(a_tokens))

    # Generic topic check: for contentful questions, at least one shared focus token.
    return len(q_tokens) >= 2 and overlap == 0


def is_explicitly_dangerous_query(text: str) -> bool:
    q = normalize_food_name(text)
    if not q:
        return False
    return any(k in q for k in HARD_BLOCK_KEYWORDS)


def extract_weight_from_text(text: str) -> Optional[float]:
    q = normalize_food_name(text)
    m = re.search(r"\b(\d{2,3})\s*kg\b", q)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_kcal_target_from_text(text: str) -> Optional[int]:
    q = normalize_food_name(text)
    for match in re.finditer(r"\b(\d{3,4})\b", q):
        try:
            value = int(match.group(1))
        except Exception:
            continue
        if 1000 <= value <= 4000:
            return value
    return None


def extract_target_loss_kg(text: str) -> Optional[float]:
    q = normalize_food_name(text)
    if not q:
        return None

    # Examples: "giam 1 can", "giam 2kg", "giam 1.5 kg"
    patterns = [
        r"\bgiam\s*(\d+(?:[\.,]\d+)?)\s*can\b",
        r"\bgiam\s*(\d+(?:[\.,]\d+)?)\s*kg\b",
    ]
    for p in patterns:
        m = re.search(p, q)
        if not m:
            continue
        try:
            value = float(m.group(1).replace(",", "."))
            if 0.2 <= value <= 20:
                return value
        except Exception:
            continue
    return None


def build_numeric_nutrition_fallback(question: str) -> Optional[str]:
    q = normalize_food_name(question)
    if not q or not has_nutrition_intent(question):
        return None

    weight = extract_weight_from_text(question)
    kcal_target = extract_kcal_target_from_text(question)
    if kcal_target is None:
        if weight:
            kcal_target = max(1200, int(weight * 24 - 400))
        else:
            kcal_target = 1600

    if weight:
        protein_low = int(round(1.6 * weight))
        protein_high = int(round(2.2 * weight))
    else:
        protein_low, protein_high = 90, 130

    if has_meal_plan_intent(question):
        target_loss = extract_target_loss_kg(question)
        if target_loss is None:
            target_loss = 1.0

        weekly_loss = round(target_loss / 4.0, 2)
        daily_deficit = int(round((target_loss * 7700) / 30.0))
        daily_deficit = max(180, min(550, daily_deficit))

        breakfast = int(round(kcal_target * 0.25))
        lunch = int(round(kcal_target * 0.35))
        snack = int(round(kcal_target * 0.10))
        dinner = max(250, kcal_target - breakfast - lunch - snack)
        return (
            f"Kế hoạch 1 tháng để giảm khoảng {target_loss:.1f}kg: đặt mức {kcal_target} kcal/ngày, "
            f"protein {protein_low}-{protein_high}g/ngày, thâm hụt trung bình ~{daily_deficit} kcal/ngày "
            f"(mục tiêu ~{weekly_loss:.2f}kg/tuần). "
            f"Phân bổ mỗi ngày: sáng {breakfast} kcal, trưa {lunch} kcal, xế {snack} kcal, tối {dinner} kcal. "
            "Tuần 1 theo đúng khung và cân 3 lần; tuần 2 tăng rau + giữ đạm; tuần 3 giảm đồ ngọt/chiên còn 1-2 bữa/tuần; "
            "tuần 4 nếu cân đứng thì giảm thêm 100 kcal hoặc tăng 1500-2000 bước/ngày."
        )

    if "thuc don" in q:
        breakfast = int(round(kcal_target * 0.25))
        lunch = int(round(kcal_target * 0.35))
        snack = int(round(kcal_target * 0.10))
        dinner = max(250, kcal_target - breakfast - lunch - snack)
        return (
            f"Gợi ý thực đơn khoảng {kcal_target} kcal/ngày cho người bận rộn: "
            f"sáng {breakfast} kcal, trưa {lunch} kcal, xế {snack} kcal, tối {dinner} kcal. "
            f"Mỗi bữa chính ưu tiên đạm nạc + rau; tổng protein nên ở mức {protein_low}-{protein_high}g/ngày để giữ cơ."
        )

    if "protein" in q:
        meals = 4 if "bua" in q else 3
        per_meal_low = int(round(protein_low / meals))
        per_meal_high = int(round(protein_high / meals))
        return (
            f"Bạn nên đặt tổng protein khoảng {protein_low}-{protein_high}g/ngày và chia {meals} bữa, "
            f"mỗi bữa khoảng {per_meal_low}-{per_meal_high}g protein. "
            "Ưu tiên ức gà, cá, trứng, sữa chua Hy Lạp, đậu hũ để giữ cơ khi giảm mỡ."
        )

    if ("truoc buoi tap" in q or "truoc tap" in q) and ("sau buoi tap" in q or "sau tap" in q):
        return (
            "Trước tập 60-90 phút: 30-50g carb dễ tiêu + 15-25g protein. "
            "Sau tập trong 1-2 giờ: 25-35g protein + 40-70g carb để phục hồi cơ và hỗ trợ giảm mỡ."
        )

    return (
        f"Mục tiêu phù hợp để giảm mỡ an toàn là khoảng {kcal_target} kcal/ngày, "
        f"kèm {protein_low}-{protein_high}g protein/ngày. Theo dõi 2 tuần rồi điều chỉnh thêm 100-150 kcal nếu cần."
    )


def looks_refusal_answer(answer: str) -> bool:
    lower = (answer or "").strip().lower()
    if not lower:
        return False
    return any(marker in lower for marker in REFUSAL_MARKERS)


def normalize_refusal_answer(answer: str) -> str:
    if looks_refusal_answer(answer):
        return "Mình không thể hỗ trợ yêu cầu này vì lý do an toàn. Nếu bạn muốn, mình có thể hỗ trợ nội dung dinh dưỡng an toàn hơn."
    return (answer or "").strip()


def looks_mojibake_text(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return False

    markers = ("Ã", "á»", "â€", "Æ°", "Ä‘", "ï¸", "�")
    return any(m in text for m in markers)


def is_english_dominant(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return False

    words = re.findall(r"[a-z]+", text)
    if len(words) < 6:
        return False

    english_hits = sum(1 for w in words if w in ENGLISH_COMMON_WORDS)
    accented_hits = len(re.findall(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", text))

    english_ratio = english_hits / max(1, len(words))
    return english_ratio >= 0.18 and accented_hits == 0


def looks_noisy_answer(question: str, answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True

    lower = text.lower()
    if any(marker in lower for marker in PROMPT_LEAK_MARKERS):
        return True

    if URL_RE.search(text):
        return True

    chunks = [c.strip() for c in re.split(r"[.!?\n]+", text) if c.strip()]
    if len(chunks) >= 3:
        norm_chunks = [normalize_food_name(c) for c in chunks]
        repeated = len(norm_chunks) - len(set(norm_chunks))
        if repeated >= 1:
            return True

    if looks_low_quality_answer(question, text):
        return True

    if text.lower().startswith("câu hỏi") or text.lower().startswith("cau hoi"):
        return True

    return False


def is_focus_sufficient_answer(question: str, answer: str) -> bool:
    q = normalize_food_name(question)
    a = normalize_food_name(answer)
    if not q or not a:
        return False

    if needs_numeric_response(question) and not has_numeric_signal(answer):
        return False

    if has_meal_plan_intent(question):
        structure_tokens = ("sang", "trua", "toi", "bua", "ngay", "tuan", "kcal", "calo", "protein")
        hit_count = sum(1 for token in structure_tokens if token in a)
        if hit_count < 3:
            return False
        if "tuan" not in a:
            return False
        if len((answer or "").split()) < 20:
            return False

    if ("thuc don" in q) and ("1600" in q or "kcal" in q or "calo" in q):
        meal_tokens = ("sang", "trua", "toi", "bua")
        if not any(t in a for t in meal_tokens):
            return False
        if "protein" not in a:
            return False

    if ("chia protein" in q) or (("protein" in q) and ("giu co" in q)):
        per_meal_marker = any(t in a for t in ("moi bua", "bua", "lan an"))
        gram_marker = bool(re.search(r"\b\d+\s*g\b", a))
        if not ("protein" in a and per_meal_marker and gram_marker):
            return False

    return True


def build_feedback_loop_summary(limit: int = 400) -> dict:
    if not FEEDBACK_LOG_PATH.exists():
        return {
            "total": 0,
            "up": 0,
            "down": 0,
            "down_rate": 0.0,
            "greeting_down_rate": 0.0,
        }

    lines = FEEDBACK_LOG_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
    items = []
    for line in lines:
        try:
            items.append(json.loads(line))
        except Exception:
            continue

    total = len(items)
    up = sum(1 for i in items if str(i.get("rating", "")).lower() == "up")
    down = sum(1 for i in items if str(i.get("rating", "")).lower() == "down")
    greeting_items = [i for i in items if is_short_greeting(str(i.get("question", "")))]
    greeting_down = sum(1 for i in greeting_items if str(i.get("rating", "")).lower() == "down")

    return {
        "total": total,
        "up": up,
        "down": down,
        "down_rate": round(down / total, 4) if total else 0.0,
        "greeting_down_rate": round(greeting_down / len(greeting_items), 4) if greeting_items else 0.0,
    }


def normalize_food_name(text: str) -> str:
    base = (text or "").strip().lower()
    base = re.sub(r"\s+", " ", base)
    replacements = str.maketrans(
        {
            "à": "a", "á": "a", "ạ": "a", "ả": "a", "ã": "a",
            "â": "a", "ầ": "a", "ấ": "a", "ậ": "a", "ẩ": "a", "ẫ": "a",
            "ă": "a", "ằ": "a", "ắ": "a", "ặ": "a", "ẳ": "a", "ẵ": "a",
            "è": "e", "é": "e", "ẹ": "e", "ẻ": "e", "ẽ": "e",
            "ê": "e", "ề": "e", "ế": "e", "ệ": "e", "ể": "e", "ễ": "e",
            "ì": "i", "í": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
            "ò": "o", "ó": "o", "ọ": "o", "ỏ": "o", "õ": "o",
            "ô": "o", "ồ": "o", "ố": "o", "ộ": "o", "ổ": "o", "ỗ": "o",
            "ơ": "o", "ờ": "o", "ớ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
            "ù": "u", "ú": "u", "ụ": "u", "ủ": "u", "ũ": "u",
            "ư": "u", "ừ": "u", "ứ": "u", "ự": "u", "ử": "u", "ữ": "u",
            "ỳ": "y", "ý": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y",
            "đ": "d",
        }
    )
    return base.translate(replacements)


def load_nutrition_db_from_dashboard() -> dict:
    global NUTRITION_DB_CACHE
    if isinstance(NUTRITION_DB_CACHE, dict):
        return NUTRITION_DB_CACHE

    nutrition_db = {}
    js_path = BASE_DIR / "streamlit_assets" / "embedded_fpt_build" / "assets" / "dashboard-sync.js"
    if not js_path.exists():
        NUTRITION_DB_CACHE = nutrition_db
        return nutrition_db

    try:
        content = js_path.read_text(encoding="utf-8")
        block_match = re.search(r"const\s+FOOD_NUTRITION_DB\s*=\s*\{(.*?)\};", content, re.DOTALL)
        if not block_match:
            NUTRITION_DB_CACHE = nutrition_db
            return nutrition_db

        entry_re = re.compile(
            r"^\s*(?:\"([^\"]+)\"|([A-Za-zÀ-ỹà-ỹĐđ_][A-Za-z0-9À-ỹà-ỹĐđ_\s]*))\s*:\s*\{\s*"
            r"calories:\s*([0-9.]+)\s*,\s*protein:\s*([0-9.]+)\s*,\s*carbs:\s*([0-9.]+)\s*,\s*fat:\s*([0-9.]+)\s*\}\s*,?\s*$"
        )

        for raw_line in block_match.group(1).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = entry_re.match(line)
            if not m:
                continue

            food_name = (m.group(1) or m.group(2) or "").strip()
            if not food_name:
                continue

            nutrition_db[normalize_food_name(food_name)] = {
                "name": food_name,
                "calories": float(m.group(3)),
                "protein": float(m.group(4)),
                "carbs": float(m.group(5)),
                "fat": float(m.group(6)),
            }
    except Exception:
        nutrition_db = {}

    NUTRITION_DB_CACHE = nutrition_db
    return nutrition_db


def build_nutrition_context(question: str, limit: int = 10) -> str:
    nutrition_db = load_nutrition_db_from_dashboard()
    if not nutrition_db:
        return ""

    normalized_q = normalize_food_name(question)
    matched = []
    for key, info in nutrition_db.items():
        if key and key in normalized_q:
            matched.append(info)

    if not matched:
        return ""

    lines = ["Du lieu chi so dinh duong noi bo (tu module nhap mon an):"]
    for item in matched[:limit]:
        lines.append(
            f"- {item['name']}: {item['calories']:.0f} kcal, protein {item['protein']:.1f}g, carbs {item['carbs']:.1f}g, fat {item['fat']:.1f}g"
        )
    return "\n".join(lines)


def build_answer_prompt(question: str, nutrition_context: str = "") -> str:
    style_rule = (
        f"Ưu tiên câu ngắn và dễ đọc, khoảng {ANSWER_MIN_SENTENCES}-{ANSWER_MAX_SENTENCES} câu "
        f"(không quá {ANSWER_MAX_WORDS} từ)."
    )
    instruction_blocks = [
        style_rule,
        *BASE_RESPONSE_RULES,
        GREETING_RESPONSE_RULE,
        NUMERIC_RESPONSE_RULE,
    ]

    prompt = (
        f"Câu hỏi người dùng: {question}\n"
        f"Yêu cầu phản hồi: {' '.join(instruction_blocks)}"
    )

    if nutrition_context:
        prompt = (
            f"{prompt}\n\n"
            f"{nutrition_context}\n"
            "Nếu dữ liệu nội bộ liên quan thì ưu tiên sử dụng."
        )

    return prompt


def load_rag_and_models():
    q_filter_instance = None
    model_path = BASE_DIR / "weight" / "question_filter_model.pkl"
    if model_path.exists():
        q_filter_instance = QuestionFilter(model_path=str(model_path))

    preferred_backend = str(CONFIG.get("llm_backend", "local_hf")).lower()
    candidates = [preferred_backend] + [b for b in ("local_hf", "ollama") if b != preferred_backend]

    errors = []
    for backend in candidates:
        try:
            CONFIG["llm_backend"] = backend

            # For ollama, skip quickly if server isn't reachable.
            if backend == "ollama" and not check_ollama():
                errors.append("ollama: server không sẵn sàng")
                continue

            # Try full RAG first; if retrieval stack fails, still keep direct-LLM mode.
            vectorstore = None
            rag_error = ""
            try:
                vectorstore = build_vectorstore(force_rebuild=False)
            except Exception as exc:
                rag_error = f"RAG unavailable: {exc}"

            try:
                chain = build_rag_chain(vectorstore)
                return q_filter_instance, chain, True, rag_error
            except Exception as chain_exc:
                errors.append(f"{backend} chain: {chain_exc}")
                continue
        except Exception as exc:
            errors.append(f"{backend}: {exc}")

    return (
        q_filter_instance,
        None,
        False,
        "Không khởi tạo được backend model. " + " | ".join(errors),
    )


def ensure_models_loaded(force_reload: bool = False):
    with MODEL_LOCK:
        if MODEL_STATE["initialized"] and not force_reload:
            return MODEL_STATE

        try:
            q_filter_instance, chain, ready, err = load_rag_and_models()
            MODEL_STATE.update(
                {
                    "initialized": True,
                    "q_filter": q_filter_instance,
                    "rag_chain": chain,
                    "ready": ready,
                    "error": err,
                }
            )
        except Exception as exc:
            MODEL_STATE.update(
                {
                    "initialized": True,
                    "q_filter": None,
                    "rag_chain": None,
                    "ready": False,
                    "error": f"Lỗi khởi tạo model: {exc}",
                }
            )

        return MODEL_STATE


def start_model_preload_once():
    global MODEL_PRELOAD_STARTED
    if MODEL_PRELOAD_STARTED:
        return

    MODEL_PRELOAD_STARTED = True

    def _worker():
        try:
            ensure_models_loaded()
        except Exception:
            # Keep preload failures non-fatal; runtime call will surface actual error.
            pass

    threading.Thread(target=_worker, daemon=True).start()


def _cache_key(text: str) -> str:
    return f"{CACHE_VERSION}:{normalize_food_name((text or '').strip())[:320]}"


def get_cached_answer(text: str) -> Optional[str]:
    key = _cache_key(text)
    if not key:
        return None
    with RESPONSE_CACHE_LOCK:
        cached = RESPONSE_CACHE.get(key)
        if cached is None:
            return None
        if is_off_topic_answer(text, cached):
            RESPONSE_CACHE.pop(key, None)
            return None
        RESPONSE_CACHE.move_to_end(key)
        return cached


def set_cached_answer(text: str, answer: str) -> None:
    key = _cache_key(text)
    clean_answer = (answer or "").strip()
    if not key or not clean_answer:
        return
    with RESPONSE_CACHE_LOCK:
        RESPONSE_CACHE[key] = clean_answer
        RESPONSE_CACHE.move_to_end(key)
        while len(RESPONSE_CACHE) > max(16, RESPONSE_CACHE_MAX):
            RESPONSE_CACHE.popitem(last=False)


def run_local_chat_query(user_text: str) -> dict:
    text = (user_text or "").strip()
    if not text:
        return {"ok": False, "answer": "Vui lòng nhập câu hỏi."}

    # Keep prompt compact for small local models.
    text = text[:320]

    if is_explicitly_dangerous_query(text):
        return {
            "ok": False,
            "blocked": True,
            "answer": "⚠️ Câu hỏi bị chặn bởi bộ lọc an toàn. Mình không thể hỗ trợ nội dung nguy hiểm hoặc bất hợp pháp.",
            "source": "local_rule",
        }

    # Deterministic greeting response to avoid unstable LLM behavior on very short input.
    if is_short_greeting(text):
        answer = "Xin chào bạn. Mình có thể hỗ trợ bạn về calo, thực đơn và kế hoạch giảm mỡ."
        set_cached_answer(text, answer)
        return {"ok": True, "answer": answer, "source": "local_rule"}

    cached_answer = get_cached_answer(text)
    if cached_answer:
        return {"ok": True, "answer": cached_answer, "source": "local_ai_cache"}

    # Fast path for numeric nutrition planning: deterministic formula-based answer
    # to keep latency low and avoid unstable local LLM outputs.
    if FAST_NUMERIC_FALLBACK_MODE and needs_numeric_response(text):
        quick_answer = build_numeric_nutrition_fallback(text)
        if quick_answer:
            quick_answer = finalize_display_answer(quick_answer, get_answer_max_len(text))
            set_cached_answer(text, quick_answer)
            return {"ok": True, "answer": quick_answer, "source": "local_dynamic_fast_path"}

    state = ensure_models_loaded()
    q_filter_instance = state.get("q_filter")
    chain = state.get("rag_chain")

    if not state.get("ready") or chain is None:
        # Auto-retry once to recover from transient init errors.
        state = ensure_models_loaded(force_reload=True)
        q_filter_instance = state.get("q_filter")
        chain = state.get("rag_chain")

    if not state.get("ready") or chain is None:
        detail = state.get("error") or "Model chưa sẵn sàng."
        return {
            "ok": False,
            "answer": f"Hệ thống AI local chưa sẵn sàng. {detail}",
            "source": "local_ai",
        }

    if q_filter_instance and not should_skip_safety_filter(text):
        try:
            is_dangerous = bool(q_filter_instance.is_dangerous(text))
            should_block = is_dangerous

            # Reduce false positives: only hard-block when model is confident enough.
            pipeline = getattr(q_filter_instance, "pipeline", None)
            if is_dangerous and pipeline is not None and hasattr(pipeline, "predict_proba"):
                try:
                    proba = pipeline.predict_proba([text])[0]
                    danger_conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
                    should_block = danger_conf >= float(os.getenv("NUTRIBOT_SAFETY_BLOCK_THRESHOLD", "0.92"))
                except Exception:
                    should_block = is_dangerous

            if should_block:
                return {
                    "ok": False,
                    "blocked": True,
                    "answer": "⚠️ Câu hỏi bị chặn bởi bộ lọc an toàn.",
                    "source": "local_ai",
                }
        except Exception:
            pass

    try:
        last_raw_answer = ""
        answer_max_len = get_answer_max_len(text)
        nutrition_context = build_nutrition_context(text, limit=4)
        use_retrieval = bool(nutrition_context) or has_nutrition_intent(text)

        def call_chain_safe(payload_question: str, skip_retrieval: bool):
            try:
                res = chain({"question": payload_question, "skip_retrieval": skip_retrieval})
                return str(res.get("answer", "")).strip()
            except Exception:
                return ""

        def repair_answer_with_feedback(raw_candidate: str, focus_only: bool = False) -> str:
            candidate = (raw_candidate or "").strip()
            if not candidate:
                return ""

            repair_instruction = (
                FOCUS_REPAIR_HINT if focus_only else REWRITE_RESPONSE_HINT
            )
            numeric_requirement = ""
            if needs_numeric_response(text):
                numeric_requirement = " Bắt buộc có số liệu cụ thể (kcal/protein hoặc số gram phù hợp câu hỏi)."
            if has_meal_plan_intent(text):
                numeric_requirement += " Trả theo khung rõ ràng theo tuần hoặc theo bữa (sáng/trưa/tối), tránh trả lời chung chung."

            prompt = (
                f"{repair_instruction}\n"
                f"Câu hỏi gốc: {text}\n"
                f"Bản nháp hiện tại: {candidate}\n"
                "Ràng buộc: chỉ trả lời tiếng Việt có dấu, không URL, không meta-instruction, "
                f"không lặp lại câu hỏi.{numeric_requirement}"
            )

            rewritten_raw = call_chain_safe(prompt[:420], True)
            rewritten = sanitize_answer_text(text, rewritten_raw)
            if not rewritten:
                return ""

            rewritten = finalize_display_answer(rewritten, answer_max_len)
            if not rewritten:
                return ""

            if not is_focus_sufficient_answer(text, rewritten):
                return ""

            if looks_noisy_answer(text, rewritten) or is_off_topic_answer(text, rewritten):
                return ""
            if looks_mojibake_text(rewritten) or is_english_dominant(rewritten):
                return ""
            return rewritten

        max_calls = max(1, min(3, FAST_MAX_MODEL_CALLS))
        answer = ""
        for attempt in range(max_calls):
            if attempt == 0:
                payload_question = text[:220]
                skip_retrieval = not use_retrieval
            else:
                payload_question = text[:220]
                skip_retrieval = True

            raw = call_chain_safe(payload_question, skip_retrieval)
            last_raw_answer = raw or last_raw_answer
            answer = sanitize_answer_text(text, raw)

            if answer and looks_refusal_answer(answer):
                refusal_answer = normalize_refusal_answer(answer)
                if has_nutrition_intent(text) or is_short_greeting(text):
                    answer = ""
                else:
                    return {
                        "ok": False,
                        "blocked": True,
                        "answer": refusal_answer,
                        "source": "local_ai",
                    }

            if answer and (looks_mojibake_text(answer) or is_english_dominant(answer)):
                repaired = repair_answer_with_feedback(answer, focus_only=False)
                answer = repaired or ""

            if (
                answer
                and not looks_noisy_answer(text, answer)
                and not is_off_topic_answer(text, answer)
                and is_focus_sufficient_answer(text, answer)
            ):
                break

        if answer and is_off_topic_answer(text, answer):
            answer = ""

        if not answer:
            fallback_answer = sanitize_answer_text(text, last_raw_answer or "")
            if (
                fallback_answer
                and not looks_noisy_answer(text, fallback_answer)
                and not is_off_topic_answer(text, fallback_answer)
                and is_focus_sufficient_answer(text, fallback_answer)
                and not looks_mojibake_text(fallback_answer)
                and not is_english_dominant(fallback_answer)
            ):
                fallback_answer = finalize_display_answer(fallback_answer, answer_max_len)
                set_cached_answer(text, fallback_answer)
                return {"ok": True, "answer": fallback_answer, "source": "local_ai"}

            repaired_fallback = repair_answer_with_feedback(last_raw_answer or "", focus_only=True)
            if repaired_fallback:
                set_cached_answer(text, repaired_fallback)
                return {"ok": True, "answer": repaired_fallback, "source": "local_ai_repair"}

            if has_meal_plan_intent(text):
                dynamic_fallback = build_numeric_nutrition_fallback(text)
                if dynamic_fallback:
                    dynamic_fallback = finalize_display_answer(dynamic_fallback, answer_max_len)
                    set_cached_answer(text, dynamic_fallback)
                    return {"ok": True, "answer": dynamic_fallback, "source": "local_dynamic_fallback"}

            if needs_numeric_response(text):
                plan_requirement = ""
                if has_meal_plan_intent(text):
                    plan_requirement = " Bắt buộc nêu kế hoạch theo tuần hoặc theo bữa (sáng/trưa/tối)."
                strict_prompt = (
                    "Trả lời trực tiếp câu hỏi dinh dưỡng sau bằng tiếng Việt có dấu. "
                    "Bắt buộc có số liệu cụ thể phù hợp (kcal/protein hoặc số gram), "
                    f"không hỏi lại người dùng, không URL, không meta-instruction.{plan_requirement} "
                    f"Câu hỏi: {text}"
                )
                strict_raw = call_chain_safe(strict_prompt[:420], True)
                strict_answer = sanitize_answer_text(text, strict_raw)
                if (
                    strict_answer
                    and not looks_noisy_answer(text, strict_answer)
                    and not is_off_topic_answer(text, strict_answer)
                    and is_focus_sufficient_answer(text, strict_answer)
                    and not looks_mojibake_text(strict_answer)
                    and not is_english_dominant(strict_answer)
                ):
                    strict_answer = finalize_display_answer(strict_answer, answer_max_len)
                    set_cached_answer(text, strict_answer)
                    return {"ok": True, "answer": strict_answer, "source": "local_ai_strict_retry"}

            loose_answer = sanitize_answer_text_loose(last_raw_answer or "")
            if (
                loose_answer
                and not looks_mojibake_text(loose_answer)
                and not is_english_dominant(loose_answer)
                and is_focus_sufficient_answer(text, loose_answer)
            ):
                loose_answer = finalize_display_answer(loose_answer, 280)
                set_cached_answer(text, loose_answer)
                return {"ok": True, "answer": loose_answer, "source": "local_ai"}

            dynamic_fallback = build_numeric_nutrition_fallback(text)
            if dynamic_fallback:
                dynamic_fallback = finalize_display_answer(dynamic_fallback, answer_max_len)
                set_cached_answer(text, dynamic_fallback)
                return {"ok": True, "answer": dynamic_fallback, "source": "local_dynamic_fallback"}

            return {
                "ok": False,
                "answer": "AI local đã xử lý nhưng chưa tạo được câu trả lời rõ ràng. Bạn thử diễn đạt cụ thể hơn một chút nhé.",
                "source": "local_ai",
            }

        if not is_focus_sufficient_answer(text, answer):
            repaired_focus = repair_answer_with_feedback(answer, focus_only=True)
            if repaired_focus:
                answer = repaired_focus

        if answer and not is_focus_sufficient_answer(text, answer):
            dynamic_fallback = build_numeric_nutrition_fallback(text) if needs_numeric_response(text) else None
            if dynamic_fallback:
                dynamic_fallback = finalize_display_answer(dynamic_fallback, answer_max_len)
                set_cached_answer(text, dynamic_fallback)
                return {"ok": True, "answer": dynamic_fallback, "source": "local_dynamic_fallback"}
            answer = ""

        if answer and (looks_mojibake_text(answer) or is_english_dominant(answer)):
            repaired = repair_answer_with_feedback(answer, focus_only=False)
            answer = repaired or ""

        if answer and looks_refusal_answer(answer):
            return {
                "ok": False,
                "blocked": True,
                "answer": normalize_refusal_answer(answer),
                "source": "local_ai",
            }

        if not answer:
            return {
                "ok": False,
                "answer": "AI local đã xử lý nhưng chưa tạo được câu trả lời tiếng Việt đủ rõ. Bạn thử nêu câu hỏi cụ thể hơn nhé.",
                "source": "local_ai",
            }

        answer = finalize_display_answer(answer, answer_max_len)
        set_cached_answer(text, answer)
        return {"ok": True, "answer": answer, "source": "local_ai"}
    except Exception as exc:
        return {"ok": False, "answer": f"Lỗi xử lý AI local: {exc}", "source": "local_ai"}


def append_feedback(payload: dict):
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": int(time.time()),
        "question": str(payload.get("question", ""))[:2000],
        "answer": str(payload.get("answer", ""))[:5000],
        "rating": str(payload.get("rating", ""))[:16],
        "source": str(payload.get("source", "iframe"))[:64],
    }
    with FEEDBACK_LOCK:
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = build_feedback_loop_summary()
        FEEDBACK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        FEEDBACK_REPORT_PATH.write_text(
            json.dumps({
                "updated_at": int(time.time()),
                "summary": summary,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

def resolve_fpt_project_dir() -> Optional[Path]:
    candidates = [
        BASE_DIR.parent / FPT_PROJECT_NAME,
        BASE_DIR / FPT_PROJECT_NAME,
        Path.cwd() / FPT_PROJECT_NAME,
        BASE_DIR.parent.parent / FPT_PROJECT_NAME,
    ]

    seen = set()
    for candidate in candidates:
        normalized = str(candidate.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        if (candidate / "package.json").exists():
            return candidate
    return None


def build_fpt_project(project_dir: Path) -> tuple[bool, str]:
    npm_path = shutil.which("npm")
    if npm_path is None:
        return False, "Khong tim thay lenh 'npm', vui long cai NodeJS."
    try:
        subprocess.run([npm_path, "install"], cwd=project_dir, check=True)
        result = subprocess.run([npm_path, "run", "build"], cwd=project_dir, capture_output=True, text=True)
        if result.returncode != 0:
            error_output = (result.stderr or result.stdout or "").strip()
            if len(error_output) > 800:
                error_output = error_output[-800:]
            return False, f"Loi khi Build: {error_output}"
        return True, "Build FPT thanh cong."
    except Exception as exc:
        return False, f"Exception khi Build: {exc}"


def sync_build_to_embedded(source_build_dir: Path) -> tuple[bool, str]:       
    try:
        EMBEDDED_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        for item in source_build_dir.iterdir():
            target = EMBEDDED_BUILD_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        return True, "Da dong bo ban build FPT vao streamlit_assets/embedded_fpt_build."
    except Exception as exc:
        return False, f"Dong bo build vao Streamlit that bai: {exc}"

def resolve_fpt_build_dir() -> tuple[Optional[Path], str]:
    project_dir = resolve_fpt_project_dir()
    if project_dir is not None:
        source_build_dir = project_dir / "build"
        if not (source_build_dir / "index.html").exists():
            ok, message = build_fpt_project(project_dir)
            if not ok:
                return None, message
            if not (source_build_dir / "index.html").exists():
                return None, "Khong tao duoc build/index.html sau khi auto-build."    

        embedded_index = EMBEDDED_BUILD_DIR / "index.html"
        ok, message = sync_build_to_embedded(source_build_dir)

    embedded_index = EMBEDDED_BUILD_DIR / "index.html"
    if embedded_index.exists():
        return EMBEDDED_BUILD_DIR, "Dang su dung ban build FPT da duoc lap vao Streamlit."

    return None, "Khong tim thay folder FPT University Portal Redesign co package.json."

def build_inline_react_html(build_dir: Path) -> str:
    index_html_path = build_dir / "index.html"
    if not index_html_path.exists():
        return ""
    return ""

class SilentStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_POST(self):
        if self.path == "/api/warmup":
            state = ensure_models_loaded()
            self._send_json({
                "ok": bool(state.get("ready")),
                "ready": bool(state.get("ready")),
                "error": state.get("error", ""),
            })
            return

        if self.path == "/api/local-chat":
            payload = self._read_json_body()
            user_text = payload.get("message", "")
            result = run_local_chat_query(user_text)
            status = 200 if result.get("ok") else 400
            self._send_json(result, status=status)
            return

        if self.path == "/api/feedback":
            payload = self._read_json_body()
            try:
                append_feedback(payload)
                self._send_json({"ok": True})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        self._send_json({"ok": False, "error": "Not Found"}, status=404)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

def ensure_static_server(build_dir: Path) -> tuple[Optional[str], str]:       
    if STATIC_SERVER["server"] is not None and STATIC_SERVER["thread"] is not None:
        if STATIC_SERVER["thread"].is_alive() and STATIC_SERVER["port"] is not None:
            return f"http://127.0.0.1:{STATIC_SERVER['port']}/", "Static server dang chay cho FPT build."

    try:
        port = find_free_port()
        handler = partial(SilentStaticHandler, directory=str(build_dir))
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)       
        thread.start()
        STATIC_SERVER["server"] = server
        STATIC_SERVER["thread"] = thread
        STATIC_SERVER["port"] = port
        return f"http://127.0.0.1:{port}/", "Da khoi dong static server cho FPT build."
    except Exception as exc:
        return None, f"Khong khoi dong duoc static server: {exc}"

# ================= STREAMLIT MULTI-COLUMN LAYOUT =================
st.set_page_config(page_title="Nutrious Consultant", layout="wide")
start_model_preload_once()

if not os.environ.get("STREAMLIT_SERVER_FILE_WATCHER_TYPE"):
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

build_dir, build_status = resolve_fpt_build_dir()
server_url, server_status = ensure_static_server(build_dir) if build_dir else (None, "")
inline_html = build_inline_react_html(build_dir) if build_dir else ""

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {  
        height: 100%;
        overflow: hidden;
    }
    [data-testid="stAppViewContainer"] {
        background: transparent;
    }
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    #MainMenu,
    footer {
        visibility: hidden;
        height: 0;
    }
    .block-container {
        padding-top: 0;
        padding-bottom: 0;
        padding-left: 0;
        padding-right: 0;
        max-width: 100%;
    }
    [data-testid="stVerticalBlock"] {
        gap: 0;
    }
    .element-container, [data-testid="stElementContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    .fpt-fullscreen-wrap {
        position: relative;
        width: 100%;
        height: 100vh;
        overflow: hidden;
        z-index: 1;
    }
    .fpt-fullscreen-wrap iframe {
        width: 100%;
        height: 100%;
        border: 0;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if server_url:
    st.markdown(
        f"""
        <div class="fpt-fullscreen-wrap">
        <iframe src="{server_url}" title="FPT University Portal Redesign"></iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif inline_html:
    components.html(inline_html, height=900, scrolling=False)
else:
    st.error(build_status or "Khong tim thay ban build FPT.")
