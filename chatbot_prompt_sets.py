import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROMPT_FILE = BASE_DIR / "data" / "chatbot_basic_prompts_vi.json"

DEFAULT_PROMPT_SETS = {
    "basic_vi": [
        "Xin chào",
        "Mình cao 1m68 nặng 72kg, hãy gợi ý mục tiêu calo mỗi ngày để giảm mỡ an toàn.",
        "100g ức gà luộc có bao nhiêu calo và protein?",
        "Gợi ý thực đơn 1 ngày khoảng 1600 kcal, ưu tiên món dễ nấu.",
        "Nếu tối nay mình ăn phở bò thì cần điều chỉnh các bữa còn lại như thế nào?",
        "Mình nên chia protein theo từng bữa ra sao để giữ cơ khi giảm cân?",
    ],
    "fast_vi": [
        "Xin chào",
        "Mình muốn giảm cân trong 1 tháng, bắt đầu từ đâu?",
        "100g ức gà có bao nhiêu protein và calo?",
        "Gợi ý thực đơn giảm mỡ trong 1 ngày cho người 75kg.",
        "Mình cao 1m6 nặng 60kg, nên ăn bao nhiêu kcal mỗi ngày để giảm mỡ?",
    ],
    "focus_calorie": [
        "Tư vấn calo ngày hôm nay cho tôi",
        "Hôm nay tôi nên ăn bao nhiêu calo để giảm mỡ?",
        "Mức calo hôm nay cho người mới giảm cân là bao nhiêu?",
        "Calo mục tiêu hôm nay của tôi là bao nhiêu?",
        "Hôm nay nên nạp bao nhiêu kcal thì hợp lý?",
        "Tính calo hôm nay giúp tôi: nam, 28 tuổi, 72kg, cao 1m72, vận động nhẹ.",
    ],
}


def _normalize_prompt_list(items):
    cleaned = []
    seen = set()
    for item in items or []:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def load_prompt_set(name: str) -> list[str]:
    fallback = _normalize_prompt_list(DEFAULT_PROMPT_SETS.get(name, []))

    if not PROMPT_FILE.exists():
        return fallback

    try:
        payload = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return fallback

    from_file = payload.get(name) if isinstance(payload, dict) else None
    loaded = _normalize_prompt_list(from_file)
    return loaded if loaded else fallback
