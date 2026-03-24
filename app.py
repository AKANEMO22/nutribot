import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import json
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


def should_skip_safety_filter(text: str) -> bool:
    normalized = normalize_food_name(text)
    if not normalized:
        return True

    # Avoid false-positive blocks for simple greetings/short benign messages.
    if normalized in {"hi", "hello", "hey", "xin chao", "chao"}:
        return True

    if len(normalized) <= 3 and normalized.isalpha():
        return True

    return False


def is_short_greeting(text: str) -> bool:
    normalized = normalize_food_name(text)
    return normalized in {"hi", "hello", "hey", "xin chao", "chao"}


def has_nutrition_intent(text: str) -> bool:
    normalized = normalize_food_name(text)
    return any(
        token in normalized
        for token in ("calo", "calories", "protein", "carb", "fat", "beo", "thuc don", "mon an", "giam can", "tang can")
    )


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

    has_nutrition_intent = any(
        token in normalized_q
        for token in ("calo", "calories", "protein", "carb", "fat", "beo", "thuc don", "mon an", "giam can", "tang can")
    )

    if not matched and has_nutrition_intent:
        matched = list(nutrition_db.values())[:limit]

    if not matched:
        return ""

    lines = ["Du lieu chi so dinh duong noi bo (tu module nhap mon an):"]
    for item in matched[:limit]:
        lines.append(
            f"- {item['name']}: {item['calories']:.0f} kcal, protein {item['protein']:.1f}g, carbs {item['carbs']:.1f}g, fat {item['fat']:.1f}g"
        )
    return "\n".join(lines)


def load_rag_and_models():
    q_filter_instance = None
    model_path = BASE_DIR / "weight" / "question_filter_model.pkl"
    if model_path.exists():
        q_filter_instance = QuestionFilter(model_path=str(model_path))

    if check_ollama():
        vs = build_vectorstore(force_rebuild=False)
        chain = build_rag_chain(vs)
        return q_filter_instance, chain, True, ""

    if CONFIG.get("llm_backend") == "ollama":
        return q_filter_instance, None, False, "Ollama chưa chạy hoặc chưa sẵn sàng."

    return (
        q_filter_instance,
        None,
        False,
        "Thiếu local weights hoặc model local chưa sẵn sàng.",
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


def run_local_chat_query(user_text: str) -> dict:
    text = (user_text or "").strip()
    if not text:
        return {"ok": False, "answer": "Vui lòng nhập câu hỏi."}

    # Keep prompt compact for small local models.
    text = text[:320]

    state = ensure_models_loaded()
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
            if q_filter_instance.is_dangerous(text):
                return {
                    "ok": False,
                    "blocked": True,
                    "answer": "⚠️ Câu hỏi bị chặn bởi bộ lọc an toàn.",
                    "source": "local_ai",
                }
        except Exception:
            pass

    try:
        greeting_mode = is_short_greeting(text)
        use_retrieval = has_nutrition_intent(text)
        nutrition_context = build_nutrition_context(text, limit=4) if use_retrieval else ""

        if use_retrieval and nutrition_context:
            final_question = (
                f"{text}\n\n"
                f"{nutrition_context}\n"
                "Khi tra loi, uu tien su dung du lieu noi bo o tren neu lien quan."
            )
        elif use_retrieval:
            final_question = text
        else:
            final_question = (
                f"Nguoi dung vua nhan: {text}. "
                "Hay tra loi ngắn gon, tu nhien bang tieng Viet, khong trich dan URL, khong dua tai lieu hoc thuat."
            )

        final_question = final_question[:1200]

        def call_chain(payload_question: str, skip_retrieval: bool):
            return chain({"question": payload_question, "skip_retrieval": skip_retrieval})

        try:
            result = call_chain(final_question, (not use_retrieval))
            answer = str(result.get("answer", "")).strip()
        except Exception as first_exc:
            msg = str(first_exc).lower()
            if ("max_length" in msg) or ("input length" in msg) or ("indexing errors" in msg):
                compact_question = (
                    f"Tra loi bang tieng Viet, toi da 2 cau, khong URL. Cau hoi: {text}"
                )[:260]
                result = call_chain(compact_question, True)
                answer = str(result.get("answer", "")).strip()
            else:
                raise

        if not answer or looks_low_quality_answer(text, answer):
            feedback_summary = build_feedback_loop_summary()
            strict_tone = ""
            if feedback_summary.get("greeting_down_rate", 0.0) >= 0.3:
                strict_tone = " Tra loi rat ngan gon (toi da 2 cau)."

            retry_question = (
                f"Người dùng hỏi: {text}. "
                "Hãy trả lời ngắn gọn, rõ ràng bằng tiếng Việt trong 1-3 câu."
                " Không trích URL, không đưa tài liệu học thuật không liên quan."
                f"{strict_tone}"
            )
            retry_result = chain({"question": retry_question[:600], "skip_retrieval": True})
            answer = str(retry_result.get("answer", "")).strip()

        if greeting_mode and (not answer or looks_low_quality_answer(text, answer) or len(answer) > 180):
            greeting_retry = (
                "Người dùng vừa chào. "
                "Hãy chào lại bằng tiếng Việt có dấu trong 1 câu dưới 20 từ, thân thiện, không ký tự lạ."
            )
            gr = chain({"question": greeting_retry, "skip_retrieval": True})
            answer = str(gr.get("answer", "")).strip()

        if looks_unaccented_vietnamese(answer):
            rewrite_q = (
                "Hãy viết lại câu sau bằng tiếng Việt có dấu, giữ nguyên nghĩa, ngắn gọn và tự nhiên:\n"
                f"{answer[:300]}"
            )
            rewritten = chain({"question": rewrite_q, "skip_retrieval": True})
            rewritten_answer = str(rewritten.get("answer", "")).strip()
            if rewritten_answer:
                answer = rewritten_answer

        if greeting_mode and (not answer or looks_low_quality_answer(text, answer) or len(answer) > 180):
            answer = "Chào bạn! Mình là NutriBot, bạn muốn mình hỗ trợ calories hay thực đơn hôm nay?"

        if not answer:
            return {
                "ok": False,
                "answer": "AI local da xu ly nhung chua tao duoc cau tra loi ro rang. Vui long thu lai voi cau hoi cu the hon.",
                "source": "local_ai",
            }

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
