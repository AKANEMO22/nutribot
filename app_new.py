from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
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

# Khởi tạo RAG và Model ở Cache để tránh load lại lúc re-render UI
@st.cache_resource
def load_rag_and_models():
    # 1. Khởi tạo Q_Filter
    q_filter_instance = None
    if Path("weight/question_filter_model.pkl").exists():
        q_filter_instance = QuestionFilter(model_path="weight/question_filter_model.pkl")
    
    # 2. Khởi tạo RAG Chain
    if check_ollama():
        vs = build_vectorstore(force_rebuild=False)
        chain = build_rag_chain(vs)
        return q_filter_instance, chain, True
    else:
        return q_filter_instance, None, False

q_filter, rag_chain, ollama_ok = load_rag_and_models()

# Chia UI làm 2 cột: 75% cho Web Portal, 25% cho Chatbot
col1, col2 = st.columns([3, 1])

with col1:
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

with col2:
    st.markdown("<h3 style='text-align: center;'>🤖 NutriBot Chat</h3>", unsafe_allow_html=True)
    
    if not ollama_ok:
        if CONFIG.get("llm_backend") == "ollama":
            st.error("Ollama chưa chạy. Vui lòng bật Ollama trên máy tính của bạn!")
        else:
            st.error("Chưa có local weights. Chạy: python script_download/download_local_weights.py")
    else:
        # Khởi tạo session state lưu trữ chat
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Hiển thị tin nhắn cũ
        chat_container = st.container(height=600)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Hãy hỏi NutriBot..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message(role="user"):
                    st.markdown(prompt)

            # --- QUESTION FILTER LAYER ---
            is_blocked = False
            if q_filter:
                try:
                    if q_filter.is_dangerous(prompt):
                        is_blocked = True
                except Exception as e:
                    pass

            if is_blocked:
                # Nếu câu hỏi bị chặn bởi mô hình Filter
                response_text = "⚠️ **[CẢNH BÁO]** Phát hiện nội dung nhạy cảm, độc hại. Hệ thống từ chối trả lời để đảm bảo an toàn."
                with chat_container:
                    with st.chat_message(role="assistant"):
                        st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                # --- RAG CHAIN EXECUTION ---
                with chat_container:
                    with st.chat_message(role="assistant"):
                        with st.spinner("Đang suy nghĩ..."):
                            try:
                                result = rag_chain({"question": prompt})
                                answer = result.get("answer", "Xin lỗi, tôi không thể trả lời câu hỏi này lúc này.")
                            except Exception as e:
                                answer = f"Đã có lỗi xảy ra: {e}"
                        st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
