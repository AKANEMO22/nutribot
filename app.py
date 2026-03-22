from pathlib import Path
import re

import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parent
FPT_BUILD_DIR = BASE_DIR.parent / "FPT University Portal Redesign" / "build"


def build_inline_react_html() -> str:
    index_html_path = FPT_BUILD_DIR / "index.html"
    if not index_html_path.exists():
        return ""

    index_html = index_html_path.read_text(encoding="utf-8")

    css_match = re.search(r'href="(/assets/[^"]+\\.css)"', index_html)
    js_match = re.search(r'src="(/assets/[^"]+\\.js)"', index_html)
    if not css_match or not js_match:
        return ""

    css_path = FPT_BUILD_DIR / css_match.group(1).lstrip("/")
    js_path = FPT_BUILD_DIR / js_match.group(1).lstrip("/")
    if not css_path.exists() or not js_path.exists():
        return ""

    css_content = css_path.read_text(encoding="utf-8")
    js_content = js_path.read_text(encoding="utf-8")

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FPT University Portal Redesign</title>
  <style>{css_content}</style>
  <style>
    html, body, #root {{ margin: 0; min-height: 100vh; }}
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="module">{js_content}</script>
</body>
</html>
"""


st.set_page_config(page_title="Nutrious Consultant", layout="wide")
inline_html = build_inline_react_html()

st.markdown(
    """
    <style>
      header[data-testid="stHeader"],
      div[data-testid="stToolbar"],
      div[data-testid="stDecoration"],
      #MainMenu,
      footer {
        visibility: hidden;
        height: 0;
      }
      .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if inline_html:
    st.info("Dang nhung truc tiep ban build React tu FPT University Portal Redesign vao Streamlit.")
    components.html(inline_html, height=1200, scrolling=True)
else:
    st.error("Khong tim thay ban build FPT. Hay chay npm run build trong folder FPT University Portal Redesign.")
