from pathlib import Path
import re
import os

import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parent


def resolve_build_dir() -> Path | None:
    """Find a React build directory from common locations."""
    env_build_dir = os.getenv("FPT_BUILD_DIR")
    candidates = []
    if env_build_dir:
        candidates.append(Path(env_build_dir))

    candidates.extend(
        [
            BASE_DIR / "build",
            BASE_DIR / "frontend" / "build",
            BASE_DIR.parent / "FPT University Portal Redesign" / "build",
        ]
    )

    for candidate in candidates:
        if (candidate / "index.html").exists():
            return candidate
    return None


def build_inline_react_html(build_dir: Path) -> str:
    index_html_path = build_dir / "index.html"
    if not index_html_path.exists():
        return ""

    index_html = index_html_path.read_text(encoding="utf-8")

    css_match = re.search(r'href="(/assets/[^"]+\\.css)"', index_html)
    js_match = re.search(r'src="(/assets/[^"]+\\.js)"', index_html)
    if not css_match or not js_match:
        return ""

    css_path = build_dir / css_match.group(1).lstrip("/")
    js_path = build_dir / js_match.group(1).lstrip("/")
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
build_dir = resolve_build_dir()
inline_html = build_inline_react_html(build_dir) if build_dir else ""

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
    st.info(f"Dang nhung truc tiep ban build React tu: {build_dir}")
    components.html(inline_html, height=1200, scrolling=True)
else:
    st.warning("Chua tim thay ban build React. App van chay, nhung se hien che do huong dan.")
    st.markdown(
        """
### Huong dan khoi phuc giao dien FPT
1. Mo project frontend React cua ban.
2. Chay lenh `npm run build`.
3. Dat thu muc `build` vao mot trong cac vi tri sau:
   - `nutribot/build`
   - `nutribot/frontend/build`
   - `../FPT University Portal Redesign/build`
4. Hoac dat bien moi truong `FPT_BUILD_DIR` tro den thu muc build.
        """
    )
