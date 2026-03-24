"""Compatibility entrypoint.

This file is intentionally kept so existing commands such as:
    streamlit run app_new.py
continue to work after merging functionality into app.py.
"""

from app import *  # noqa: F401,F403
