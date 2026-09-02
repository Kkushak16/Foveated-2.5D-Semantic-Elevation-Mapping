"""
app.py — Unified Dual-Sensor Foveated 2.5D Perception Web Dashboard
===================================================================
Master entrypoint for the unified web dashboard.

Usage:
    py app.py               -> Launches local server & opens Web Dashboard at http://localhost:8080
    streamlit run app.py    -> Launches Streamlit App embedding the Unified Web HUD
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Check if running under Streamlit
try:
    import streamlit as st
    is_streamlit = True
except ImportError:
    is_streamlit = False

def run_standalone():
    import run_dashboard
    run_dashboard.main()

if __name__ == "__main__":
    # Check if executed directly with python (e.g. `py app.py`) vs `streamlit run app.py`
    if not is_streamlit or "streamlit" not in sys.modules or not sys.argv[0].endswith("streamlit"):
        run_standalone()
    else:
        st.set_page_config(
            page_title="Unified Foveated 2.5D Perception HUD",
            page_icon="🚗",
            layout="wide"
        )
        
        # Hide default padding
        st.markdown("""
        <style>
            .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 1rem; padding-right: 1rem; }
            header { visibility: hidden; }
        </style>
        """, unsafe_allow_html=True)
        
        # Serve index.html content directly in Streamlit
        html_path = os.path.join(SCRIPT_DIR, "web", "ui", "index.html")
        styles_path = os.path.join(SCRIPT_DIR, "web", "ui", "styles.css")
        js_path = os.path.join(SCRIPT_DIR, "web", "ui", "teleop_dashboard.js")

        if os.path.exists(html_path) and os.path.exists(styles_path) and os.path.exists(js_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            with open(styles_path, "r", encoding="utf-8") as f:
                css_content = f.read()
            with open(js_path, "r", encoding="utf-8") as f:
                js_content = f.read()

            combined_html = f"""
            <style>
            {css_content}
            body {{ height: 95vh !important; }}
            </style>
            {html_content}
            <script>
            {js_content}
            </script>
            """
            st.components.v1.html(combined_html, height=850, scrolling=False)
        else:
            st.error("Dashboard assets not found in web/ui/")
