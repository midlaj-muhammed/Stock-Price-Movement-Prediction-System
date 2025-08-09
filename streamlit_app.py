"""
Root entrypoint for Streamlit deployments.

This shim ensures hosting platforms (e.g., Streamlit Community Cloud)
can discover the app without custom main file configuration.

It imports and runs the actual app located at `src/web/app.py`.
"""

from src.web.app import main as run_app


if __name__ == "__main__":
    run_app()


