"""Launch the optional sampled-cohort Streamlit user evaluation app."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_streamlit_script_run() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


def main() -> None:
    # PyArrow 25 uses mimalloc by default in this environment. Streamlit executes
    # app reruns in worker threads, where that allocator has caused native crashes.
    os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
    if not _is_streamlit_script_run():
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).resolve()),
                "--",
                *sys.argv[1:],
            ],
        )

    from embedding_clustering_experiment.user_evaluation import main as run_app

    run_app()


if __name__ == "__main__":
    main()
