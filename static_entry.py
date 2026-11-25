"""Browser (stlite) entry point for the DPU Filing Assistant.

A web page can't keep an API key secret, and the AI providers' Python libraries
can't make network calls from inside a browser. So this always-on copy runs the
app's own "No AI" mode: search, key figures from tables and text patterns, the
arithmetic checks, template information requests and filing comparisons.

It removes the AI models from the model menu, then runs app.py unchanged.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import llm  # noqa: E402

llm.MODELS.clear()          # the menu then offers only "No AI · search and checks only"

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
