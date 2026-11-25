# Browser version of the app (GitHub Pages)

This branch holds an always-on copy of the DPU Filing Assistant that runs
entirely in the visitor's browser with [stlite](https://github.com/whitphx/stlite).
There is no server, so it never goes to sleep, and uploaded filings never leave
the visitor's device. The Streamlit Community Cloud app keeps running from `main`.

It runs the app's **No AI** mode: search, key figures from tables and text
patterns, the arithmetic checks, template information requests and filing
comparisons. A public web page can't keep an API key secret, and the AI
providers' Python libraries can't make network calls from inside a browser, so
the AI models are left to the Streamlit app.

- `index.html` loads `app.py`, `rag/` and `samples/` from `main` each time it
  opens, so changes pushed there appear here automatically.
- `static_entry.py` empties the AI model menu, then runs `app.py` unchanged.
- If the app gains a new `rag/` module or sample file, add it to `FROM_MAIN`
  in `index.html` (and `404.html`).
