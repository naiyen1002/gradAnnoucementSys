# Notebook → Streamlit App

This repo contains a Streamlit app auto-generated from a Jupyter notebook.

## Files
- `app.py` — Streamlit entry point
- `IP-checkpoint.ipynb` — original notebook (for reference)
- `requirements.txt` — Python dependencies
- `.gitignore` — ignores common Python/build files

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io (Streamlit Community Cloud) and **New app**.
3. Select your repo, branch, and set **Main file path** to `app.py`.
4. Click **Deploy**. The app will build using `requirements.txt`.

### Notes
- If your notebook relied on IPython magics (e.g., `%pip`, `%matplotlib`, `!wget`), they were commented out. Replace with standard Python equivalents if needed.
- Matplotlib `plt.show()` is patched to render inside Streamlit.
- Use the file uploader in the app to provide any data files at runtime.
