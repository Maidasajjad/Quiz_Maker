# 📝 Quiz Maker from PDF

A Streamlit app that turns your study notes into an instant quiz using Google Gemini.

## Features
- Upload a **PDF, DOCX or TXT** file (up to 10 MB)
- Choose a **page range** so you can quiz yourself on one chapter
- Choose **3-20 questions**, **difficulty** (Easy / Medium / Hard) and **type** (Multiple choice, True/False, Mixed)
- Take the quiz in the app and get an **instant score**
- See the **correct answer and explanation** for every question
- Get a list of **topics to revise** based on your wrong answers
- Download the quiz with or without the answer key
- Answer options are shuffled in code so the correct answer is not always in the same position

## How it works
1. The app extracts text from your file (`pypdf` for PDF, `python-docx` for Word).
2. Your selected pages and settings are sent to the Gemini model as a prompt.
3. Gemini returns the quiz as JSON.
4. The app validates the JSON, removes broken or duplicate questions, and shuffles the options.
5. You answer in the app and the score is calculated in code.

## Run locally

```bash
git clone https://github.com/<your-username>/quiz-maker.git
cd quiz-maker

python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

pip install -r requirements.txt
```

Get a free API key at <https://aistudio.google.com/apikey>, then provide it **one** of these ways:

1. Paste it into the sidebar when the app runs, or
2. Set an environment variable:
   - macOS/Linux: `export GEMINI_API_KEY="your_key"`
   - Windows PowerShell: `$env:GEMINI_API_KEY="your_key"`
3. Create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your_key"
   ```

Start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this project to a GitHub repository.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** and select your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_key"
   ```
5. Click **Deploy**.

**Never commit your API key.** Keep `.streamlit/secrets.toml` and `.env` in `.gitignore`.

## Configuration
| Setting | How | Default |
|---|---|---|
| API key | sidebar / `GEMINI_API_KEY` env var / Streamlit secret | none |
| Model | sidebar field or `GEMINI_MODEL` env var | `gemini-3.5-flash` (falls back to `gemini-2.5-flash` if unavailable) |

## Project structure
```
app.py             # Streamlit UI + quiz logic
requirements.txt   # Python dependencies (file name must be lowercase)
README.md
```

## Limitations
- Scanned/image-only PDFs have no text to read. Use a text-based PDF.
- Questions are AI-generated, so check important answers against your notes.

## Privacy
Document text is sent to the Gemini API to write the quiz. The app does not store uploaded files.
