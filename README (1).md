# Resume Screening System

A Streamlit app built from the resume-screening notebook. Three tools in one app:

1. **Decision Predictor** — paste a candidate's skills, or upload a resume
   (`.txt`, `.pdf`, `.docx`, `.jpg`, `.jpeg`, `.png`), and get a predicted Recruiter Decision
   (TF-IDF + SVM, trained on skills text).
2. **JD Matching & Ranking** — paste a job description and add multiple resumes (pasted
   text or uploaded `.txt`/`.pdf`/`.docx`/image files), get a ranked cosine-similarity match
   table with matched keywords per resume (negation-aware: "no SQL experience" won't count
   as an SQL match).
3. **ATS Score Checker** — paste a job description and one resume, get a 0-100 ATS
   compatibility score broken down into keyword match, contact info, standard sections
   (experience/education/skills), formatting red flags (scanned PDFs, tables), and resume
   length — plus concrete suggestions to improve the score.

Image and scanned-PDF resumes are read with OCR (Tesseract). Text-based PDFs and Word
documents are parsed directly, no OCR needed.

## Project structure

```
resume-screening-app/
├── app.py                     # Streamlit app (entry point)
├── utils.py                   # shared text cleaning + matching functions
├── train_model.py             # reproducible training script
├── requirements.txt
├── packages.txt                # system packages (Tesseract OCR, Poppler) for Streamlit Cloud
├── .gitignore
├── .streamlit/
│   └── config.toml            # app theme
├── models/                    # put tfidf.pkl, model.pkl, label_encoder.pkl here
└── sample_data/
    └── sample_resumes.py      # sample JD + resumes for quick testing
```

## 1. Get the model files

The Decision Predictor tab needs three files in `models/`:

- `tfidf.pkl`
- `model.pkl`
- `label_encoder.pkl`

You already generated these in your Colab notebook (cells 90–93) and downloaded them —
just drop them into `models/`.

**Or regenerate them** from your dataset:

```bash
pip install -r requirements.txt
python train_model.py --csv path/to/AI_Resume_Screening.csv
```

This saves fresh `tfidf.pkl`, `model.pkl`, and `label_encoder.pkl` into `models/`.

> The JD Matching & Ranking tab does **not** need these files — it builds a fresh TF-IDF
> comparison at request time, so it works even before you add a trained model.

## 2. Run locally

Image and scanned-PDF resume support needs two system packages (Tesseract OCR and Poppler,
for PDF-to-image conversion). Text PDFs, DOCX, and TXT work without them.

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# Windows: install Tesseract from
# https://github.com/UB-Mannheim/tesseract/wiki, and Poppler from
# https://github.com/oschwartz10612/poppler-windows/releases
```

Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 3. Push to GitHub

```bash
git init
git add .
git commit -m "Resume screening Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If your `model.pkl` / `tfidf.pkl` are large (a few MB is usually fine on GitHub;
tens of MB+ needs [Git LFS](https://git-lfs.com)), check the file sizes before pushing:

```bash
du -h models/*.pkl
```

## 4. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**, pick your repo/branch, and set the main file to `app.py`.
3. Deploy. Streamlit Cloud automatically installs the system packages listed in
   `packages.txt` (Tesseract, Poppler) alongside `requirements.txt` — no extra setup needed.
4. First run may take a minute while NLTK downloads `stopwords`/`wordnet`/`punkt`
   (handled automatically by `utils.ensure_nltk_data()`).

## Notes

- `clean_text()` in `utils.py` matches the cleaning used for the classifier (matches
  notebook cells 71–72).
- `clean_text_negation()` and `rank_resumes()` power the JD matching tab (matches
  notebook cells 83–87).
- `train_model.py` expects a CSV with `Skills` and `Recruiter Decision` columns,
  matching the `AI_Resume_Screening.csv` dataset from the notebook.
