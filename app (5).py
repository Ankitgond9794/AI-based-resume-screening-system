"""
Resume Screening System - Streamlit app.

Two tools:
1. Decision Predictor - paste a candidate's skills, get a predicted
   Recruiter Decision from the trained SVM (model.pkl + tfidf.pkl +
   label_encoder.pkl).
2. JD Matching & Ranking - paste a job description and one or more
   resumes, get a ranked cosine-similarity match table with matched
   keywords shown per resume.
"""

import os

import joblib
import pandas as pd
import streamlit as st

from utils import clean_text, rank_resumes, matched_keywords

MODELS_DIR = "models"
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

st.set_page_config(
    page_title="Resume Screening System",
    page_icon="🧾",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained TF-IDF vectorizer, classifier, and label encoder."""
    missing = [p for p in (TFIDF_PATH, MODEL_PATH, ENCODER_PATH) if not os.path.exists(p)]
    if missing:
        return None, None, None, missing
    tfidf = joblib.load(TFIDF_PATH)
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return tfidf, model, encoder, []


def predict_decision(skills_text: str, tfidf, model, encoder):
    cleaned = clean_text(skills_text)
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)[0]
    label = encoder.inverse_transform([pred])[0]

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba_values = model.predict_proba(vector)[0]
            proba = dict(zip(encoder.classes_, proba_values))
        except Exception:
            proba = None
    return label, proba


def render_predictor_tab():
    st.subheader("Predict a Recruiter Decision from Skills")
    st.caption(
        "Paste a candidate's skills (comma or free text) and get the model's "
        "predicted decision — trained on skills text mapped to recruiter outcomes."
    )

    tfidf, model, encoder, missing = load_artifacts()
    if missing:
        st.warning(
            "Model files not found: "
            + ", ".join(missing)
            + ". Add `tfidf.pkl`, `model.pkl`, and `label_encoder.pkl` to the "
            "`models/` folder (download them from your Colab notebook, or run "
            "`train_model.py`)."
        )
        return

    skills_text = st.text_area(
        "Candidate skills",
        placeholder="e.g. Python, SQL, machine learning, scikit-learn, communication",
        height=120,
    )

    if st.button("Predict decision", type="primary", disabled=not skills_text.strip()):
        with st.spinner("Scoring..."):
            label, proba = predict_decision(skills_text, tfidf, model, encoder)

        st.success(f"Predicted decision: **{label}**")

        if proba:
            proba_df = pd.DataFrame(
                {"decision": list(proba.keys()), "confidence": list(proba.values())}
            ).sort_values("confidence", ascending=False)
            proba_df["confidence"] = (proba_df["confidence"] * 100).round(1)
            st.bar_chart(proba_df.set_index("decision"))


def render_matching_tab():
    st.subheader("Match & Rank Resumes Against a Job Description")
    st.caption(
        "Paste a job description and one or more resumes to get a ranked "
        "match score (TF-IDF cosine similarity, negation-aware — 'no SQL "
        "experience' won't score as a SQL match)."
    )

    job_description = st.text_area(
        "Job description",
        placeholder="Paste the job description here...",
        height=160,
    )

    st.markdown("**Resumes**")
    input_mode = st.radio(
        "How do you want to add resumes?",
        ["Paste text", "Upload .txt files"],
        horizontal=True,
        label_visibility="collapsed",
    )

    resumes = {}

    if input_mode == "Paste text":
        num_resumes = st.number_input(
            "Number of resumes", min_value=1, max_value=20, value=2, step=1
        )
        for i in range(int(num_resumes)):
            cols = st.columns([1, 3])
            with cols[0]:
                name = st.text_input(
                    f"Resume {i + 1} name", value=f"Resume_{i + 1}", key=f"name_{i}"
                )
            with cols[1]:
                text = st.text_area(
                    f"Resume {i + 1} text", key=f"text_{i}", height=100
                )
            if text.strip():
                resumes[name or f"Resume_{i + 1}"] = text
    else:
        uploaded_files = st.file_uploader(
            "Upload resume .txt files", type=["txt"], accept_multiple_files=True
        )
        for f in uploaded_files or []:
            resumes[f.name] = f.read().decode("utf-8", errors="ignore")

    run = st.button(
        "Rank resumes",
        type="primary",
        disabled=not (job_description.strip() and resumes),
    )

    if run:
        with st.spinner("Ranking resumes..."):
            results, clean_jd, clean_resumes = rank_resumes(job_description, resumes)

        st.markdown("### Ranking")
        st.dataframe(
            results.rename(columns={"resume": "Resume", "match_percent": "Match %"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Why each resume matched")
        for _, row in results.iterrows():
            keywords = matched_keywords(clean_jd, clean_resumes[row["resume"]])
            with st.expander(f"{row['resume']} — {row['match_percent']}%"):
                if keywords:
                    st.write(", ".join(keywords))
                else:
                    st.write("No overlapping keywords found.")


def main():
    st.title("🧾 Resume Screening System")
    st.write(
        "A TF-IDF + SVM resume screening toolkit: predict a recruiter decision "
        "from a candidate's skills, or rank multiple resumes against a job "
        "description."
    )

    tab1, tab2 = st.tabs(["Decision Predictor", "JD Matching & Ranking"])
    with tab1:
        render_predictor_tab()
    with tab2:
        render_matching_tab()


if __name__ == "__main__":
    main()
