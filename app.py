import streamlit as st
import joblib
import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Load saved files
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
encoder = joblib.load("label_encoder.pkl")

# NLP setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# --Streamlit UI--

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="",
    layout="centered"
)

st.title("AI-Based Resume Screening System")
st.write("Upload or paste resume text to predict the job category.")

resume = st.text_area(
    "Paste Resume Text",
    height=300,
    placeholder="Paste the complete resume here..."
)

if st.button("Predict Category"):

    if resume.strip() == "":
        st.warning("Please enter resume text.")
    else:

        clean_resume = clean_text(resume)

        vector = tfidf.transform([clean_resume])

        prediction = model.predict(vector)

        category = encoder.inverse_transform(prediction)

        st.success(f"Predicted Category: **{category[0]}**")
