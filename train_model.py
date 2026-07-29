"""
Trains the resume Skills -> Recruiter Decision classifier and saves the
artifacts the Streamlit app expects in models/.

This mirrors the "for df2" section of the original Colab notebook
(TF-IDF on cleaned Skills text -> SVM -> Recruiter Decision), cleaned up
into a runnable script.

Usage:
    python train_model.py --csv path/to/AI_Resume_Screening.csv

Expects the CSV to have at least these columns:
    - "Skills"
    - "Recruiter Decision"

Saves:
    models/tfidf.pkl
    models/model.pkl
    models/label_encoder.pkl
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from utils import clean_text


def main(csv_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()

    if "Skills" not in df.columns or "Recruiter Decision" not in df.columns:
        raise ValueError(
            "CSV must contain 'Skills' and 'Recruiter Decision' columns. "
            f"Found: {list(df.columns)}"
        )

    print("Cleaning skills text...")
    df["Clean_Skills"] = df["Skills"].apply(clean_text)

    print("Vectorizing with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df["Clean_Skills"])

    print("Encoding labels...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Recruiter Decision"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("Training SVM...")
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)

    predictions = svm.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, target_names=encoder.classes_))

    tfidf_path = os.path.join(out_dir, "tfidf.pkl")
    model_path = os.path.join(out_dir, "model.pkl")
    encoder_path = os.path.join(out_dir, "label_encoder.pkl")

    joblib.dump(tfidf, tfidf_path)
    joblib.dump(svm, model_path)
    joblib.dump(encoder, encoder_path)

    print(f"\nSaved: {tfidf_path}")
    print(f"Saved: {model_path}")
    print(f"Saved: {encoder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the resume screening model.")
    parser.add_argument(
        "--csv", required=True, help="Path to the resume screening CSV file."
    )
    parser.add_argument(
        "--out_dir",
        default="models",
        help="Directory to save trained model artifacts (default: models/).",
    )
    args = parser.parse_args()
    main(args.csv, args.out_dir)
