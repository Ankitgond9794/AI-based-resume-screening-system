# AI-based-resume-screening-system
An AI-powered Resume Screening System that automatically classifies resumes into different job categories using Machine Learning and Natural Language Processing (NLP). This project helps recruiters filter resumes quickly and accurately.
---

## Features

- Upload resume files
- Resume text preprocessing using NLP
- Text vectorization using TF-IDF
- Multiple Machine Learning models
- Predicts resume category
- Interactive web interface with Streamlit
- Fast and accurate resume classification
---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Joblib
---

## Machine Learning Models

The project compares multiple classification algorithms:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
---

## Project Structure
```
Resume-Screening-System/
│
├── app.py
├── resume screening system.ipynb
├── requirements.txt
├── README.md
├── tfidf.pkl
├── model.pkl
├── label_encoder.pkl
├── dataset/
│   └── Resume.csv
└── images/
```
---

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Resume-Screening-System.git
```

```bash
cd Resume-Screening-System
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```
---

## Dataset

The dataset contains resumes from multiple job categories such as:

- Data Science
- HR
- Java Developer
- Python Developer
- Testing
- DevOps
- Web Designing
- Business Analyst
- Sales
- Mechanical Engineer
- Civil Engineer
- Network Security
- ETL Developer
- SAP Developer
- Blockchain
- and more...
---

## Workflow

1. Load dataset
2. Clean resume text
3. Remove stopwords and punctuation
4. TF-IDF Vectorization
5. Train Machine Learning models
6. Save trained model
7. Predict resume category
8. Display prediction using Streamlit
---

## Future Improvements

- Deep Learning (LSTM/BERT)
- Resume ranking system
- PDF & DOCX parsing
- Skill extraction
- ATS score calculation
- Job recommendation system
---

## Screenshots

Add screenshots of your Streamlit application here.
---

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.
---

## Author
**Ankit Kumar**
B.Tech CSE (AI)

---
