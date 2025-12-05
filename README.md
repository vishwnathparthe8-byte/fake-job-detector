##  🕵️‍♂️ Fake Job Detector — ML + NLP + Streamlit
Detect Scam / Fraud Job Postings Automatically

This project uses NLP + Machine Learning to classify job postings as
Real (0) or Fake/Suspicious (1).


## 🚀 Features

✅ Cleans & preprocesses raw job text

✅ Extracts suspicious patterns using NLP

✅ Trains ML model (TF-IDF + Logistic Regression)

✅ Predicts whether a job is real or fake

✅ Streamlit UI for easy testing

✅ Highlight suspicious keywords

✅ Explainable output (top token contributions)


## 🗂 Folder Structure
```
fake-job-detector/
│
├── data/
│   └── jobs_raw.csv
│
├── model/
│   └── fake_job_pipeline.pkl
│
├── app/
│   └── app.py
│
├── notebook/
│   └── fake_job_notebook.ipynb
│
│
├── requirements.txt
└── README.md
```

## 🧠 ML Pipeline
```
1️⃣ Load Data
2️⃣ Preprocess (NLP rules)
3️⃣ Weak labeling (rule-based)
4️⃣ TF-IDF Vectorizer
5️⃣ Logistic Regression Model
6️⃣ Serialize model with joblib
7️⃣ Deploy with Streamlit
```

