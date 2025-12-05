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
5️⃣ Random forestclassfire Model
6️⃣ Serialize model with joblib
7️⃣ Deploy with Streamlit
```

## 📦 Installation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## 🛠 Requirements
streamlit
scikit-learn
pandas
numpy
joblib

## 🌐 Run the Streamlit App
cd app
streamlit run app.py

## 🧪 Example Prediction Output
Prediction: ⚠️ Fake / Suspicious
Probability: 0.93
Suspicious Keywords: fee, registration, pay

## 📊 Screenshots
<img width="953" height="440" alt="Screenshot 2025-12-05 141319" src="https://github.com/user-attachments/assets/6491fd38-8910-4904-a69a-1ade24da2f71" />
<img width="955" height="444" alt="Screenshot 2025-12-05 141625" src="https://github.com/user-attachments/assets/3f1350da-4ee3-4208-b0dc-9cd2eab09c0d" />

## 👨‍💻 Author
Vishwa
Fake Job Scam Detector Project (Machine Learning + NLP + Streamlit)
