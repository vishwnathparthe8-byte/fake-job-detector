# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from io import StringIO

st.set_page_config(page_title="Fake Job Detector 🔎", layout="wide",
                   page_icon="🕵️‍♂️")

# ---------- Helper utilities ----------
def load_pipeline(path="fake_job_pipeline.pkl"):
    try:
        pipe = joblib.load(path)
        return pipe
    except Exception as e:
        st.error(f"Could not load model from {path}: {e}")
        return None

SUSPICIOUS_KEYWORDS = [
    "fee","registration","activation","pay","invest","investment","earn",
    "weekly","daily","per day","per week","buy software","software fee",
    "verification fee","activation amount","pay 999","pay 499","pay 799",
    "pay 1299","pay 1500"
]

def find_keyword_matches(text):
    t = str(text).lower()
    found = []
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in t:
            found.append(kw)
    return list(dict.fromkeys(found))  # unique order-preserved

def highlight_text(text, keywords):
    # simple HTML highlighting: red for suspicious words
    safe_html = st.components.v1
    esc = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )
    # highlight longer keywords first to avoid partial overlap
    kws = sorted(keywords, key=lambda x: -len(x))
    for kw in kws:
        esc = re.sub(r"(?i)("+re.escape(kw)+r")", r'<mark style="background:#ffb3b3;color:#000">\1</mark>', esc)
    # return raw HTML
    return esc

def top_contributors(pipe, text, topn=8):
    """
    Compute token contributions using TF-IDF * coef for LogisticRegression.
    Returns list of (token, contribution) sorted descending.
    """
    try:
        vectorizer = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["clf"]
    except Exception:
        # pipeline not named that way
        return []

    X = vectorizer.transform([text])
    if hasattr(clf, "coef_"):
        # coef_[0] shape (n_features,)
        coef = clf.coef_[0]
        # contributions: feature_value * coef
        # X is sparse
        vals = X.toarray()[0] * coef
        # get top indices
        if len(vals) == 0:
            return []
        idx = np.argsort(vals)[-topn:][::-1]
        feat = vectorizer.get_feature_names_out()
        result = [(feat[i], float(vals[i])) for i in idx if vals[i] != 0.0]
        return result
    else:
        return []

# ---------- Load model ----------
with st.spinner("Loading model..."):
    PIPELINE = load_pipeline("fake_job_pipeline.pkl")

# ---------- UI ----------
st.markdown("<div style='display:flex;align-items:center;gap:12px'>"
            "<h1 style='margin:0'>🕵️‍♂️ Fake Job Detector</h1>"
            "<div style='color:gray;margin-left:8px'>A demo to flag suspicious job posts</div>"
            "</div>",
            unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Try samples")
    sample = st.selectbox("Choose a sample job post", [
        "— choose sample —",
        "Real: Python developer at NextGen",
        "Fake: Work from home, earn ₹50,000 weekly, pay ₹999 registration",
        "Fake: Earn 1 lakh monthly. Investment required 2000",
        "Real: Frontend dev - React, apply on company site"
    ])
    st.write("Or upload a CSV with columns: title,description,salary_text")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    st.markdown("---")
    st.markdown("Model status:")
    if PIPELINE:
        st.success("Model loaded ✅")
    else:
        st.error("Model NOT loaded")

# Main input area
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Enter job post")
    title = st.text_input("Job title", value="")
    description = st.text_area("Job description", value="", height=220)
    salary_text = st.text_input("Salary text (optional)", value="")
    if st.button("Predict"):
        text_input = f"{title} {description} {salary_text}".strip()
        if len(text_input) == 0:
            st.warning("Please enter job title and/or description.")
        elif PIPELINE is None:
            st.error("Model not available. Place fake_job_pipeline.pkl in the app folder.")
        else:
            proba = PIPELINE.predict_proba([text_input])[0][1]
            pred = int(proba >= 0.5)
            label = "⚠️ Suspicious / Fake" if pred == 1 else "✅ Likely Real"
            # show results
            st.markdown(f"### Result: {label}")
            st.metric("Probability (fake)", f"{proba:.3f}")
            # progress bar approximation
            st.progress(min(max(proba, 0.0), 1.0))
            # show highlighted text
            matches = find_keyword_matches(text_input)
            st.markdown("**Detected suspicious keywords:** " + (", ".join(matches) if matches else "None"))
            html = highlight_text(text_input, matches)
            st.markdown("<div style='padding:12px;border-radius:8px;border:1px solid #eee;background:#fafafa'>" +
                        f"{html}" + "</div>", unsafe_allow_html=True)

            # show top contributors
            contribs = top_contributors(PIPELINE, text_input, topn=10)
            if contribs:
                st.markdown("**Top contributing tokens** (TF-IDF × coefficient)")
                tbl = pd.DataFrame(contribs, columns=["token", "contribution"])
                st.table(tbl)
            else:
                st.info("Explainability not available for this model.")

with col2:
    st.subheader("Quick actions & tips")
    if st.button("Use sample (Fake, registration)"):
        title = "Work From Home Job"
        description = "Work from home, no experience required. Pay 999 registration charges. Weekly 25,000 income guaranteed."
        salary_text = ""
        # set via js? We can't set inputs programmatically in Streamlit easily; instruct user to paste.
        st.info("Sample text copied below — paste into the fields on left.")
        st.code(f"{title}\n\n{description}")

    st.markdown("**How it works**")
    st.write("""
    - TF-IDF converts text → numeric features  
    - Logistic Regression predicts probability of 'fake' class  
    - We highlight suspicious keywords and show top token contributions  
    """)

    st.markdown("---")
    st.subheader("Batch predict (CSV)")
    if uploaded:
        try:
            df_csv = pd.read_csv(uploaded)
            st.write("Preview of uploaded CSV:", df_csv.head())
            # simple check
            if not {"title","description"}.issubset(df_csv.columns):
                st.error("CSV must contain at least 'title' and 'description' columns.")
            else:
                if st.button("Run batch prediction on uploaded CSV"):
                    texts = (df_csv["title"].fillna("") + " " + df_csv["description"].fillna("") + " " + df_csv.get("salary_text", "").fillna("")).str.strip().tolist()
                    probs = PIPELINE.predict_proba(texts)[:,1]
                    preds = (probs >= 0.5).astype(int)
                    df_csv["prob_fake"] = probs
                    df_csv["pred_fake"] = preds
                    st.success("Batch predictions complete")
                    st.dataframe(df_csv.head(50))
                    # provide download
                    csv_out = df_csv.to_csv(index=False).encode("utf-8")
                    st.download_button("Download results CSV", data=csv_out, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

# use sample selection to populate inputs (workaround)
if "sample" in locals() and sample and sample != "— choose sample —":
    if "Fake" in sample and "registration" in sample:
        title_default = "Work From Home Job"
        desc_default = "Work from home, no experience required. Pay 999 registration charges. Weekly 25,000 income guaranteed."
    elif "1 lakh" in sample:
        title_default = "Quick Earning Job"
        desc_default = "Earn 1 lakh monthly. Investment required 2000."
    elif "Python developer" in sample:
        title_default = "Python Developer"
        desc_default = "Looking for Python developer with Django experience. 2+ years required. Apply on company site."
    elif "Frontend dev" in sample:
        title_default = "Frontend Developer"
        desc_default = "React developer required. 1–2 years experience. No charges."
    else:
        title_default = ""
        desc_default = ""

    # show the sample in a small box for user to copy
    st.markdown("---")
    st.markdown("### Selected sample (copy & paste into input fields)")
    st.code(f"{title_default}\n\n{desc_default}")

st.markdown("<br><br><hr><div style='text-align:center;color:gray'>Built with ❤️ — Streamlit demo</div>", unsafe_allow_html=True)
