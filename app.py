import os
import io
import joblib
import pandas as pd
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    session
)

from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# Flask App Setup
# ==========================================

app = Flask(__name__)
app.secret_key = "resume_screening_secret_key"

# ==========================================
# Load Models
# ==========================================

try:
    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    resumes_df = joblib.load(
        "models/processed_resumes.pkl"
    )

    print("Models loaded successfully.")

except Exception as e:

    print("Error loading models:", e)

    vectorizer = None
    resumes_df = None

# ==========================================
# Skill Dictionary
# ==========================================

SKILLS = [
    "python",
    "java",
    "sql",
    "machine learning",
    "deep learning",
    "nlp",
    "tensorflow",
    "pytorch",
    "data science",
    "data analysis",
    "flask",
    "django",
    "html",
    "css",
    "javascript",
    "react",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "git",
    "power bi",
    "tableau",
    "excel",
    "communication",
    "leadership",
    "project management"
]

# ==========================================
# Extract Skills
# ==========================================

def extract_skills(text):

    text = str(text).lower()

    found = []

    for skill in SKILLS:

        if skill.lower() in text:
            found.append(skill)

    return sorted(list(set(found)))

# ==========================================
# Candidate Ranking Function
# ==========================================

def rank_candidates(job_description, top_n=20):

    if resumes_df is None:
        raise Exception("Resume database not loaded")

    jd_skills = extract_skills(job_description)

    jd_vector = vectorizer.transform(
        [job_description]
    )

    resume_vectors = vectorizer.transform(
        resumes_df["processed_resume"]
    )

    similarity_scores = cosine_similarity(
        jd_vector,
        resume_vectors
    ).flatten()

    results_df = resumes_df.copy()

    results_df["Similarity Score"] = (
        similarity_scores * 100
    ).round(2)

    results_df["Matched Skills"] = results_df[
        "skills"
    ].apply(
        lambda x: list(
            set(x).intersection(jd_skills)
        )
    )

    results_df["Missing Skills"] = results_df[
        "skills"
    ].apply(
        lambda x: list(
            set(jd_skills) - set(x)
        )
    )

    results_df["Skill Match Count"] = results_df[
        "Matched Skills"
    ].apply(len)

    results_df = results_df.sort_values(
        by="Similarity Score",
        ascending=False
    )

    return results_df.head(top_n), jd_skills

# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ==========================================
# Screening Route
# ==========================================

@app.route(
    "/screen",
    methods=["POST"]
)
def screen():

    try:

        jd = request.form.get(
            "job_description"
        )

        top_n = request.form.get(
            "top_n",
            10
        )

        top_n = int(top_n)

        if not jd.strip():

            return render_template(
                "index.html",
                error="Please enter a Job Description."
            )

        ranked_df, jd_skills = rank_candidates(
            jd,
            top_n
        )

        session["last_jd"] = jd

        session["top_n"] = top_n

        csv_data = ranked_df.to_json()

        session["results"] = csv_data

        candidates = ranked_df.to_dict(
            orient="records"
        )

        return render_template(
            "results.html",
            candidates=candidates,
            jd_skills=jd_skills,
            total_candidates=len(candidates)
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )

# ==========================================
# Download CSV
# ==========================================

@app.route("/download")
def download():

    try:

        if "results" not in session:

            return "No results available."

        results_df = pd.read_json(
            io.StringIO(
                session["results"]
            )
        )

        output = io.BytesIO()

        results_df.to_csv(
            output,
            index=False
        )

        output.seek(0)

        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name="ranked_candidates.csv"
        )

    except Exception as e:

        return str(e)

# ==========================================
# Health Check
# ==========================================

@app.route("/health")
def health():

    return {
        "status": "running",
        "model_loaded": vectorizer is not None
    }

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )