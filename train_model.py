import os
import re
import string
import joblib
import pandas as pd
import numpy as np

from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

# -----------------------------------
# Download NLTK resources
# -----------------------------------

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# -----------------------------------
# Create folders
# -----------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# -----------------------------------
# Load Dataset
# -----------------------------------

try:
    df = pd.read_csv("data/Resume.csv")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

print("Dataset Shape:", df.shape)

# Keep required columns

df = df[["ID", "Resume_str", "Category"]]

df.rename(
    columns={
        "Resume_str": "Resume"
    },
    inplace=True
)

df.dropna(inplace=True)

# -----------------------------------
# Skill Dictionary
# -----------------------------------

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

# -----------------------------------
# NLP Setup
# -----------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------------
# Clean Resume Text
# -----------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)

    text = re.sub(r"\S+@\S+", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = re.sub(
        f"[{re.escape(string.punctuation)}]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# -----------------------------------
# NLP Processing
# -----------------------------------

def preprocess(text):

    text = clean_text(text)

    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

# -----------------------------------
# Skill Extraction
# -----------------------------------

def extract_skills(text):

    found = []

    text = text.lower()

    for skill in SKILLS:

        if skill.lower() in text:
            found.append(skill)

    return list(set(found))

# -----------------------------------
# Process Resumes
# -----------------------------------

print("Processing resumes...")

df["processed_resume"] = df["Resume"].apply(preprocess)

df["skills"] = df["Resume"].apply(extract_skills)

# -----------------------------------
# TF-IDF Training
# -----------------------------------

print("Training TF-IDF Vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=5000
)

resume_vectors = vectorizer.fit_transform(
    df["processed_resume"]
)

# -----------------------------------
# Save Models
# -----------------------------------

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

joblib.dump(
    df,
    "models/processed_resumes.pkl"
)

print("Models saved successfully!")

# -----------------------------------
# EDA
# -----------------------------------

print("Generating EDA charts...")

# Category Distribution

plt.figure(figsize=(12,6))

df["Category"].value_counts().head(15).plot(
    kind="bar"
)

plt.title("Resume Category Distribution")

plt.tight_layout()

plt.savefig(
    "static/images/category_distribution.png"
)

plt.close()

# -----------------------------------
# Skill Frequency
# -----------------------------------

all_skills = []

for skills in df["skills"]:
    all_skills.extend(skills)

skill_counts = Counter(all_skills)

skill_df = pd.DataFrame(
    skill_counts.items(),
    columns=["Skill","Count"]
)

skill_df = skill_df.sort_values(
    by="Count",
    ascending=False
)

plt.figure(figsize=(12,6))

sns.barplot(
    x="Count",
    y="Skill",
    data=skill_df.head(15)
)

plt.title("Top Skills")

plt.tight_layout()

plt.savefig(
    "static/images/top_skills.png"
)

plt.close()

# -----------------------------------
# Resume Length Distribution
# -----------------------------------

df["resume_length"] = df["Resume"].apply(len)

plt.figure(figsize=(10,6))

sns.histplot(
    df["resume_length"],
    bins=30
)

plt.title("Resume Length Distribution")

plt.tight_layout()

plt.savefig(
    "static/images/resume_length.png"
)

plt.close()

# -----------------------------------
# Word Cloud
# -----------------------------------

all_text = " ".join(
    df["processed_resume"]
)

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(all_text)

plt.figure(figsize=(14,7))

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "static/images/wordcloud.png"
)

plt.close()

print("EDA completed!")

print("Project training completed successfully.")