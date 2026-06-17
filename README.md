# Resume Screening & Candidate Ranking System

AI-powered Resume Screening and Candidate Ranking System developed for **Future Interns Task 3**.

The project uses Natural Language Processing (NLP), TF-IDF Vectorization, Cosine Similarity, Skill Extraction, and Flask to automatically rank resumes against a Job Description (JD).

---

## Project Overview

Recruiters often receive hundreds of resumes for a single job opening.

This project automates the screening process by:

- Cleaning and preprocessing resumes
- Extracting skills
- Matching resumes with Job Descriptions
- Ranking candidates
- Identifying missing skills
- Generating candidate scorecards

---

## Dataset

Dataset Used:

Resume Dataset from Kaggle

https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Dataset Columns:

| Column | Description |
|----------|------------|
| ID | Resume Identifier |
| Resume_str | Resume Text |
| Resume_html | HTML Version |
| Category | Resume Category |

Used Columns:

- ID
- Resume_str
- Category

---

## Features

### Data Preprocessing

- Resume text cleaning
- URL removal
- Email removal
- Special character removal
- Lowercase conversion
- Tokenization
- Stopword removal
- Lemmatization

### Skill Extraction

Extract skills using predefined skill dictionary.

Examples:

- Python
- SQL
- Machine Learning
- TensorFlow
- AWS
- Docker
- Tableau
- Power BI

### Exploratory Data Analysis

- Resume category distribution
- Top skills analysis
- Word cloud visualization
- Resume length analysis
- Skill frequency charts

### Resume Screening Engine

Accept Job Description input.

Perform:

- TF-IDF Vectorization
- Cosine Similarity
- Candidate Ranking
- Missing Skill Detection
- Candidate Recommendation

### Visualization

- Candidate ranking charts
- Similarity score analysis
- Skill-gap charts
- Resume insights

### Flask Web Application

Users can:

- Paste Job Description
- Screen Candidates
- View Rankings
- View Similarity Scores
- View Candidate Skills
- View Missing Skills
- Download Results CSV

---

## Project Structure

```text
Resume-Screening-System/
│
├── data/
│   └── Resume.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── processed_resumes.pkl
│
├── static/
│   ├── style.css
│   └── images/
│
├── templates/
│   ├── index.html
│   └── results.html
│
├── notebooks/
│   └── EDA.ipynb
│
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── screenshots/
```

---

## Installation

Clone repository:

```bash
git clone <repository-url>
cd Resume-Screening-System
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train Model

Run:

```bash
python train_model.py
```

Outputs:

```text
models/
├── tfidf_vectorizer.pkl
└── processed_resumes.pkl
```

EDA Charts:

```text
static/images/
├── category_distribution.png
├── top_skills.png
├── resume_length.png
├── wordcloud.png
```

---

## Run Application

Start Flask server:

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## Workflow

1. User enters Job Description
2. System extracts JD skills
3. TF-IDF vectorization performed
4. Cosine similarity calculated
5. Candidates ranked
6. Missing skills identified
7. Results displayed
8. CSV export available

---

## Sample Output

| Rank | Candidate ID | Similarity Score |
|--------|--------------|------------------|
| 1 | 1023 | 92.5% |
| 2 | 845 | 88.7% |
| 3 | 564 | 84.2% |

Recommendation:

- Strong Match
- Moderate Match
- Low Match

---

## Future Interns Task 3 Objectives Covered

### Data Processing

- Resume cleaning
- NLP preprocessing
- Skill extraction

### EDA

- Category analysis
- Skill analysis
- Word cloud

### Machine Learning

- TF-IDF Vectorization
- Cosine Similarity

### Candidate Ranking

- Resume scoring
- Ranking engine

### Evaluation

- Similarity analysis
- Ranking evaluation

### Deployment

- Flask web application

---

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-Learn
- NLTK
- spaCy
- Matplotlib
- Seaborn
- WordCloud

---

## Future Improvements

- BERT Resume Embeddings
- Semantic Search
- GPT-based Resume Analysis
- PDF Resume Upload
- Multi-JD Comparison
- Recruiter Dashboard
- PostgreSQL Database Integration

---

## Author

Future Interns Task 3

Resume Screening & Candidate Ranking System

Machine Learning + NLP + Flask

