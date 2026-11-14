# CustomerSense AI
### Customer Feedback Clustering and Sentiment Analysis Using NLP and Machine Learning

## Project Overview

CustomerSense AI is a machine learning and Natural Language Processing (NLP) project that analyzes customer reviews to discover customer segments, identify feedback patterns, and understand customer sentiment.

The project uses TF-IDF for text feature extraction, MiniBatch K-Means for customer review clustering, and VADER for sentiment analysis.

The results are presented through an interactive Streamlit dashboard.

## Objectives

- Analyze customer reviews.
- Clean and preprocess text data.
- Convert text into numerical features using TF-IDF.
- Group similar customer reviews using K-Means clustering.
- Identify positive, negative, and neutral feedback.
- Visualize customer feedback patterns through an interactive dashboard.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- MiniBatch K-Means Clustering
- VADER Sentiment Analysis
- Streamlit
- Plotly
- Matplotlib
- Seaborn

## Project Workflow

1. Dataset Collection
2. Exploratory Data Analysis (EDA)
3. Text Preprocessing
4. TF-IDF Feature Extraction
5. K-Means Clustering
6. Cluster Evaluation
7. Sentiment Analysis
8. Interactive Dashboard Development

## Project Structure

```text
CustomerSense-AI/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── models/
│
├── src/
│   ├── load_dataset.py
│   ├── eda_analysis.py
│   ├── text_preprocessing.py
│   ├── feature_extraction.py
│   ├── clustering.py
│   ├── model_evaluation.py
│   └── sentiment_analysis.py
│
├── app/
│   └── app.py
│
├── outputs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dashboard Features

- Executive Overview
- Customer Segments
- Sentiment Intelligence
- Review Explorer
- Model Performance
- Interactive filters and visualizations

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd CustomerSense-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Dashboard

```bash
streamlit run app/app.py
```

## Machine Learning Techniques

**TF-IDF:** Converts customer review text into numerical features.

**MiniBatch K-Means:** Groups