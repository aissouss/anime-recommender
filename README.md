# Anime & Game Recommender — Content-Based AI System

This project implements a **content-based recommendation system** using:

* **TF-IDF + cosine similarity** (fast model)
* **BERT embeddings** via *sentence-transformers* (advanced semantic model)
* A clean **Streamlit web interface**
* A dataset cleaned from the *Anime Recommendation Database 2020*

The app suggests similar animes based on genre, type, and semantic similarity.

---

## 🚀 Features

* Content-based filtering using TF-IDF
* Deep semantic similarity using BERT
* User-friendly Streamlit interface
* Automatic data cleaning pipeline
* Ready for deployment on Streamlit Cloud

---

## 📦 Project Structure

```
anime-recommender/
│── app.py                 # Streamlit UI
│── prepare_data.py        # Data cleaning script (Kaggle → clean CSV)
│── requirements.txt       # Dependencies
│
├── src/
│   └── recommender.py     # TF-IDF + BERT recommender models
│
└── data/
    ├── anime.csv          # Raw Kaggle dataset
    └── anime_clean.csv    # Clean dataset used by the app
```

---

## 📊 Models Used

### **TF-IDF Model**

* Vectorizes genres and titles
* Computes similarity via cosine distance
* Lightweight and fast

### **BERT Embeddings**

* Encodes anime descriptions using `all-MiniLM-L6-v2`
* High-quality semantic recommendations
* Captures meaning beyond keywords

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

## 🌐 Deployment

Can be deployed easily through **Streamlit Cloud** for public access.

---

## ✨ Author

Aissouss — Machine Learning & Data Science Projects.
*Anime & game recommendation powered by modern NLP.*
