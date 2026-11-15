# 📰 Fake News Identifier (Full-Stack NLP Web Application)

An end-to-end Natural Language Processing (NLP) web application that classifies news articles as **Real** or **Fake** with confidence probabilities. Built with a custom **PyTorch** Neural Network, **FastAPI** backend, and a responsive **Tailwind CSS** frontend.

🔗 **Live Web App:** [https://fake-news-web-fszb.onrender.com](https://fake-news-web-fszb.onrender.com)  
⚙️ **Backend API:** [https://fake-news-identifier-150i.onrender.com](https://fake-news-identifier-150i.onrender.com)

> **Note:** Both services are hosted on Render's free tier. If the server is inactive, the initial prediction request may take 40–50 seconds while the instance wakes up.

---

## 🚀 Key Features & Architecture

- **Dataset:** Trained and evaluated on the English Fake News Detection dataset (`ErfanMoosaviMonazzah/fake-news-detection-dataset-English` from Hugging Face).
- **NLP Preprocessing Pipeline:** Lowercasing, regex noise removal (URLs, HTML tags, brackets, punctuation, digits), tokenization, stopword removal, and WordNet lemmatization via **NLTK**.
- **Feature Engineering:** TF-IDF Vectorization (`max_features=5000`) using **Scikit-learn**.
- **Deep Learning Model (`CheckNewsNN`):** Custom Multi-Layer Perceptron built in **PyTorch** (`5000 → 512 → 256 → 2`) with ReLU activations and Softmax probability output, achieving **~98.6% validation accuracy**.
- **Full-Stack Deployment:** RESTful API built with **FastAPI** and **Uvicorn**, connected to a lightweight single-page frontend deployed on **Render**.

---

## 📂 Project Structure

```text
Fake-News-Identifier/
├── backend/
│   ├── models/
│   │   ├── fake_news_model.pth      # Trained PyTorch model weights
│   │   └── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
│   ├── main.py                      # FastAPI application & inference pipeline
│   └── requirements.txt             # Backend Python dependencies
├── frontend/
│   └── index.html                   # Web UI (HTML, Tailwind CSS, Vanilla JS)
├── notebooks/
│   └── fake_news_analyser.ipynb     # Data exploration, preprocessing & model training
├── .gitignore
└── README.md