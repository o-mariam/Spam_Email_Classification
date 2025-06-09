  # 📧 Email Spam Classification System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.12-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Flask-2.3-lightgrey" alt="Flask">
</div>

## 🌐 Project Overview

A full-stack email spam detection system using three different machine learning approaches, each exposed through a REST API. A Python client is included for easy interaction with the models.

### 🔍 Models Implemented

1. **Feedforward Neural Network with GloVe Embeddings**  
   - A dense neural network using pre-trained GloVe word vectors and a Flatten + Dense architecture.

2. **SetFit with E5 Embeddings**  
   - Uses `sentence-transformers/paraphrase-mpnet-base-v2` for embedding generation and SetFit for classification.

3. **SetFit with Emotion Embeddings**  
   - Uses `j-hartmann/emotion-english-distilroberta-base` as the embedding model for SetFit classification.

Each model is:
- Feedforward Neural Network with Pretrained GloVe Embeddings
- Includes consistent endpoints (`/info`, `/email`, `/emails`)
- Compatible with a shared Python client interface

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/o-mariam/Spam_Email_Classification.git
cd Spam_Email_Classification
```

# Install dependencies
pip install -r requirements.txt


---

## 🚀 API Overview

Each model runs on a dedicated Flask server and exposes the same set of endpoints for consistency.

### 🔗 API Instances

| Model                   | Base URL Example            |
|--------------------------|------------------------------|
| Feedforward + GloVe      | `http://localhost:5000/`     |
| SetFit + E5              | `http://localhost:5001/`     |
| SetFit + Emotion         | `http://localhost:5002/`     |

---

### 📡 Available Endpoints (per model)

#### 1. `/info` – Model Metadata

```http
GET /model/info
POS /model/email
POS /model/emails
```
