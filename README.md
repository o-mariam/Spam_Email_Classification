# 📧 Email Spam Classification System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.12-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Flask-2.3-lightgrey" alt="Flask">
</div>

## 🌟 Overview
A production-ready system combining:
- **Deep Learning Model**: Bidirectional LSTM neural network
- **REST API**: Flask-based classification service
- **NLP Pipeline**: Custom text preprocessing

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/o-mariam/Spam_Email_Classification.git
cd spam-classifier

# Install dependencies
pip install -r requirements.txt
```
## 📊 Performance Metrics

| Metric       | Not Spam | Spam   |
|--------------|----------|--------|
| **Precision** | 0.982    | 0.978  |
| **Recall**    | 0.991    | 0.985  |
| **F1-Score**  | 0.986    | 0.981  |
| **Support**   | 4,821    | 3,752  |


## 🚀 API Endpoints
Endpoint	Method	Input	Output
- /predict	POST	{"email_text":"..."}	{"class": "spam", "confidence": 0.98}
- /batch_predict	POST	{"emails":["...", "..."]}	List of predictions
