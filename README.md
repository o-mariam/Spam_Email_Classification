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

## Model Performance Metrics

### Classification Report (Spam Detection)
| Metric       | Class 0 (Not Spam) | Class 1 (Spam) | Macro Avg | Weighted Avg |
|--------------|-------------------|----------------|-----------|--------------|
| **Precision** | 0.88              | 0.96           | 0.92      | 0.91         |
| **Recall**    | 0.98              | 0.77           | 0.88      | 0.91         |
| **F1-Score**  | 0.93              | 0.86           | 0.89      | 0.90         |
| **Support**   | 240               | 135            | -         | 375          |

**Accuracy**: 0.91

### Key Takeaways:
- **Not Spam Detection (Class 0)**:
  - High recall (0.98) → Effectively catches most legitimate messages
  - Precision (0.88) → Some false positives (legitimate messages flagged as spam)
  
- **Spam Detection (Class 1)**:
  - High precision (0.96) → Very few false alarms (spam predictions are reliable)
  - Lower recall (0.77) → Misses ~23% of actual spam


## 🚀 API Endpoints
Endpoint	Method	Input	Output
- /predict	POST	{"email_text":"..."}	{"class": "spam", "confidence": 0.98}
- /batch_predict	POST	{"emails":["...", "..."]}	List of predictions
