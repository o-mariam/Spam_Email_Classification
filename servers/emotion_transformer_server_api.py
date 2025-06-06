from flask import Flask,request,jsonify
from keras.models import load_model  
from keras.preprocessing.text import tokenizer_from_json

import json
import tensorflow as tf
import tensorflow as tf
from setfit import SetFitModel

# print("Loading models...")
# model = load_model("/Users/mariadasina/Downloads/Spam_Email_Classification-main/model/spam_classifier.keras",compile=False)
#
# print("Loading tokenizer...")
# with open("../models/neural/tokenizer.json", "r") as f:
#     tokenizer = tokenizer_from_json(f.read())
#
# print("Loading pipeline...")
# with open("../models/neural/pipeline_metadata.json", "r") as f:
#         metadata = json.load(f)

model_path = "./models/boltuix/bert-emotion" # Or a URL to the model on Hugging Face Hub
model = SetFitModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
app=Flask(__name__)


@app.route('/model/info',methods=["GET"])
def model_info():


    result={
            "model" : "SetfitModel",
            "accuracy":0.6
            }
    return jsonify(result)

@app.route('/model/email',methods=["POST"])
def model_email():
    data = request.get_json()
    email=data['email_text']

    prediction=model.predict([email])

    return jsonify({"class": prediction.tolist()[0]})


@app.route('/model/emails',methods=["POST"])
def model_emails():
    data = request.get_json()
    emails=data['email_texts']

    predictions = model.predict(emails)
    return jsonify({"class": predictions.tolist()})


if __name__=="__main__":
    app.run(debug=True, port=5001)
