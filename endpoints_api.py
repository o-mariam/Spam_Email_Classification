from flask import Flask,request,jsonify
from keras.models import load_model  
from keras.preprocessing.text import tokenizer_from_json

import json
import tensorflow as tf
import tensorflow as tf

print("Loading model...")
model = load_model("/Users/mariadasina/Downloads/Spam_Email_Classification-main/model/spam_classifier.keras",compile=False)

print("Loading tokenizer...")
with open("tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())

print("Loading pipeline...")
with open("pipeline_metadata.json", "r") as f:
        metadata = json.load(f)


app=Flask(__name__)


@app.route('/model/info',methods=["GET"])
def model_info():


    result={
            "max_length" : metadata["max_length"],
            "accuracy": metadata["metrics"]["accuracy"]
            }
    return jsonify(result)

@app.route('/model/email',methods=["POST"])
def model_email():
    data = request.get_json()
    email=data['email_text']

    sequences=tokenizer.texts_to_sequences([email])
    padded=tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen=metadata['max_length'])
    prediction=model.predict(padded)


    return jsonify({"class": "spam" if prediction[0][0] > 0.5 else "not_spam"})


@app.route('/model/emails',methods=["POST"])
def model_emails():
    data = request.get_json()
    emails=data['email_texts']

    sequences=tokenizer.texts_to_sequences(emails)
    padded=tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen=metadata['max_length'])
    predictions=model.predict(padded)

    result=[]
    for i in range(len(predictions)):                                     
        result.append("spam" if predictions[i][0] > 0.5 else "not_spam")
        
    return jsonify({"classes":result})



if __name__=="__main__":
    app.run(debug=True)
