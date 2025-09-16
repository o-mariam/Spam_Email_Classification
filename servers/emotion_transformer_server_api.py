from flask import Flask,request,jsonify

from setfit import SetFitModel


model_path = "./models/boltuix_bert_emotion"
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
    app.run(host="0.0.0.0",debug=False, port=5001)
