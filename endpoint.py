from flask import Flask,request,jsonify
from tensorflow.python.keras.models  import load_model
import json
import tensorflow as tf


import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

Tokenizer = tf.keras.preprocessing.text.Tokenizer
tokenizer = Tokenizer()
tokenizer_json = tokenizer.to_json() 
model = load_model("spam_classifier.keras")

with open("pipeline_metadata.json", "r") as f:
        metadata = json.load(f)



app=Flask(__name__)

@app.route('/',methods=["GET"])
def model_info():


    result={
            "max_length" : metadata["max_length"],
            "accuracy": metadata["metrics"]["accuracy"]
            }
    return jsonify(result)





if __name__=="__main__":
    app.run(debug=True)