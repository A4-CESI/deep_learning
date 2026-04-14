import json, pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import tensorflow as tf

app = Flask(__name__)

def _fl(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
    bce    = -y_true*tf.math.log(y_pred)-(1-y_true)*tf.math.log(1-y_pred)
    p_t    = y_true*y_pred+(1-y_true)*(1-y_pred)
    return tf.reduce_mean(0.75*tf.pow(1-p_t,2.0)*bce)

model  = keras.models.load_model("model_production.keras", custom_objects={"loss":_fl})
with open("scaler_production.pkl","rb") as f: scaler = pickle.load(f)
with open("config_production.json") as f: cfg = json.load(f)
FEATURES=cfg["feature_names"]; SEUIL=cfg["seuil"]
RECALL=cfg["recall"]; AUC=cfg["auc"]; CFG_NAME=cfg["config"]

@app.route("/")
def index():
    return render_template("index.html", feature_names=FEATURES,
        config_name=CFG_NAME, seuil=round(SEUIL,3),
        recall=round(RECALL,4), auc=round(AUC,4))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json()
        values = [float(data.get(f,0)) for f in FEATURES]
        arr_sc = scaler.transform([values])
        prob   = float(model.predict(arr_sc, verbose=0).flatten()[0])
        pred   = int(prob >= SEUIL)
        return jsonify({"probability":round(prob,4),"prediction":pred,
            "label":"Diabetique" if pred==1 else "Non-diabetique",
            "seuil":round(SEUIL,3),"risque":"eleve" if pred==1 else "faible"})
    except Exception as e:
        return jsonify({"error":str(e)}), 400

@app.route("/health")
def health():
    return jsonify({"status":"ok","model":CFG_NAME,"recall":RECALL,"auc":AUC})

if __name__=="__main__":
    print(f"API demarree - {CFG_NAME}")
    print("-> http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
