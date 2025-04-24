from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open('modelo_ensemble_xgb.pkl', 'rb') as f:
    modelo = pickle.load(f)

@app.route('/')
def home():
    return "API en Flask corriendo en EC2"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    pred = modelo.predict(df)
    return jsonify({'popularidad_predicha': float(pred[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
