from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('modelo_popularidad_rf.pkl')

@app.route('/')
def home():
    return "API de Predicción de Popularidad en funcionamiento"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    pred = modelo.predict(df)
    return jsonify({'popularidad_predicha': float(pred[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
