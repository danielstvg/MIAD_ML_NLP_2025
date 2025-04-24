from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Carga del pipeline completo (preprocesamiento + modelo)
pipeline = joblib.load('modelo_popularidad_rf.pkl')

@app.route('/')
def home():
    return "API de predicción de popularidad de canciones corriendo"

@app.route('/predict', methods=['POST'])
def predict():
    # Recibe un JSON con las características de la canción (sin columnas irrelevantes)
    data = request.get_json()
    df = pd.DataFrame([data])
    # El pipeline cargado internamente aplicará todo el preprocesamiento
    pred = pipeline.predict(df)
    return jsonify({'popularidad_predicha': float(pred[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
