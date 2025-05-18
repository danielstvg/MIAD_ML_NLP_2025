from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Carga los objetos guardados
vectorizer = joblib.load('vectorizer.pkl')                 # TF-IDF vectorizer
model = joblib.load('modelo_multilabel_generos_rf.pkl')    # Nuevo modelo Random Forest multilabel
mlb = joblib.load('mlb_binarizer.pkl')                     # MultiLabelBinarizer

@app.route('/')
def home():
    return "API para predicción de géneros de películas corriendo"

@app.route('/predict_genres', methods=['POST'])
def predict_genres():
    # Recibe el JSON con el argumento 'plot'
    data = request.get_json()
    plot_text = data.get('plot', '')

    if not plot_text:
        return jsonify({'error': 'Se requiere el campo plot'}), 400

    # Vectorizar el texto
    X_vect = vectorizer.transform([plot_text])

    # Predecir con el modelo
    y_prob = model.predict_proba(X_vect)
    y_pred_bin = (y_prob >= 0.5).astype(int)

    # Convertir de binario a nombres de géneros
    predicted_genres = mlb.inverse_transform(y_pred_bin)

    return jsonify({'predicted_genres': predicted_genres[0] if predicted_genres else []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
