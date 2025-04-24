#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import pickle

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Phishing Prediction API',
    description='Phishing Prediction API')

ns = api.namespace('predict', 
     description='Phishing Classifier')

with open('modelo_ensemble_xgb.pkl', 'rb') as f:
    modelo = pickle.load(f)

def predict_proba(url):
    prediction = modelo.predict([url])  
    return str(prediction[0])

parser = api.parser()

parser.add_argument(
    'URL', 
    type=str, 
    required=True, 
    help='URL to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_proba(args['URL'])
        }, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
