import os
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from src.voicer_train import predicter

app = Flask(__name__)
api = Api(app)


class PredictJson(Resource):
    def post(self):
        file = request.files['file']
        file.save(file.filename)
        print(file.filename)
        if file.filename.split(".")[-1] == 'wav':
            predicted = predicter(file.filename)[0]
            os.remove(file.filename)
            print(predicted)
            return jsonify(predicted)
        return jsonify({"error": "404", "message": "we only support .wav"})


class PredictHtml(Resource):
    def post(self):
        file = request.files['file']
        file.save(file.filename)
        predicted = predicter(file.filename)[1]
        os.remove(file.filename)
        return predicted


api.add_resource(PredictJson, '/json')
api.add_resource(PredictHtml, '/html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)