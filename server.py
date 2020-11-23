from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask_jsonpify import jsonify
import json   

import PredictService as PredictService
import ConvertImage as ConvertImageUtil

app = Flask(__name__)
api = Api(app)    

@app.route('/predict',methods = ['POST'])
def predict():
	if request.method == 'POST':
		body = request.form['body']
		data = json.loads(body)["data"]
		image_converted = ConvertImageUtil.converImage(data)
		result_predict = PredictService.predict(image_converted)
		return str(result_predict)

if __name__ == '__main__':
	app.run(host='192.168.43.111', port=5555)