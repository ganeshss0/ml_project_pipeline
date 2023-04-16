from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
CORS(application)

@application.route('/')
@cross_origin()
def homepage():
    return render_template('index.html')

@application.route('/predict', methods = ['GET', 'POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            carat = float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color = request.form.get('color'),
            clarity = request.form.get('clarity')
        )
        final_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        predict = predict_pipeline.predict(final_data)

        result = round(predict[0], 2)

        return render_template('result.html', final_result = result)






if __name__ == '__main__':
    application.run(host = '0.0.0.0', debug = True)
