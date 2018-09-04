import numpy as np
import flask
import keras
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import cv2 as cv



app = Flask(__name__)
model = load_model('models/model_all.h5')
graph = tf.get_default_graph()


@app.route("/")
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        datas = []
        label_data = pd.read_csv("labels_all.csv",delimiter=",")
        file = request.files['image']
        test_image = misc.imread(file, flatten=True)

        #resize the input image for prediction
        test_image = cv.resize(test_image, (32, 32))
        datas.append(test_image.flatten())
        X_test = np.array(datas)
        X_test /= 255

        with graph.as_default():
            prediction = model.predict_classes(X_test)
            probability = model.predict_proba(X_test)

        #squeeze value from 1D array
        label = int(np.squeeze(prediction))
        max_probability=np.amax(probability)

        return render_template('index.html',label = label_data.iloc[label,:].values[0],probability=max_probability*100)


if __name__ == '__main__':
    app.run(debug=True)
