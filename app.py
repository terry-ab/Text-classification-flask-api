import flask
import numpy as np
from keras.models import load_model
from flask import Flask,render_template,request
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import pickle

app=Flask(__name__)
graph = tf.get_default_graph()
sess = tf.Session()
set_session(sess)
model=load_model('textmodel.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global sess
    global graph
    maxlen=4491
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    le=preprocessing.LabelEncoder()
    le.classes_ = np.load('classes.npy')
    newtext=[str(request.form["comment"])]
    print(newtext)
    with graph.as_default():
        set_session(sess)
        pre=model.predict_classes(pad_sequences(tokenizer.texts_to_sequences(newtext),maxlen=maxlen,padding='post'))
    result=le.inverse_transform(pre)[0]

    return render_template('index.html',prediction_text='Predicted class is :   {}'.format(result))

if __name__=="__main__":
    app.run(debug=True)
