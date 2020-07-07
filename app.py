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
import os

Image_folder=os.path.join('static','icons')
app=Flask(__name__)
app.config['UPLOAD_FOLDER']=Image_folder
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
    if result=="business":
        img_file=os.path.join(app.config['UPLOAD_FOLDER'],'marketing.svg')
        img_name="Business"
    elif result=='entertainment':
        img_file=os.path.join(app.config['UPLOAD_FOLDER'],'entertain.svg')
        img_name="Entertainment"
    elif result=='politics':
        img_file=os.path.join(app.config['UPLOAD_FOLDER'],'politics.svg')
        img_name="Politics"
    elif result=='tech':
        img_file=os.path.join(app.config['UPLOAD_FOLDER'],'tech.svg')
        img_name="Technology"
    else:
        img_file=os.path.join(app.config['UPLOAD_FOLDER'],'sport.svg')
        img_name="Sports"
    return render_template('index.html',img_file=img_file,img_name=img_name)

if __name__=="__main__":
    app.run(debug=True)
