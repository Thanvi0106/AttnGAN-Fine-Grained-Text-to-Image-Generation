from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pickle
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from keras.models import Sequential
from keras.layers import Dense, Flatten, Bidirectional, LSTM, RepeatVector, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint
import os
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score

global uname
global X_train, X_test, y_train, y_test, tfidf_vectorizer, sc
global model
global filename
global X, Y, dataset

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

X = np.load("model/X.npy")
Y = np.load("model/Y.npy")
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
X = tfidf_vectorizer.fit_transform(X).toarray()
data = X
sc = StandardScaler()
X = sc.fit_transform(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
Y = np.reshape(Y, (Y.shape[0], (Y.shape[1] * Y.shape[2] * Y.shape[3])))
Y = Y.astype('float32')
Y = Y/255
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

model = Sequential()
#creating gan model
model.add(Conv2D(32, (1, 1), activation='relu', input_shape=(X.shape[1], X.shape[2], X.shape[3])))
#max layer to collect relevant features from gan layer
model.add(MaxPooling2D((1, 1)))
#adding another layer
model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(MaxPooling2D((1, 1)))
model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(MaxPooling2D((1, 1)))
model.add(Flatten())
model.add(RepeatVector(2))
#adding spatial attention model
model.add(Bidirectional(LSTM(128, activation = 'relu')))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(Y.shape[1], activation='sigmoid'))
# Compile and train the model.
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = model.fit(X_train, y_train, batch_size = 16, epochs = 15, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    model.load_weights("model/cnn_weights.hdf5")
f = open('model/cnn_history.pckl', 'rb')
data = pickle.load(f)
f.close()
print(data['accuracy'])
accuracy_value = 1 - data['accuracy'][14]

def PredictPerformance(request):
    if request.method == 'GET':
       return render(request, 'PredictPerformance.html', {})

def TexttoImageAction(request):
    if request.method == 'POST':
        global tfidf_vectorizer, sc, model
        text_data = request.POST.get('t1', False)
        answer = text_data.lower().strip()
        model = load_model("model/cnn_weights.hdf5")
        data = answer
        data = cleanText(data)
        test = tfidf_vectorizer.transform([data]).toarray()
        test = sc.transform(test)
        test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
        predict = model.predict(test)
        predict = predict[0]
        predict = np.reshape(predict, (128, 128, 3))
        predict = cv2.resize(predict, (300, 300))
        img = cv2.cvtColor(predict, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':"Text = "+text_data, 'img': img_b64}
        return render(request, 'ViewResult.html', context)    

def TrainModel(request):
    if request.method == 'GET':
        global accuracy_value  
        context= {'data':"Spatial Attention GAN Model Accuracy : "+str(accuracy_value)}
        return render(request, 'ViewResult.html', context)        

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def AdminLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        if username == "admin" and password == "admin":
            context= {'data':'welcome '+username}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'AdminLogin.html', context)          


def TexttoImage(request):
    if request.method == 'GET':
       return render(request, 'TexttoImage.html', {})
        
