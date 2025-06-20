import nltk
nltk.download('punkt_tab')
import numpy as np
import pandas as pd 

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import random
import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import LabelEncoder

#Cargar los datos
path= "/content/drive/MyDrive/Bootcamp_IA_TalentoTech/Chatbot/intents.json"

with open(path, 'r', encoding='utf-8') as file:
    data= json.load(file)

#Crear el stemmer
stemmer= PorterStemmer()

#Preprocesamiento
vocab= []
tags= []
patterns= []
labels=[]

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens= word_tokenize(pattern.lower())
        stemmed= [stemmer.stem(w) for w in tokens]
        vocab.extend(stemmed)
        patterns.append(stemmed)
        labels.append(intent['tag'])
    if intent['tag'] not in tags:
        tags.append(intent['tag'])

vocab = sorted(set(vocab))

# One-hot input
X = []
Y= []

encoder= LabelEncoder()
encoder_labels= encoder.fit_transform(labels)

for pattern in patterns:
    bag= [1 if word in pattern else 0 for word in vocab ] # pondra 1 a cada palabra que reconozca
    X.append(bag)

Y= encoder_labels

#Convertir las variables a arreglos de numpy

X= np.array(X) #Cantidad de datos por fila 
Y= np.array(Y)

#Modelo 

D= len(X[0]) #Cantidad de entradas (filas)
C= len(tags) #Cantidad de etiquetas

model= Sequential()
# Capa de entrada - Densa
model.add(Dense(8, input_shape= (D,), activation= 'relu')) #Se deja vacio despues del "D" porque no se sabe cuantas salidas tendra la capa
# Capa Densa 2
model.add(Dense(8, activation='relu'))
model.add(Dense(C, activation= 'softmax')) #Se le asigna al menos una neurona a cada categoria, C es el tamaño de etiqueta

#Compilar  
model.compile(
    loss= 'sparse_categorical_crossentropy',
    optimizer= 'adam',
    metrics= ['accuracy']
)

#entrenar
model.fit(X, Y, epochs= 200, verbose= 0)

#funcion para procesar la entrada
def predict_class(text):
    tokens= word_tokenize(text.lower())
    stemmed= [stemmer.stem(w) for w in tokens]
    bag= np.array([1 if word in stemmed else 0 for word in vocab])
    res= model.predict(np.array([bag]), verbose= 0)[0] #Predice de la lista de la bolsa de palabras
    idx= np.argmax(res) #Indice
    tag= encoder.inverse_transform([idx])[0] #decodificar la etiqueta para que entregue palabras
    return tag

#Función para dar respuestas
def get_response(tag, context): #Reconoce dos entradas
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "No entendí eso, ¿Puedes repetirlo?"




