# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pandas
from keras.utils.np_utils import to_categorical

dataframe = pandas.read_csv("lensesDataSet2.csv", header = None)
dataset = dataframe.values
datasetX = dataset[:,0:4].astype(float)
datasetY = dataset[:,4]
predict = pandas.read_csv("lensesDataSetPredict.csv", header=None)
predictValues = predict.values
predictX = predictValues[:,0:4].astype(float)
predictY = predictValues[:,4]
encoder = LabelEncoder()
encoder.fit(datasetY)
encoderY = encoder.transform(datasetY)
dummyY = np_utils.to_categorical(encoderY)
model = Sequential()
model.add(Dense(17, input_dim=4, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
model.fit(datasetX, dummyY, epochs = 200, batch_size = 5)
scores = model.evaluate(datasetX, dummyY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict_classes(predictX)
prediction_ = np.argmax(to_categorical(predictions), axis = 1)
prediction_ = encoder.inverse_transform(prediction_)
for i, j in zip(prediction_ , predictY):
    print( " Keras przewidzial soczewke {}, a powinna byc {}".format(i,j))

