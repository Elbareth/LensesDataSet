# -*- coding: utf-8 -*-
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pandas

class Main:    
    def baselineModel(self):
        model = Sequential()
        model.add(Dense(13, input_dim=4, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    def createModel(self, datasetX, datasetY):        
        encoder = LabelEncoder()
        encoder.fit(datasetY)
        encoderY = encoder.transform(datasetY)
        dummyY = np_utils.to_categorical(encoderY)       
        #model.fit(datasetX, datasetY, epochs=150, batch_size=20)
        #_,accuracy = model.evaluate(datasetX, datasetY)
        #print('Accuracy %.2f' % (accuracy*100))        
        estimator = KerasClassifier(build_fn = self.baselineModel(), epochs = 100, batch_size = 2, verbose = 0)
        print(estimator)
        kfold = KFold(n_splits = 10, shuffle = True)       
        result = cross_val_score(estimator, datasetX, dummyY, cv = kfold)        
        print("Baseline: %.2f%% (%.2f%%)" % (result.mean() * 100, result.std() *100))
#dataset = loadtxt('lensesDataSet.csv', delimiter=',')
#datasetX = dataset[:,0:4]
#datasetY = dataset[:,4]
dataframe = pandas.read_csv("lensesDataSet.csv", header=None)
dataset = dataframe.values
datasetX = dataset[:,0:4].astype(float)
datasetY = dataset[:,4]
main = Main()
main.createModel(datasetX, datasetY)


