import numpy as np 
import pandas as pd 

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
#initialize our neural network
from keras.models import Sequential
#to built our layers
from keras.layers import Dense

#Initialiasing the Ann
# we have to implement the Neural Network as a sequence of layers
classifier = Sequential()

#Adding Layers
# we are going to use the add function and the dence function
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))
#Addding second layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu' ))
#Output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid' ))

#Compiling ANN
#We are going to use the compile function
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)