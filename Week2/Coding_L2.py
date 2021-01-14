import keras

from keras.models import Sequential
from keras.layers import Dense


#Initialiasing the Ann
# we have to thing the Neural Network as a sequence of layers
classifier = Sequential()


#Adding Layers
# we are going to use the add function and the dence function

#input layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))

#hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling ANN
#We are going to use the compile function
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
