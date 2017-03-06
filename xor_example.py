from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
y = np.array([[1, 0],[0, 1],[0, 1],[1, 0]])

model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

model.fit(X, y, batch_size=1, nb_epoch=500)
print(model.predict_proba(X))