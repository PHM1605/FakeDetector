import os, glob
import numpy as np 
from PIL import Image, ImageChops, ImageEnhance
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

from utils import parse_images_and_labels

X_train, Y_train, X_val, Y_val = [], [], [], []
train_files_real = glob.glob('data/train/real/*')
train_files_fake = glob.glob('data/train/fake/*')
val_files_real = glob.glob('data/validation/real/*')
val_files_fake = glob.glob('data/validation/fake/*')

X_train_real, Y_train_real = parse_images_and_labels(train_files_real, 0)
X_train_fake, Y_train_fake = parse_images_and_labels(train_files_fake, 1)
X_val_real, Y_val_real = parse_images_and_labels(val_files_real, 0)
X_val_fake, Y_val_fake = parse_images_and_labels(val_files_fake, 1)

# build model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', activation='relu', input_shape=(128,128,3)))
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

optimizer = RMSprop(learning_rate=0.0005, rho=0.9, epsilon=1e-8, decay=0.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
epochs = 100
batch_size = 5
print(X_train_real.shape, X_train_fake.shape)
X_train = np.concatenate((X_train_real, X_train_fake))
Y_train = np.concatenate((Y_train_real, Y_train_fake))
X_val = np.concatenate((X_val_real, X_val_fake))
Y_val = np.concatenate((Y_val_real, Y_val_fake))
early_stopping = EarlyStopping(monitor='val_acc', restore_best_weights=True, mode="max")
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), verbose=2, shuffle=True, callbacks=[early_stopping])
model.save('best.h5')


