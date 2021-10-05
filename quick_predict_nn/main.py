import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

# from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

K.set_image_data_format("channels_last")

print("Loading the data...")
envelope = np.load('./database/envelope.npy')
donut = np.load('./database/donut.npy')
snowflake = np.load('./database/snowflake.npy')
see_saw = np.load('./database/see saw.npy')
mountain = np.load('./database/mountain.npy')
ocean = np.load('./database/ocean.npy')


print("Number of images in dataset and numpy array size of each image:")
print(envelope.shape)
print(donut.shape)
print(snowflake.shape)
print(see_saw.shape)
print(mountain.shape)
print(ocean.shape)

# add a column with labels, 0=cat, 1=sheep, 2=snowflake, 3=see_saw 
envelope = np.c_[envelope, np.zeros(len(envelope))]
donut = np.c_[donut, np.ones(len(donut))]
snowflake = np.c_[snowflake, np.full(len(snowflake), 2)]
see_saw = np.c_[see_saw, np.full(len(see_saw), 3)]
mountain = np.c_[mountain, np.full(len(mountain), 4)]
ocean = np.c_[ocean, np.full(len(ocean), 5)]

# merge the envelope, donut, snowflake, mountain, ocean and see_saw arrays, and split the features (X) and labels (y).
# Convert to float32 to save some memory.
X = np.concatenate((envelope[:20000, :-1], donut[:20000, :-1], snowflake[:20000, :-1], see_saw[:20000, :-1],
                    mountain[:20000, :-1], ocean[:20000, :-1]), axis=0).astype('float32')
y = np.concatenate((envelope[:20000, -1], donut[:20000, -1], snowflake[:20000, -1], see_saw[:20000, -1],
                    mountain[:20000, -1], ocean[:20000, -1]), axis=0).astype('float32')  # the last column

# train/test split (divide by 255 to obtain normalized values between 0 and 1)
X_train, X_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.5, random_state=0)

# one hot encode outputs
y_train_cnn = to_categorical(y_train)
y_test_cnn = to_categorical(y_test)
num_classes = y_test_cnn.shape[1]

X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# define the CNN model
def cnn_model():
    # create model
    model = Sequential()

    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


np.random.seed(0)
# build the model
model_cnn = cnn_model()
# Fit the model
model_cnn.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=15, batch_size=200)
# Final evaluation of the model
scores = model_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)

print('Final CNN accuracy: ', scores[1])
# Saving the model prediction
model_cnn.save('./saved_model/')

# Finding the accuracy score
y_pred_cnn = model_cnn.predict_classes(X_test_cnn, verbose=0)
acc_cnn = accuracy_score(y_test, y_pred_cnn)
print ('CNN accuracy: ', acc_cnn)
