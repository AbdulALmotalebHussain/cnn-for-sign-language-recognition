# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Input
import cv2
import matplotlib.pyplot as plt
import random as rd

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Corrected path to include 'Desktop' instead of 'ktop'
for dirname, _, filenames in os.walk('C:/Users/Ananymis/Desktop/mostafa/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

# Assuming the corrected path is accurate and the file exists
df_train = pd.read_csv("C:/Users/Ananymis/Desktop/mostafa/input/sign_mnist_test/sign_mnist_test.csv")

# Display the first few rows of the dataframe to verify successful loading
print(df_train.head())


alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
n = rd.randrange(df_train.shape[0])
ar = np.array(df_train.loc[n][1:]).reshape((28,28))
plt.imshow(ar, cmap='gray')
plt.title(alphabet[df_train.loc[n][0]])
plt.show()


y = df_train["label"]
X = df_train.drop(['label'], axis=1)

X = np.array(X)/255
y = np.array(y)

Y = np.zeros((len(alphabet),df_train.shape[0]))
for i in range(len(y)):
  Y[y[i],i] = 1
X = X.reshape((-1, 28,28,1))
Y = Y.reshape((26,-1))


model = tf.keras.Sequential()
model.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1),padding='same'))
model.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2,2), strides=None,padding='same'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu',padding='same'))
model.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2,2), strides=None,padding='same'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(556, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(26, activation='softmax'))

model.summary()

model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=["accuracy"])
history = model.fit(X,y,batch_size=64,epochs=3, validation_split=0.2) #training

import pandas as pd 

df_valid = pd.read_csv("C:/Users/Ananymis/Desktop/mostafa/input/sign_mnist_train/sign_mnist_train.csv")

#preprocessing
n = rd.randrange(df_valid.shape[0])
y = df_valid["label"]
X = df_valid.drop(['label'], axis=1)

ar = np.array(df_valid.loc[n][1:]).reshape((28,28))

X = np.array(X)/255
y = np.array(y)

Y = np.zeros((26,df_valid.shape[0]))
for i in range(len(y)):
  Y[y[i],i] = 1
X = X.reshape((-1, 28,28,1))
Y = Y.reshape((26,-1))

plt.imshow(ar, cmap='gray')
plt.title(f"Prediction :  {alphabet[ np.argmax(model.predict(X[n].reshape(1,28,28,1)))]} | had to predict {alphabet[df_valid.loc[n][0]]}")
plt.show()

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop loop if there are no frames to capture

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    roi = frame[100:300, 100:300]
    roi_resized = cv2.resize(roi, (28, 28))
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    roi_normalized = roi_gray / 255.0  # Normalize the image

    processed_frame = roi_normalized.reshape(1, 28, 28, 1)
    try:
        prediction = model.predict(processed_frame)
        predicted_letter = alphabet[np.argmax(prediction)]
        
        # Draw the predicted letter on the video frame within the green box
        cv2.putText(frame, predicted_letter, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Prediction error: {e}")

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop

cap.release()
cv2.destroyAllWindows()


