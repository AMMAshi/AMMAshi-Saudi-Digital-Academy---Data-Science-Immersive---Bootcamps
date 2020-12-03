# ====================================================================
# ====================================================================
# Author: Arwa Ashi
# Saudi Digital Academy 
# Speech Emotion Detection - Final Project - Dec 3rd, 2020
# ====================================================================
# ====================================================================


# Speech Emotion Detection Deep Learning Model code
# ====================================================================


# Calling the Data
# --------------------------------------------------------------------
from p1_SAVEE_DataFrame import df_Final


# importing pacakges
# --------------------------------------------------------------------
import pandas as pd
import numpy as np
# Model Training
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
# CNN Model
import tensorflow 
from tensorflow.keras.models import Sequential
# CNN Model layers
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Dense
# CNN Model Optimizers
from tensorflow.keras.optimizers import Adam
# CNN Model Callbacks
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
# Model Evalution
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# DataFrame
# --------------------------------------------------------------------
df       = df_Final[df_Final.emotion != 5]#'neutral']
features = df.drop(['path','emotion','source'],axis=1)
targets  = df['emotion']
#print(features.head())
#print(targets.head())


# Split into traning and test sets
# --------------------------------------------------------------------
training, testing = train_test_split(df, train_size=0.8, test_size=0.20, stratify=df['emotion'])
train, val        = train_test_split(training, train_size=0.8, test_size=0.20, stratify=training['emotion'])

X_train = train.drop(['path','emotion','source'],axis=1)
y_train = train['emotion']

X_val = val.drop(['path','emotion','source'],axis=1)
y_val = val['emotion']

X_test = testing.drop(['path','emotion','source'],axis=1)
y_test = testing['emotion']


# Using .ravel() converting 2D to 1D
# --------------------------------------------------------------------
X_train = np.array(X_train)
X_val   = np.array(X_val)
X_test  = np.array(X_test)

y_train = np.array(y_train).ravel()
y_val   = np.array(y_val).ravel()
y_test  = np.array(y_test).ravel()

print(y_train[:5]) # [4 7 1 1 4]


# One-Hot Encoding
# --------------------------------------------------------------------
lb = LabelEncoder()

y_train = utils.to_categorical(lb.fit_transform(y_train))
y_val   = utils.to_categorical(lb.fit_transform(y_val))
y_test  = utils.to_categorical(lb.fit_transform(y_test))

# 1 means which class has the highest probability
print(y_train)


# Changing Dimension for CNN Model
# --------------------------------------------------------------------
X_traincnn = np.expand_dims(X_train, axis=2)
X_valcnn   = np.expand_dims(X_val, axis=2)
X_testcnn  = np.expand_dims(X_test, axis=2)

print(X_testcnn.shape) # (108, 65, 1)


# CNN Model
# --------------------------------------------------------------------
def get_model():
    model = Sequential([
        # 1st layer
        Conv1D(256, 5, padding='same',input_shape=(X_traincnn.shape[1],X_traincnn.shape[2])),
        BatchNormalization(),
        Activation('relu'),
        # 2nd layer
        Conv1D(128, 5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.1),
        # 3rd layer
        #MaxPooling1D(pool_size=(8)),
        Conv1D(128, 5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Flatten(),
        Dense(y_train.shape[1]),
        Activation('softmax'), 
    ])
    opt = tensorflow.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    model.compile(
        loss      = 'categorical_crossentropy',#'categorical_crossentropy',#'mse',#binary_crossentropy,
        optimizer = opt,#'adam', # opt,
        metrics   = ['accuracy']
    )
    return model

get_model().summary()

es_cb   = EarlyStopping(monitor='val_loss', patience=5)
model   = get_model()
history = model.fit(
    X_traincnn, y_train,
    batch_size      = 16,
    validation_data = (X_testcnn, y_test),
    epochs          = 100,
    callbacks       = [es_cb]
    )


# Model Evalution
# --------------------------------------------------------------------
def plot_loss(history):
    h     = history.history
    x_lim = len(h['loss'])
    plt.figure(figsize=(8, 8))
    plt.plot(range(x_lim), h['val_loss'], label = 'Validation Loss')
    plt.plot(range(x_lim), h['loss']    , label = 'Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train','test'], loc='upper right')
    plt.show()
    return

plot_loss(history)

preds_on_trained = model.predict(X_testcnn, batch_size=200, verbose=1)
report = classification_report(y_test, preds_on_trained.round())
print(report)


# Accuracy Score comparison
# --------------------------------------------------------------------
train_score = model.evaluate(X_traincnn, y_train, verbose=0)
val_score   = model.evaluate(X_valcnn, y_val, verbose=0)
test_score  = model.evaluate(X_testcnn, y_test, verbose=0)

result = {'Data Type'      : ['Training Data','Validation Data','Testing Data'],
          'Accuracy Score' : [train_score[1]*100,
                              val_score[1]*100  ,
                              test_score[1]*100 ]}

df_resutl = pd.DataFrame(result, columns = ['Data Type', 'Accuracy Score'])

print('\n -------------------------------------------------------------\n')
print('\n -------------------------------------------------------------\n')
print(' --------- Speech Emotion Detection - Deep Learning ----------')
print('\n -------------------------------------------------------------\n')
print('\n -------------------------------------------------------------\n')
print(df_resutl)





