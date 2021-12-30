# ====================================================================
# ====================================================================
# Author: Arwa Ashi
# Saudi Digital Academy 
# Speech Emotion Detection - Final Project - Dec 4th, 2020
# ====================================================================
# ====================================================================


# Speech Emotion Detection Deep Learning Model code
# ====================================================================


# Calling the DataFrame
# --------------------------------------------------------------------
from new_p1_SAVEE_DataFrame import df_Final


# importing pacakges
# --------------------------------------------------------------------
import pandas as pd
import numpy as np
# Selecting Features
from sklearn.feature_selection import SelectKBest, f_classif
# Deal with Imbalanced Data using SMOTE
from imblearn.over_sampling import SMOTE
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
df       = df_Final#[df_Final.emotion != 5]#'neutral']
features = df.drop(['path','emotion','source'],axis=1)
targets  = df['emotion']
#print(features.head())
#print(targets.head())


# Split into traning and test sets
# --------------------------------------------------------------------
training, testing = train_test_split(df, train_size=0.8, test_size=0.20, stratify=df['emotion'])#,random_state=46)
train, val        = train_test_split(training, train_size=0.8, test_size=0.20, stratify=training['emotion'])#,random_state=46)

X_train_old = train.drop(['path','emotion','source'],axis=1)
y_train_old = train['emotion']

unique, count = np.unique(y_train_old, return_counts=True)
y_train_dict_emotion_count = {k:v for (k,v) in zip(unique, count)}
print(y_train_dict_emotion_count)

X_val = val.drop(['path','emotion','source'],axis=1)
y_val = val['emotion']

X_test = testing.drop(['path','emotion','source'],axis=1)
y_test = testing['emotion']

# Selecting Features
# --------------------------------------------------------------------
selector = SelectKBest(f_classif, k=90)
selector.fit(X_train_old, y_train_old)
# print(selector.scores_)

# Top features with the highest F score
cols = selector.get_support(indices=True)
# print(cols)

# Picking subset of traning and testing
X_train_s_old = X_train_old.iloc[:,cols]
X_val_s       = X_val.iloc[:,cols]
X_test_s      = X_test.iloc[:,cols]
# print(X_train_s.head())


# Deal with Imbalanced Data using SMOTE
# ONLY USE the SMOTE in TRAINING Data 
# --------------------------------------------------------------------
sm = SMOTE(random_state=2)#, ratio=1.0)

# using all features
# --------------------------
X_train, y_train   = sm.fit_sample(X_train_old.values,y_train_old)

unique, count = np.unique( y_train, return_counts=True)
y_train_dict_emotion_count = {k:v for (k,v) in zip(unique, count)}
# print(y_train_dict_emotion_count)

# using selected features
# --------------------------
X_train_s, y_train_s = sm.fit_sample(X_train_s_old.values,y_train_old)
unique, count = np.unique( y_train_s, return_counts=True)
y_train_dict_emotion_count = {k:v for (k,v) in zip(unique, count)}
# print(y_train_dict_emotion_count)


# Using .ravel() converting 2D to 1D
# --------------------------------------------------------------------

# using all features
# --------------------------
X_train = np.array(X_train)
X_val   = np.array(X_val)
X_test  = np.array(X_test)

y_train = np.array(y_train).ravel()
y_val   = np.array(y_val).ravel()
y_test  = np.array(y_test).ravel()

print(y_train[:5]) # [4 7 1 1 4]


# using selected features
# --------------------------
X_train_s = np.array(X_train_s)
X_val_s   = np.array(X_val_s)
X_test_s  = np.array(X_test_s)

y_train_s = np.array(y_train_s).ravel()


# One-Hot Encoding
# --------------------------------------------------------------------
lb = LabelEncoder()

# using all features
# --------------------------
y_train = utils.to_categorical(lb.fit_transform(y_train))
y_val   = utils.to_categorical(lb.fit_transform(y_val))
y_test  = utils.to_categorical(lb.fit_transform(y_test))

# 1 means which class has the highest probability
print(y_train)

# using selected features
# --------------------------
y_train_s = utils.to_categorical(lb.fit_transform(y_train_s))


# Changing Dimension for CNN Model
# --------------------------------------------------------------------

# using all features
# --------------------------
X_traincnn = np.expand_dims(X_train, axis=2)
X_valcnn   = np.expand_dims(X_val, axis=2)
X_testcnn  = np.expand_dims(X_test, axis=2)

print(X_testcnn.shape) # (108, 65, 1)

# using selected features
# --------------------------
X_traincnn_s = np.expand_dims(X_train_s, axis=2)
X_valcnn_s   = np.expand_dims(X_val_s, axis=2)
X_testcnn_s  = np.expand_dims(X_test_s, axis=2)

print(X_testcnn_s.shape) # (108, 65, 1)


# CNN Model
# --------------------------------------------------------------------


# using all features
# --------------------------
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
    validation_data = (X_valcnn, y_val),
    epochs          = 100,
    callbacks       = [es_cb]
    )


# using selected features
# --------------------------
def get_model_s():
    model = Sequential([
        # 1st layer
        Conv1D(256, 5, padding='same',input_shape=(X_traincnn_s.shape[1],X_traincnn_s.shape[2])),
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
        Dense(y_train_s.shape[1]),
        Activation('softmax'), 
    ])
    opt = tensorflow.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    model.compile(
        loss      = 'categorical_crossentropy',#'categorical_crossentropy',#'mse',#binary_crossentropy,
        optimizer = opt,#'adam', # opt,
        metrics   = ['accuracy']
    )
    return model

get_model_s().summary()

es_cb     = EarlyStopping(monitor='val_loss', patience=5)
model_s   = get_model_s()
history_s = model_s.fit(
    X_traincnn_s, y_train_s,
    batch_size      = 16,
    validation_data = (X_valcnn_s, y_val),
    epochs          = 100,
    callbacks       = [es_cb]
    )


# Accuracy Score comparison
# --------------------------------------------------------------------

# using all features
# --------------------------
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

# using selected features
# --------------------------
train_score_s = model_s.evaluate(X_traincnn_s, y_train_s, verbose=0)
val_score_s   = model_s.evaluate(X_valcnn_s, y_val, verbose=0)
test_score_s  = model_s.evaluate(X_testcnn_s, y_test, verbose=0)

result_s = {'Data Type'      : ['Training Data','Validation Data','Testing Data'],
          'Accuracy Score' : [train_score_s[1]*100,
                              val_score_s[1]*100  ,
                              test_score_s[1]*100 ]}

df_resutl_s = pd.DataFrame(result_s, columns = ['Data Type', 'Accuracy Score'])

print('\n -------------------------------------------------------------\n')
print('\n -------------------------------------------------------------\n')
print(' --------- Speech Emotion Detection - Deep Learning ----------')
print('\n -------------------------------------------------------------\n')
print('\n -------------------------------------------------------------\n')
print(df_resutl_s)


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

# using all features
# --------------------------
plot_loss(history)

# using selected features
# --------------------------
plot_loss(history_s)


preds_on_trained = model.predict(X_testcnn, batch_size=200, verbose=1)
report = classification_report(y_test, preds_on_trained.round())
print(report)





