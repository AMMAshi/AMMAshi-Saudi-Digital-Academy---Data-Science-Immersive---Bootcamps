# ====================================================================
# ====================================================================
# Author: Arwa Ashi
# Saudi Digital Academy 
# Speech Emotion Detection - Final Project - Dec 4th, 2020
# ====================================================================
# ====================================================================


# Speech Emotion Detection Testing Models' Performance code
# ====================================================================


# Calling the models
# --------------------------------------------------------------------
#from p3_SAVEE_Baseline_model_01_DeepLearning import model
#from p4_SAVEE_Baseline_model_02_MachineLearning import gridsearch


# Pacakges
# --------------------------------------------------------------------
# Packages to read the audioes and its features
import librosa
import librosa.display
# Packages for visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from sklearn.metrics import confusion_matrix
import os
import sys
import glob 
import warnings

# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# Load the dataset
# --------------------------------------------------------------------
test_angry  = 'Test_angry/'
dir_list    = os.listdir(test_angry)

emotion = []
path    = []

for i in dir_list:
    if i[-8:-6]=='a':
        emotion.append(1)#'angry')
    path.append(test_angry + i)

# Now check out the label count distribution 
df           = pd.DataFrame(emotion, columns = ['emotion'])
df['source'] = 'SAVEE'
df           = pd.concat([df, pd.DataFrame(path, columns = ['path'])], axis = 1)
# print(df)


# Feature
# --------------------------------------------------------------------
df1 = pd.DataFrame(columns=['feature'])
df2 = pd.DataFrame(columns=['feature'])
df3 = pd.DataFrame(columns=['feature'])
df4 = pd.DataFrame(columns=['feature'])


# Extraction 65 features for each audio
# --------------------------------------------------------------------
counter = 0
for index,path in enumerate(df.path):
    X, sample_rate = librosa.load(path
                                  ,res_type = 'kaiser_fast'
                                  ,duration = 2.5
                                  ,sr       = 44100 
                                  ,offset   = 0.5
                                 )
    sample_rate = np.array(sample_rate)

    # Added https://www.youtube.com/watch?v=yvxpxcncSGs
    y_harmonic, y_percussive = librosa.effects.hpss(X)
    pitches, magnitudes      = librosa.core.pitch.piptrack(y=X, sr=sample_rate)
    
    # The feature
    # ------------------------------------------------------
    # mfccs has 30 feature 
    mfccs      = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=1) # <<< axis = 1 Original axis = 0
    # taking only the 20 first value of the audio
    pitches    = np.trim_zeros(np.mean(pitches,axis=1))[:20]
    # taking only the 20 first value of the audio
    magnitudes = np.trim_zeros(np.mean(magnitudes, axis=1))[:20]
    # C has 12 feature 
    C          = np.mean(librosa.feature.chroma_cqt(y=y_harmonic, sr=44100), axis=1)  

    df1.loc[counter] = [mfccs]
    df2.loc[counter] = [pitches]
    df3.loc[counter] = [magnitudes]
    df4.loc[counter] = [C]
    counter          = counter + 1

# Creating The Final Features DataFrame
# --------------------------------------------------------------------
df5 = pd.concat([df ,pd.DataFrame(df1['feature'].values.tolist())],axis=1)
df6 = pd.concat([df5,pd.DataFrame(df2['feature'].values.tolist())],axis=1)
df7 = pd.concat([df6,pd.DataFrame(df3['feature'].values.tolist())],axis=1)
df8 = pd.concat([df7,pd.DataFrame(df4['feature'].values.tolist())],axis=1)
#print(df8[:5])


# Replacing NAs with 0
# --------------------------------------------------------------------
df_Final = df8.fillna(0)
#print(df_Final.head())

X_test     = df_Final.drop(['path','emotion','source'],axis=1)
X_test_arr = np.array(X_test)
X_testcnn  = np.expand_dims(X_test_arr, axis=2)


# testing the models 
# --------------------------------------------------------------------
# testing Deep Learning model
#test_wav_DL = model.predict(X_testcnn, batch_size = 32, verbose=1)
#print('\n-------------------------------------------------------\n')
#print('Testing the angry wave result')
#print('\n-------------------------------------------------------\n')
#print(test_wav_DL)
#print('\n')
#print('[angry   , disgust, fear  , happy   , sad    , surprise]')
'''
              precision    recall  f1-score   support

           0       0.67      0.33      0.44        12
           1       0.50      0.33      0.40        12
           2       0.50      0.17      0.25        12
           3       0.20      0.08      0.12        12
           4       0.67      0.33      0.44        12
           5       0.20      0.08      0.12        12

   micro avg       0.47      0.22      0.30        72
   macro avg       0.46      0.22      0.30        72
weighted avg       0.46      0.22      0.30        72
 samples avg       0.22      0.22      0.22        72


 -------------------------------------------------------------


 -------------------------------------------------------------

 --------- Speech Emotion Detection - Deep Learning ----------

 -------------------------------------------------------------


 -------------------------------------------------------------

         Data Type  Accuracy Score
0    Training Data       86.956519
1  Validation Data       56.896549
2     Testing Data       45.833334
1/1 [==============================] - ETA: 0s1/1 [==============================] - 0s 104ms/step

-------------------------------------------------------

Testing the angry wave result

-------------------------------------------------------

[[9.8705494e-01 3.9408444e-03 5.5059744e-03 1.4425812e-03 1.2239718e-04
  1.9332444e-03]]


[angry   , disgust, fear  , happy   , sad    , surprise]
'''

# testing Machine Learning model
#test_wav_ML = gridsearch.predict(X_test)
#print('\n-------------------------------------------------------\n')
#print('Testing the angry wave result')
#print('\n-------------------------------------------------------\n')
#print(test_wav_ML)

'''
Best accuracy train data : 0.526087 using {'clf': RandomForestClassifier()}
              precision    recall  f1-score   support

           1       0.73      0.67      0.70        12
           2       0.83      0.83      0.83        12
           3       0.44      0.67      0.53        12
           4       0.60      0.50      0.55        12
           6       1.00      0.92      0.96        12
           7       0.60      0.50      0.55        12

    accuracy                           0.68        72
   macro avg       0.70      0.68      0.68        72
weighted avg       0.70      0.68      0.68        72


 -------------------------------------------------------------


 -------------------------------------------------------------

 -------- Speech Emotion Detection - Machine Learning --------

 -------------------------------------------------------------


 -------------------------------------------------------------

         Data Type  Accuracy Score                    Best Classifier
0    Training Data        0.526087  {'clf': RandomForestClassifier()}
1  Validation Data        0.603448  {'clf': RandomForestClassifier()}
2     Testing Data        0.680556  {'clf': RandomForestClassifier()}

-------------------------------------------------------

Testing the angry wave result

-------------------------------------------------------

[1] = 'angry'

'''

# ['angry'   , 'disgust', 'fear'  , 'happy'   , 'sad'    , 'surprise']












