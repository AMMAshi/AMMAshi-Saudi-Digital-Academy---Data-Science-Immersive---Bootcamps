# ====================================================================
# ====================================================================
# Author: Arwa Ashi
# Saudi Digital Academy 
# Speech Emotion Detection - Final Project - Dec 3rd, 2020
# ====================================================================
# ====================================================================


# Creating Speech Emotion Detection DataFrame code
# ====================================================================


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
# SAVEE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# https://www.kaggle.com/barelydedicated/savee-database
# --------------------------------------------------------------------
SAVEE_DC = "SAVEE_AudioData/DC/"
SAVEE_JE = "SAVEE_AudioData/JE/"
SAVEE_JK = "SAVEE_AudioData/JK/"
SAVEE_KL = "SAVEE_AudioData/KL/"

# Run DC example 
dir_list_DC = os.listdir(SAVEE_DC)
#print(dir_list_DC[0:5])

# Run JE example 
dir_list_JE = os.listdir(SAVEE_JE)
#print(dir_list_JE[0:5])

# Run JK example 
dir_list_JK = os.listdir(SAVEE_JK)
#print(dir_list_JK[0:5])

# Run KL example 
dir_list_KL = os.listdir(SAVEE_KL)
#print(dir_list_KL[0:5])


# parse the filename to get the emotions
# --------------------------------------------------------------------
emotion = []
path    = []

for i in dir_list_DC:
    if i[-8:-6]=='a':
        emotion.append(1)#'angry')
    elif i[-8:-6]=='d':
        emotion.append(2)#'disgust')
    elif i[-8:-6]=='f':
        emotion.append(3)#'fear')
    elif i[-8:-6]=='h':
        emotion.append(4)#'happy')
    elif i[-8:-6]=='n':
        emotion.append(5)#'neutral')
    elif i[-8:-6]=='sa':
        emotion.append(6)#'sad')
    elif i[-8:-6]=='su':
        emotion.append(7)#'surprise')
    else:
        emotion.append('error') 
    path.append(SAVEE_DC + i)

for i in dir_list_JE:
    if i[-8:-6]=='a':
        emotion.append(1)#'angry')
    elif i[-8:-6]=='d':
        emotion.append(2)#'disgust')
    elif i[-8:-6]=='f':
        emotion.append(3)#'fear')
    elif i[-8:-6]=='h':
        emotion.append(4)#'happy')
    elif i[-8:-6]=='n':
        emotion.append(5)#'neutral')
    elif i[-8:-6]=='sa':
        emotion.append(6)#'sad')
    elif i[-8:-6]=='su':
        emotion.append(7)#'surprise')
    else:
        emotion.append('error') 
    path.append(SAVEE_JE + i)

for i in dir_list_JK:
    if i[-8:-6]=='a':
        emotion.append(1)#'angry')
    elif i[-8:-6]=='d':
        emotion.append(2)#'disgust')
    elif i[-8:-6]=='f':
        emotion.append(3)#'fear')
    elif i[-8:-6]=='h':
        emotion.append(4)#'happy')
    elif i[-8:-6]=='n':
        emotion.append(5)#'neutral')
    elif i[-8:-6]=='sa':
        emotion.append(6)#'sad')
    elif i[-8:-6]=='su':
        emotion.append(7)#'surprise')
    else:
        emotion.append('error') 
    path.append(SAVEE_JK + i)

for i in dir_list_KL:
    if i[-8:-6]=='a':
        emotion.append(1)#'angry')
    elif i[-8:-6]=='d':
        emotion.append(2)#'disgust')
    elif i[-8:-6]=='f':
        emotion.append(3)#'fear')
    elif i[-8:-6]=='h':
        emotion.append(4)#'happy')
    elif i[-8:-6]=='n':
        emotion.append(5)#'neutral')
    elif i[-8:-6]=='sa':
        emotion.append(6)#'sad')
    elif i[-8:-6]=='su':
        emotion.append(7)#'surprise')
    else:
        emotion.append('error') 
    path.append(SAVEE_KL + i)

# Now check out the label count distribution 
df           = pd.DataFrame(emotion, columns = ['emotion'])
df['source'] = 'SAVEE'
df           = pd.concat([df, pd.DataFrame(path, columns = ['path'])], axis = 1)
#print(df.emotion.value_counts())
#print(df.head())


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
# print(df8[:5])


# Replacing NAs with 0
# --------------------------------------------------------------------
df_Final = df8.fillna(0)
# print(df.shape)
# print(df_Final[:5])




