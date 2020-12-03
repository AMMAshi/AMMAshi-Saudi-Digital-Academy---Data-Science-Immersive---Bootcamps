# ====================================================================
# ====================================================================
# Author: Arwa Ashi
# Saudi Digital Academy 
# Speech Emotion Detection - Final Project - Dec 3rd, 2020
# ====================================================================
# ====================================================================

# Speech Emotion Detection Exploratory Data Analysis code
# ====================================================================

# Calling The DataFrame
# --------------------------------------------------------------------
from p1_SAVEE_DataFrame import df_Final


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


# Exploring the data
# --------------------------------------------------------------------

# Total Waves For Each Emotion
# --------------------------------------------------------------------
plt.figure(figsize=(10,8))
sns.countplot(df_Final['emotion'])
plt.title('Total Waves For Each Emotion')
plt.show()


# Exploring DC Waves
# --------------------------------------------------------------------
SAVEE_DC = "SAVEE_AudioData/DC/"


# Fear Wave
# --------------------------------------------------------------------
# Calling Fear Wave
fname = SAVEE_DC + 'f11.wav'
data, sample_rate        = librosa.load(fname)
y_harmonic, y_percussive = librosa.effects.hpss(data)
pitches, magnitudes      = librosa.core.pitch.piptrack(y=data, sr=sample_rate)

# Fear Wave feature
mfcc       = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)    
pitches    = pitches
magnitudes = magnitudes
C          = librosa.feature.chroma_cqt(y=y_harmonic, sr=44100)  

# Fear Wave Visualization
librosa.display.waveplot(data, sr=sample_rate)
plt.title("Fear Wave")
plt.show()

librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()
plt.title("Fear Wave")
plt.show()

librosa.display.specshow(pitches, x_axis='time')
plt.ylabel('Pitches')
plt.colorbar()
plt.title("Fear Wave")
plt.show()

librosa.display.specshow(magnitudes, x_axis='time')
plt.ylabel('Magnitudes')
plt.colorbar()
plt.title("Fear Wave")
plt.show()

librosa.display.specshow(C, x_axis='time')
plt.ylabel('C')
plt.colorbar()
plt.title("Fear Wave")
plt.show()


# Sad Wave
# --------------------------------------------------------------------
# Calling Sad Wave
sname = SAVEE_DC + 'sa11.wav'
data, sample_rate        = librosa.load(sname)
y_harmonic, y_percussive = librosa.effects.hpss(data)
pitches, magnitudes      = librosa.core.pitch.piptrack(y=data, sr=sample_rate)

# Fear Sad feature
mfcc       = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)    
pitches    = pitches
magnitudes = magnitudes
C          = librosa.feature.chroma_cqt(y=y_harmonic, sr=44100)  

# Fear Sad Visualization
librosa.display.waveplot(data, sr=sample_rate)
plt.title("Sad Wave")
plt.show()

librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()
plt.title("Sad Wave")
plt.show()

librosa.display.specshow(pitches, x_axis='time')
plt.ylabel('Pitches')
plt.colorbar()
plt.title("Sad Wave")
plt.show()

librosa.display.specshow(magnitudes, x_axis='time')
plt.ylabel('Magnitudes')
plt.colorbar()
plt.title("Sad Wave")
plt.show()

librosa.display.specshow(C, x_axis='time')
plt.ylabel('C')
plt.colorbar()
plt.title("Sad Wave")
plt.show()


# Angry Wave
# --------------------------------------------------------------------
# Calling Angry Wave
aname = SAVEE_DC + 'a11.wav'
data, sample_rate        = librosa.load(aname)
y_harmonic, y_percussive = librosa.effects.hpss(data)
pitches, magnitudes      = librosa.core.pitch.piptrack(y=data, sr=sample_rate)

# Angry Wave feature
mfcc       = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)    
pitches    = pitches
magnitudes = magnitudes
C          = librosa.feature.chroma_cqt(y=y_harmonic, sr=44100)  

# Angry Wave Visualization
librosa.display.waveplot(data, sr=sample_rate)
plt.title("Angry Wave")
plt.show()

librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()
plt.title("Angry Wave")
plt.show()

librosa.display.specshow(pitches, x_axis='time')
plt.ylabel('Pitches')
plt.colorbar()
plt.title("Angry Wave")
plt.show()

librosa.display.specshow(magnitudes, x_axis='time')
plt.ylabel('Magnitudes')
plt.colorbar()
plt.title("Angry Wave")
plt.show()

librosa.display.specshow(C, x_axis='time')
plt.ylabel('C')
plt.colorbar()
plt.title("Angry Wave")
plt.show()


