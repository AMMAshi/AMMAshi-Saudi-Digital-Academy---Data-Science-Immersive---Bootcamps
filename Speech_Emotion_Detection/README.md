
# Author: Arwa Ashi

# Saudi Digital Academy 

# Speech Emotion Detection - Final Project - Dec 3rd, 2020
# -------------------------------------------------------------------

# Objective
As a Data Scientist, I have been tasked with designing a model that detects speech emotions such as anger, disgust, fear, happiness, neutral, sadness and surprise. 

# Project outline: 
![Test Image 1](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/Project_outline.png)

# Code
The code is divided into 5 .py files:

p1_SAVEE_DataFrame.py 

p2_SAVEE_Exploratory_data_analysis.py 

p3_SAVEE_Baseline_model_01_DeepLearning.py

p4_SAVEE_Baseline_model_02_MachineLearning.py

p5_SAVEE_Baseline_models_test.py
# -------------------------------------------------------------------

# 1- Finding dataset
![Test Image 2](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/DataFrame.png)
Using SAVEE Database, speech emotion annotated data for emotion recognition systems, from
https://www.kaggle.com/barelydedicated/savee-database.

# 2-  Extraction 65 features for each audio
![Test Image 3](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/features.png)
1- parse the filename to get the emotions
2- Extraction 65 features for each audio by using mfccs, pitches, magnitudes, and C. where:

For mfccs: 13 feature 

For pitches:  taking only the 20 first value of the audio

For magnitudes:  taking only the 20 first value of the audio

For C: 12 feature 
