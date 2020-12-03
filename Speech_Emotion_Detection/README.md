
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
1- parse the filename to get the emotions.

2- Extraction 65 features for each audio by using mfccs, pitches, magnitudes, and C. where:

For mfccs: 13 feature 

For pitches:  taking only the 20 first value of the audio

For magnitudes:  taking only the 20 first value of the audio

For C: 12 feature 

# 3- EDA Exploratory data analysis
![Test Image 4](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/EDA.png)
1- Comparing emotion waves. 

2- Exploring total waves for each emotion.

# 4- Splitting the data frame
Dividing the dataset into 3 dataset Training, Validation and Testing. 

# 5- Creating the models
![Test Image 5](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/result.png)
# Machine Learning: 
1- Using pipeline.

2- GridSearchCV with search_space = [{'clf':[PassiveAggressiveClassifier()]},
                {'clf':[LogisticRegression()]         },
                {'clf':[MultinomialNB()]              },
                {'clf':[BernoulliNB()]                },
                {'clf':[RidgeClassifier()]            },
                {'clf':[RandomForestClassifier()]     },
                {'clf':[SGDClassifier()]              },
                {'clf':[GradientBoostingClassifier()] },
                {'clf':[MLPClassifier()]              },
                {'clf':[Perceptron()]                 }
                ]
                
3- Evaluation 

4- Result 

a) The best model randomly selected are GradientBoostingClassifier, RidgeClassifier, and 
RandomForestClassifier.

b) The Accuracy Score range by using Training, Validation, and Testing Data is 44% up to 68.05%.

# Deep Learning 
1- using .ravel() converting 2D to 1D.

2- One-Hot Encoding >>  LabelEncoder() <<

3- Changing Dimension for CNN Model

4- CNN Model

5- Evaluation 

6- Result: The Accuracy Score range by using Training, Validation, and Testing Data is 52.78% up to 86.97%. 

# 6- Validating the model
By using classification report and visualization 

# 7- Testing the models’ performance
![Test Image 6](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/test.png)
using a new angry wave to test the models’ performance:

1- For The Machine Learning Model: the model predicted that the wave was an angry wave.

2- For The Deep Learning Model: the model predicted that the wave was 99% as an angry wave. 
