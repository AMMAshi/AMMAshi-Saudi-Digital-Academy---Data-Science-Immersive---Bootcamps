
# Author: Arwa Ashi

# Saudi Digital Academy 

# Speech Emotion Detection - Final Project - Dec 3rd, 2020
# -------------------------------------------------------------------

# Objective
As a Data Scientist, I have been tasked with designing a model that detects speech emotions such as anger, disgust, fear, happiness, neutral, sadness and surprise. 

# Project outline: 
![Test Image 1](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/Project_outline.png)

# Code
The code is divided into 5 .py files to organize my work.

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

# 2-  Extracting 65 features for each audio
![Test Image 3](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/features.png)
1- Parsing the filename to get the emotions.

2- Creating a data frame that contains has 3 columns (emotion, source, and path). 

3- Extracting 65 features for each audio by using mfccs, pitches, magnitudes, and C. where:

For mfccs: it gives 13 feature for each audio. 

For pitches, taking only the 20 first value of the audio.

For magnitudes: taking only the 20 first value of the audio.

For C: it gives 12 feature for each audio.


# 3- EDA Exploratory data analysis
![Test Image 4](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/EDA.png)
1- Comparing emotion waves: the graph shows a comparison between angry, fear, and sad waves. Each emotion has its unique wave. 

2- Exploring total waves for each emotion: the neutral emotion has 120 waves in dataset and 60 waves for each other emotions.


# 4- Splitting the data frame
Dividing the dataset into 3 data frame Training, Validation and Testing. The Validation data frame provided an unbiased evaluation of a model fit on the training data frame. 

# 5- Creating the models
![Test Image 5](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/result.png)
# Machine Learning: 
The following are the step that I used to create the model:

1- Creating a pipeline.

2- Passing 10 classifiers into the pipeline by using GridSearchCV with search_space that contains: Passive Aggressive Classifier, Logistic Regression, Multinomial NB, Bernoulli NB, Ridge Classifier, Random Forest Classifier, SGD Classifier, Gradient Boosting Classifier, MLP Classifier, Perceptron; to find the best classifier with the highest accuracy score to fit the training data frame.
  
3- Using the result of GridSearchCV for the prediction then generating classification report.

4- Evaluation the model’s performance by using confusion matrix and heatmap.

5- Result
 
a) The best classifiers that were selected based on its highest accuracy score are Gradient Boosting Classifier, Ridge Classifier, and Random Forest Classifier.

b) The model accuracy score range by using Training, Validation, and Testing Data frame is 44% up to 68.05%.


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
