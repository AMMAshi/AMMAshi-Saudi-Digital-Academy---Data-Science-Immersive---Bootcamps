
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

# 5- Creating and evaluating the models
![Test Image 5](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/result.png)
# Machine Learning: 
The following are the steps that I used to create and evaluate the Machine Learning model:

1- Creating a pipeline.

2- Passing 10 classifiers into the pipeline by using GridSearchCV with search_space that contains: Passive Aggressive Classifier, Logistic Regression, Multinomial NB, Bernoulli NB, Ridge Classifier, Random Forest Classifier, SGD Classifier, Gradient Boosting Classifier, MLP Classifier, Perceptron; to find the best classifier with the highest accuracy score to fit the training data frame.
  
3- Using the result of GridSearchCV for the prediction then generating classification report.

4- Evaluating the model’s performance by using confusion matrix and heatmap.

5- Result
 
a) The best classifiers that were selected based on its highest accuracy score are Gradient Boosting Classifier, Ridge Classifier, and Random Forest Classifier.

b) The model accuracy score range by using Training, Validation, and Testing Data frame is from 44% up to 68.05%.


# Deep Learning 
The following are the steps that I used to create and evaluate the Deep Learning model:

1- Using .ravel() to convert the y (target or emotion column) data frame from two dimension to one dimension.

2- Applying One-Hot Encoding >> LabelEncoder() << into the y (target or emotion column) data frame to get array that contain zeros and  has a mean equals to 1 to which emotion that row contains.

3- Changing features dimension into one for Conv1D, (convolve) along one dimensions, in a convolutional neural network CNN model. 

4- Creating CNN model that contain three layers. Where loss = 'categorical_crossentropy', optimizer = tensorflow.optimizers.RMSprop(lr=0.00001, decay=1e-6), and metrics = ['accuracy'].

5- Using the model result for the prediction then generating classification report.

6- Evaluating by plotting model history epochs and loss.

7- Result: the accuracy score range by using Training, Validation, and Testing Data frame is from 52.78% up to 86.97%. 

# 6- Testing the models’ performance
![Test Image 6](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/test.png)
using a new angry wave to test the models’ performance:

1- For The Machine Learning Model: the model predicted that the wave was an angry wave.

2- For The Deep Learning Model: the model predicted that the wave was 99% as an angry wave. 
