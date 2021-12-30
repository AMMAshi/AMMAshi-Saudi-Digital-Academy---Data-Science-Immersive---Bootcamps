
# Author: Arwa Ashi

# Saudi Digital Academy 

# Speech Emotion Detection - Final Project - Dec 3rd, 2020
# -------------------------------------------------------------------

# Objective
As a Data Scientist, I have been tasked with designing a model that detects speech emotions such as anger, disgust, fear, happiness, neutral, sadness and surprise. The project can add more value and support in a decision making that is related to psychiatrists understanding their patients' emotions and improving analyzing the diverse cases, or a customer service call center to improve customer experience and journey.

# Project outline: 
![Test Image 1](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/Project_outline.png)

# Code
The code is divided into 5 .py files to organize my work.

- [p1_SAVEE_DataFrame.py](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/p1_SAVEE_DataFrame.py)

This code where the project DataFrame was created and prepared.

- [p2_SAVEE_Exploratory_data_analysis.py](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/p2_SAVEE_Exploratory_data_analysis.py) 

This code where the DataFrame exploration was done.  

- [p3_SAVEE_Baseline_model_01_DeepLearning.py](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/p3_SAVEE_Baseline_model_01_DeepLearning.py)

This code where the deep learning model was created.

- [p4_SAVEE_Baseline_model_02_MachineLearning.py](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/p4_SAVEE_Baseline_model_02_MachineLearning.py)

This code where the machine learning model was created.

- [p5_SAVEE_Baseline_models_test.py](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/p5_SAVEE_Baseline_models_test.py)

The code where the models' performance evaluation was done by interducing unknown audio.

# -------------------------------------------------------------------

# 1- Finding dataset
![Test Image 2](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/DataFrame.png)
Using SAVEE Database, speech emotion annotated data for emotion recognition systems, from
https://www.kaggle.com/barelydedicated/savee-database.

# 2-  Extracting 65 features for each audio
![Test Image 3](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/new_features.png)

1- Reading audio waves by using [Librosa](https://librosa.org/doc/latest/index.html) which is a python package for music and audio analysis. It provides the building blocks necessary to create audio information retrieval systems.

2- Parsing the filename to get the emotions.

3- Creating a data frame with 3 columns (Emotion, Source, Path). 

4- Extracting 65 features for each audio by using [mfccs](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), [pitches](https://en.wikipedia.org/wiki/Pitch_(music)), magnitudes, and [C](https://en.wikipedia.org/wiki/Chroma_feature). and adding them as columns to the data frame where:

- mfccs: it gives 13 feature for each audio. 

- pitches, taking only the 20 first value of the audio.

- magnitudes: taking only the 20 first value of the audio.

- C: it gives 12 feature for each audio.


# 3- EDA Exploratory data analysis

[Exploratory data analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis) step is to identify missing data, explore data type, visualize the data before applying any analysis to it, look at data distribution, mean and standard deviation, and etc.

![Test Image 4a](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/EDA_02.png)
1- Comparing emotion waves: the graph shows a comparison between anger, disgust, fear, happiness, neutral, sadness and surprise waves. Each emotion has its length of time and unique wave. 

![Test Image 4b](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/EDA_03.png)
2- Exploring total waves for each emotion: the neutral emotion has 120 waves in dataset and 60 waves for each other emotions.


# 4- Splitting the data frame
Dividing the dataset into 3 data frame Training, Validation and Testing. The Validation data frame provided an unbiased evaluation of a model fit on the training data frame. 

# 5- Creating and evaluating the models
![Test Image 5](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/result.png)
# Machine Learning: 
The following are the steps that I used to create and evaluate the Machine Learning model:

1- Creating a pipeline.

2- Passing 10 classifiers into the pipeline by using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) with search_space that contains: Passive Aggressive Classifier, Logistic Regression, Multinomial NB, Bernoulli NB, Ridge Classifier, Random Forest Classifier, SGD Classifier, Gradient Boosting Classifier, MLP Classifier, Perceptron; to find the best classifier with the highest accuracy score to fit the training data frame.
  
3- Using the result of GridSearchCV for the prediction then generating classification report.

4- Evaluating the model’s performance by using confusion matrix and heatmap.

5- Result:
 
a) The best classifiers that were selected based on its highest accuracy score are Gradient Boosting Classifier, Ridge Classifier, and Random Forest Classifier.

b) The model accuracy score range by using Training, Validation, and Testing Data frame is from 44% up to 68.05%.


# Deep Learning 
The following are the steps that I used to create and evaluate the Deep Learning model:

1- Using .ravel() to convert the y (target or emotion column) data frame from two dimension to one dimension.

2- Applying One-Hot Encoding >> LabelEncoder() << into the y (target or emotion column) data frame to get array that contains zeros and  has a mean equals to 1 to which emotion that row has.

3- Changing features dimension into one for Conv1D, (convolve) along one dimensions, in a convolutional neural network CNN model. 

4- Creating CNN model where loss = 'categorical_crossentropy', optimizer = tensorflow.optimizers.RMSprop(lr=0.00001, decay=1e-6), and metrics = ['accuracy'].

5- Using the model result for the prediction then generating classification report.

6- Evaluating by plotting model history epochs and loss.

7- Result: the accuracy score range by using Training, Validation, and Testing Data frame is from 52.78% up to 86.97%. 

# 6- Testing the models’ performance
![Test Image 6](https://github.com/AMMAshi/AMMAshi-Saudi-Digital-Academy---Data-Science-Immersive---Bootcamps/blob/master/Speech_Emotion_Detection/images/test.png)
Using a new angry wave from CREMA dataset to test the models’ performance:

1- For The Machine Learning Model: the model predicted that the wave was an angry wave.

2- For The Deep Learning Model: the model predicted that the wave was 98% as an angry wave.
 
# 7- Future work
