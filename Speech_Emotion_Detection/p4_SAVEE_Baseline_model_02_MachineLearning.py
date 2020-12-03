# ====================================================================
# ====================================================================
# Author: Arwa Ashi
# Saudi Digital Academy 
# Speech Emotion Detection - Final Project - Dec 4th, 2020
# ====================================================================
# ====================================================================


# Speech Emotion Detection Machine Learning Model code
# ====================================================================


# Calling the Data
# --------------------------------------------------------------------
from p1_SAVEE_DataFrame import df_Final


# Pacakges
# --------------------------------------------------------------------
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Model Training
from sklearn.model_selection import train_test_split
# Model Pipeline
from sklearn.pipeline import Pipeline
#from sklearn.pipeline import make_pipeline
#from sklearn.impute import SimpleImputer
#import category_encoders as ce
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.impute import SimpleImputer
#from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
# Pipline Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.linear_model import Perceptron
# from sklearn import svm
# Evaluation
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import cross_val_score


# DataFrame
# --------------------------------------------------------------------
df       = df_Final[df_Final.emotion != 5]#'neutral']
features = df.drop(['path','emotion','source'],axis=1)
targets  = df['emotion']
# print(features.head())
# print(targets.head())


# Model Training
# --------------------------------------------------------------------
training, testing = train_test_split(df, train_size=0.8, test_size=0.20, stratify=df['emotion'])
train, val        = train_test_split(training, train_size=0.8, test_size=0.20, stratify=training['emotion'])

X_train = train.drop(['path','emotion','source'],axis=1)
y_train = train['emotion']

X_val = val.drop(['path','emotion','source'],axis=1)
y_val = val['emotion']

X_test = testing.drop(['path','emotion','source'],axis=1)
y_test = testing['emotion']

# print(y_train.value_counts(normalize=True))
# print(y_val.value_counts(normalize=True))
# print(df.describe(exclude='number'))


# Model Pipeline
# --------------------------------------------------------------------
pipeline = Pipeline(steps = [('clf', PassiveAggressiveClassifier())])

search_space = [{'clf':[PassiveAggressiveClassifier()]},
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

gridsearch = GridSearchCV(estimator   = pipeline,
                          param_grid  = search_space,
                          scoring     = 'accuracy'
                          )

best_model = gridsearch.fit(X_train, y_train)
cv_results = gridsearch.cv_results_['mean_test_score']
print('mean_test_score',cv_results)
print('Best accuracy train data : %f using %s'%(best_model.best_score_, best_model.best_params_))


# Prediction
# --------------------------------------------------------------------
y_pred = gridsearch.predict(X_test)
print(classification_report(y_test, y_pred))


# Evaluation
# --------------------------------------------------------------------
# labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
plot_confusion_matrix(best_model, X_test, y_test,display_labels=labels)
plt.show()

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index =labels, columns = labels)
plt.figure(figsize = (10, 8))
sns.heatmap(cm, annot = True, cbar = False, fmt = 'g')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()

result = {'Data Type'      : ['Training Data','Validation Data','Testing Data'],
          'Accuracy Score' : [best_model.best_score_,
                              best_model.score(X_val, y_val),
                              best_model.score(X_test, y_test)],
          'Best Classifier': [best_model.best_params_,
                              best_model.best_params_,
                              best_model.best_params_]}

df_resutl = pd.DataFrame(result, columns = ['Data Type', 'Accuracy Score', 'Best Classifier'])

print('\n -------------------------------------------------------------\n')
print('\n -------------------------------------------------------------\n')
print(' -------- Speech Emotion Detection - Machine Learning --------')
print('\n -------------------------------------------------------------\n')
print('\n -------------------------------------------------------------\n')
print(df_resutl)

