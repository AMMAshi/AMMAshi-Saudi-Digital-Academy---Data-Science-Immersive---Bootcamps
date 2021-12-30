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
from new_p1_SAVEE_DataFrame import df_Final


# Pacakges
# --------------------------------------------------------------------
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Model Training
from sklearn.model_selection import train_test_split
# Selecting Features
from sklearn.feature_selection import SelectKBest, f_classif
# Deal with Imbalanced Data using SMOTE
from imblearn.over_sampling import SMOTE
# Model Pipeline
from sklearn.pipeline import Pipeline
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
df       = df_Final#[df_Final.emotion != 5]#'neutral']
features = df.drop(['path','emotion','source'],axis=1)
targets  = df['emotion']
# print(features.head())
# print(targets.head())

# Model Training
# --------------------------------------------------------------------
training, testing = train_test_split(df, train_size=0.8, test_size=0.20, stratify=df['emotion'])#,random_state=46)
train, val        = train_test_split(training, train_size=0.8, test_size=0.20, stratify=training['emotion'])#,random_state=46)

X_train_old = train.drop(['path','emotion','source'],axis=1)
y_train_old = train['emotion']

unique, count = np.unique(y_train_old, return_counts=True)
y_train_dict_emotion_count = {k:v for (k,v) in zip(unique, count)}
# print(y_train_dict_emotion_count)

X_val = val.drop(['path','emotion','source'],axis=1)
y_val = val['emotion']

X_test = testing.drop(['path','emotion','source'],axis=1)
y_test = testing['emotion']

# print(y_train.value_counts(normalize=True))
# print(y_val.value_counts(normalize=True))
# print(df.describe(exclude='number'))


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

gridsearch   = GridSearchCV(estimator   = pipeline,
                          param_grid  = search_space,
                          scoring     = 'accuracy'
                          )

gridsearch_s = GridSearchCV(estimator   = pipeline,
                          param_grid  = search_space,
                          scoring     = 'accuracy'
                          )

# using all features
# --------------------------------------------------------------------
best_model = gridsearch.fit(X_train, y_train)
#print('Best accuracy train data : %f using %s'%(best_model.best_score_, best_model.best_params_))

cv_results = gridsearch.cv_results_['mean_test_score']
#print('mean_test_score',cv_results)

# Prediction
# --------------------------
y_pred = gridsearch.predict(X_test)
#print(classification_report(y_test, y_pred))

# Evaluation
# --------------------------
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
print(' --------------------- Using All Features --------------------')
print(' -------- Speech Emotion Detection - Machine Learning --------')
print('\n -------------------------------------------------------------\n')
print('\n -------------------------------------------------------------\n')
print(df_resutl)
print('\n')


# using selected features
# --------------------------------------------------------------------
best_model_s = gridsearch_s.fit(X_train_s, y_train)
#print('Best accuracy train data : %f using %s'%(best_model_s.best_score_, best_model_s.best_params_))

cv_results = gridsearch_s.cv_results_['mean_test_score']
#print('mean_test_score',cv_results)

# Prediction
# --------------------------
y_pred_s = gridsearch_s.predict(X_test_s)
#print(classification_report(y_test, y_pred_s))

# Evaluation
# --------------------------
result_s = {'Data Type'      : ['Training Data','Validation Data','Testing Data'],
          'Accuracy Score' : [best_model_s.best_score_,
                              best_model_s.score(X_val_s, y_val),
                              best_model_s.score(X_test_s, y_test)],
          'Best Classifier': [best_model_s.best_params_,
                              best_model_s.best_params_,
                              best_model_s.best_params_]}

df_resutl_s = pd.DataFrame(result_s, columns = ['Data Type', 'Accuracy Score', 'Best Classifier'])

print('\n -------------------------------------------------------------\n')
print('\n -------------------------------------------------------------\n')
print(' ---------------------- Seleted Features ---------------------')
print(' -------- Speech Emotion Detection - Machine Learning --------')
print('\n -------------------------------------------------------------\n')
print('\n -------------------------------------------------------------\n')
print(df_resutl_s)


