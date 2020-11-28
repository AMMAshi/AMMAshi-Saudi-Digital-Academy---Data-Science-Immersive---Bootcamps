

# We'll be using a text classificaton model on a dataset which contains
# real info on what corporations actually talk about on social media.

# The statements were labelled as into following categories
# 1- information (objective statements about the company or it's activities),
# 2- dialog (replies to users, etc.), or
# 3- action (messages that ask for votes or ask users to click on links, etc.).

# Our aim is to build a model to automatically categorize the text into
# their respective categories.


# Packages
# -----------------------------------------------------------------------------
# Task 1: Understanding and loading the dataset
import numpy as np
import pandas as pd
# Task 2: Text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Task 3: EDA
import seaborn as sns
import sys
import matplotlib.pyplot as plt
# Task 4: Creating custom transformers
from sklearn.base import BaseEstimator, TransformerMixin
# Task 5: Model Building using FeatureUnion
from sklearn.pipeline import Pipeline, FeatureUnion # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Pipeline
from sklearn.model_selection import GridSearchCV    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< improving Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
# Task 6: Model Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Task 1: Understanding and loading the dataset
# -----------------------------------------------------------------------------
data = pd. read_csv('https://raw.githubusercontent.com/daniel-dc-cd/data_science/master/daily_materials/NLP_pipeline_activity/corporate_messaging_dfe.csv')
# print(data.head())
# print(data.shape) # (3118, 11)
# print(data.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3118 entries, 0 to 3117
Data columns (total 11 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   unit_id              3118 non-null   int64  
 1   golden               3118 non-null   bool   
 2   unit_state           3118 non-null   object 
 3   trusted_judgments    3118 non-null   int64  
 4   last_judgment_at     2811 non-null   object 
 5   category             3118 non-null   object 
 6   category_confidence  3118 non-null   float64
 7   category_gold        307 non-null    object 
 8   id                   3118 non-null   int64  
 9   screenname           3118 non-null   object 
 10  text                 3118 non-null   object 
dtypes: bool(1), float64(1), int64(3), object(6)
memory usage: 246.8+ KB
None
'''

# check distribution of target column i.e. category
# print(data['category'].value_counts())
'''
Information    2129
Action          724
Dialogue        226
Exclude          39
Name: category, dtype: int64
'''

# check distribution of column - category_confidence
# print(data['category_confidence'].value_counts())
'''
1.0000    2430
0.6614      35
0.6643      33
0.6747      32
0.6775      29
          ... 
0.8547       1
0.6641       1
0.8578       1
0.9089       1
0.8245       1
Name: category_confidence, Length: 194, dtype: int64
'''

# Remove those observation where category_confidence < 1 and category = Exclude
# The data majority lay on other variables
data = data[(data['category_confidence']==1) & (data['category']!='Exclude')]
# print(data['category'].value_counts())
# print(data['category_confidence'].value_counts())

# extract features i.e. the column - text and target i.e. the column category
features = data['text']
target   = data['category']


# Task 2: Text preprocessing
# -----------------------------------------------------------------------------
# let's observe a test in the dataset, extract the first text
# print(features[0])
'''
Barclays CEO stresses the importance of regulatory and cultural
reform in financial services at Brussels conference  http://t.co/Ge9Lp7hpyG
'''
# now extract the third text from this dataset
# print(features[2])
'''
Barclays publishes its prospectus for its �5.8bn Rights Issue: http://t.co/YZk24iE8G6
'''

# -----------------------------------------------------------------------------
# We will do the below pre-processing tasks on the text
# -----------------------------------------------------------------------------
# 1- tokenizing the sentences
# 2- replace the urls with a placeholder
# 3- removing non ascii characters
# 4- text normalizing using lemmatization
# -----------------------------------------------------------------------------
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
stoplist = set(stopwords.words('english'))

url_regex       = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
non_ascii_regex = r'[^\x00-\x7F]+'

def tokenize(text):
    text       = re.sub(non_ascii_regex, ' '             , text)
    text       = re.sub(url_regex      , 'urlplaceholder', text)
    tokens     = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for word in tokens:
        if word not in stoplist:
            clean_tokens.append(lemmatizer.lemmatize(word))

    return clean_tokens


# Task 3: EDA
# -----------------------------------------------------------------------------
# In this task, we will do exploratory data analysis to check if there is any
# new feature that we can generate based on the existing text that we have in
# the dataset

# Hypothesis 1: The length of the text in each category might be different from
# each other. 

# Hypothesis 2: The total number of URLs that are present in the text might be
# different in each category
# -----------------------------------------------------------------------------
# Hypothesis 1:
# create a new column in the original dataset - 'length' to capture lenght
# of each text
data['length'] = data['text'].apply(lambda x: len(word_tokenize(x)))

# use seaborn boxplot to visualize the pattern in length for each category
fig = plt.figure(figsize=(16,8))
sns.boxplot(x='category', y='length', data=data)
# plt.show()

# Hypothesis 2:
# create a new column in the original dataset - 'url_count' to capture total count
# of urls present in each text
data['url_count'] = data['text'].apply(lambda x: len(re.findall(url_regex, x)))

# use pandas crosstab to see the distribution of different url count in each category
# print(pd.crosstab(data['category'], data['url_count'], normalize=True))
'''
url_count           0         1         2
category                                 
Action       0.014565  0.166042  0.009155
Dialogue     0.032876  0.018727  0.000000
Information  0.205576  0.533916  0.019143
'''


# Task 4: Creating custom transformers
# -----------------------------------------------------------------------------
# An estimator is any object that learns from data, whether it's a classification,
# regression, or clustering algorithm, or a transformer that extracts or filters useful
# features from raw data. Since estimators learn from data, they each must have a fit
# method that takes a dataset.

# There are two kinds of estimators:
# 1- Transformer Estimators i.e. transformers in short. 
# 2- Predictor Estimators i.e. predictor in short.

# In transformers we also need to have another method transform and predictors need
# to have another method predict:
# Some examples of transformers are - CountVectorizer, TfidfVectorizer, MinMaxScaler, StandardScaler etc ...
# Some examples of predictors are - LinearRegression, LogisticRegression, RandomForestClassifier etc ...
# -----------------------------------------------------------------------------
# create a custom transformer LengthExtractor to extract length of each sentences
class LengthExtractor(BaseEstimator, TransformerMixin):
    def compute_lenght(self, text):
        sentence_list = word_tokenize(text)
        return len(sentence_list)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_length = pd.Series(X).apply(self.compute_lenght)
        return pd.DataFrame(X_length)

# create a custom transformer UrlCounter to count number of urls in each sentences
class UrlCounter(BaseEstimator, TransformerMixin):
    def count_url(self,text):
        urls = re.findall(url_regex, text)
        return len(urls)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        url_count = pd.Series(X).apply(self.count_url)
        return pd.DataFrame(url_count)


# Task 5: Model Building using FeatureUnion >>>>>> PipeLine<<<<<<<
# -----------------------------------------------------------------------------
# Feature union applies a list of transformer objects in parallel to the input
# data, then concatenates the results.
# This is useful to combine several feature extraction mechanisms into a single
# transformer.
# -----------------------------------------------------------------------------
# create an instance of Pipeline class
pipeline = Pipeline([
    # create a FeatureUnion pipeline
    ('features',FeatureUnion([
        # add a pipeline element to extract features using
        # CountVectorizer and TfidfTransformer
        ('text_pipline', Pipeline([
            ('vect' , CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
            ])),
        # add the pipeline element - LengthExtractor to extract lenght of
        # each sentence as feature
        ('text_len', LengthExtractor()),
        # add another pipeline element - UrlCounter to extract url counts
        # in each sentence as feature
        ('url_count', UrlCounter()),
        ])),
    # use the predictor/estimator RandomForestClassifier to train the model
    ('clf', RandomForestClassifier())
    ])

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# use pipeline.fit method to train the model
pipeline.fit(X_train, y_train)

# Task 6: Model Evaluation
# -----------------------------------------------------------------------------
# Now, once the model is trained, in this task we will evaluate how the model
# behaves in the test data
# -----------------------------------------------------------------------------
y_pred = pipeline.predict(X_test)

labels = np.unique(y_pred)
sns.heatmap(confusion_matrix(y_test,y_pred, labels=labels), annot=True, fmt='.0f')
plt.show()

print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

      Action       0.92      0.82      0.87        79
    Dialogue       1.00      0.89      0.94        36
 Information       0.95      0.98      0.97       366

    accuracy                           0.95       481
   macro avg       0.96      0.90      0.93       481
weighted avg       0.95      0.95      0.95       481
'''

# Task 7: Conclusion and next steps
# -----------------------------------------------------------------------------
# How to improve this model -
# 1- more feature engineering
# 2- feature selection
# 3- trying different predictors

search_space = [{'clf': [RandomForestClassifier()]     },
                {'clf': [PassiveAggressiveClassifier()]},
                {'clf': [SGDClassifier()]              },
                ]

gridsearch   = GridSearchCV(estimator  = pipeline,
                            param_grid = search_space,
                            scoring    = 'accuracy')
best_model   = gridsearch.fit(X_train, y_train)

# Evaluation
print('Best accracy: %f using %s'%(best_model.best_score_, best_model.best_params_))

cv_results = gridsearch.cv_results_['mean_test_score']
print(cv_results)

y_pred = gridsearch.predict(X_test)

labels = np.unique(y_pred)
sns.heatmap(confusion_matrix(y_test,y_pred, labels=labels), annot=True, fmt='.0f')
plt.show()

print(classification_report(y_test, y_pred))
                
                
    




















