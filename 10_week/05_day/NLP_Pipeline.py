# Nov 12, 2020

from __future__ import print_function
# %matplotlib inline
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import pandas as pd
import pandas_profiling
import seaborn as sns
# plt.style.use("ggplot")
# warnings.simplefilter("ignore")

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive, IntSlider, FloatSlider, interact
import ipywidgets
from IPython.display import Image
from subprocess import call
import matplotlib.image as mpimg
from pandas_profiling import ProfileReport

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive, IntSlider, FloatSlider, interact
import ipywidgets
from IPython.display import Image
from subprocess import call
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from yellowbrick.model_selection import FeatureImportances



# Task 1: Load the data and examine it.
# -------------------------------------------------------------------------
data = pd. read_csv('https://raw.githubusercontent.com/daniel-dc-cd/data_science/master/daily_materials/NLP_pipeline_activity/corporate_messaging_dfe.csv')
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

# print(data['category'].value_counts())
'''
Information    2129
Action          724
Dialogue        226
Exclude          39
Name: category, dtype: int64
'''

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

data = data[(data['category_confidence']==1)&(data['category'] != 'Exclude')]

features = data['text']
target   = data['category']

print(features.shape)
print(target.shape)

# Task 2: Text preprocessing
# -------------------------------------------------------------------------
# We will do the below pre-processing tasks on the text
# tokenizing the sentences
# replace the urls with a placeholder
# removing non ascii characters
# text normalizing using lemmatization
# -------------------------------------------------------------------------

# print(features[0])
# print(features[2])

import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# download the stopwords and wordnet corpus

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
stoplist = stopwords. words('english')

# write a regular expression to identify urls in text
url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# write a regular expression to identify non-ascii characters in text
non_ascii_regrex = r'[^\x00-\x7F]+'


# write a function to tokenize text after performing preprocessing 
def tokenize(text):
    text = re.sub(non_ascii_regrex,'', text)
    test = re.sub(url_regex, 'urlplaceholder', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_token = []

    for word in tokens:
        if word not in stoplist:
            clean_token.append(lemmatizer.lemmatize(word))
    return clean_token

# Task 3: EDA
# -------------------------------------------------------------------------
# Hypothesis 1: The length of the text in each category might be different from each other
# Hypothesis 2: The total number of URLs that are present in text might be different in each category
# -------------------------------------------------------------------------
data['length'] = data['text'].apply(lambda x: len(word_tokenize(x)))

#fig = plt.figure(figsize(16,8))
sns.boxplot(x = 'category', y='length', data=data)
# plt.show()

# create a new column in the original dataset - 'url_count' to capture total count of urls present in each text
data['url_count'] = data['text'].apply(lambda x: len(re.findall(url_regex, x)))

# use pandas crosstab to see the distibution of different url counts in each category
# print(pd.crosstab(data['category'], data['url_count'], normalize=True))
'''
url_count           0         1         2
category                                 
Action       0.014565  0.166042  0.009155
Dialogue     0.032876  0.018727  0.000000
Information  0.205576  0.533916  0.019143
'''

# Task 4: Creating custom transformers
# -------------------------------------------------------------------------
'''
An estimator is any object that learns from data, whether it's a classification, regression, or clustering
algorithm, or a transformer that extracts or filters useful features from raw data. Since estimators learn
from data, they each must have a fit method that takes a dataset.

There are two kinds of estimators - Transformer Estimators i.e. transformers in short and Predictor Estimators
i.e. predictor in short. In transformers we also need to have another method transform and predictors need to
have another method predict.

Some examples of transformers are - CountVectorizer, TfidfVectorizer, MinMaxScaler, StandardScaler etc

Some examples of predictors are - LinearRegression, LogisticRegression, RandomForestClassifier etc
'''

from sklearn.base import BaseEstimator, TransformerMixin

# create a custom transformer LengthExtractor to extract length of each sentences
class LengthExtractor(BaseEstimator, TransformerMixin):

    def compute_length(self, text):
        sentence_list = word_tokenize(text)
        return len(sentence_list)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_length = pd.Series(X).apply(self.compute_length)
        return pd.DataFrame(X_length)



# create a custom transformer UrlCounter to count number of urls in each sentences
class UrlCounter(BaseEstimator, TransformerMixin):

    def count_url(self, text):
        urls = re.findall(url_regex, text)
        return len(urls)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        url_count = pd.Series(X).apply(self.count_url)
        return pd.DataFrame(url_count)


# Task 5: Model Building using FeatureUnion
# -------------------------------------------------------------------------
'''
Feature union applies a list of transformer objects in parallel to the input data, then concatenates
the results. This is useful to combine several feature extraction mechanisms into a single transformer.
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# create an instance of Pipeline class
# create a FeatureUnion pipeline
# add a pipeline element to extract features using CountVectorizer and TfidfTransformer
# add the pipeline element - LengthExtractor to extract lenght of each sentence as feature
# add another pipeline element - UrlCounter to extract url counts in each sentence as feature
# use the predictor/estimator RandomForestClassifier to train the model
'''
pipeline = Pipeline([
    ('features',  FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
            ])),
        ('text_len', LengthExtractor()),
        ('url_count',UrlCounter()),
        ])),
    ('clf', RandomForestClassifier())
    ])
'''
pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('text_len', LengthExtractor()),
            ('url_count', UrlCounter()),
        ])),
        ('clf', RandomForestClassifier())
    ])
'''
# split the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, text_size= 0.2, random_state=42, stratify=target)
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# use pipeline.fit method to train the model
pipeline.fit(X_train,y_train)
print(pipeline.fit(X_train,y_train))

# Task 6: Model Evaluation
# -------------------------------------------------------------------------
# use the method pipeline.predict on X_test data to predict the labels
y_pred = pipeline.predict(X_test)

# create the confustion matrix, import confusion_matrix from sklearn
# count the number of labels
# use sns.heatmap on top of confusion_matrix to show the confusuin matrix
from sklearn.metrics import confusion_matrix
labels = np.unique(y_pred)
sns.heatmap(confusion_matrix(y_test, y_pred, labels=labels), annot=True, fmt='.0f')
plt.show()

# High number from true positive
# doing for effecncy
#
# create the classification report, import classification_report from sklearn
# apply the function classification_report on y_test, y_pred and print it
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Task 7: Conclusion and next steps
# -------------------------------------------------------------------------
# How to improve this model -

# more feature engineering
# feature selection
# trying different predictors


















