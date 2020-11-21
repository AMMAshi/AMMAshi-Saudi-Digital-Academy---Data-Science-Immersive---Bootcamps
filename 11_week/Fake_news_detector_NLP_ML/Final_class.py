# Make necessary imports
# 1- (Raw Text) <<<<<  Reading and Exploring your Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
# 2- (Tokenization) and (Text Cleaning) and (Vectorization)
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from string import punctuation
# 3- (ML Algorithm)
from sklearn.model_selection import train_test_split
# 4- (Transformer)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# 5- Pipline Classifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn import svm
# 6- (Evaluation)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# Data
# 1- Reading and Exploring your Data
# ---------------------------------------------------------------------------------------------
# Read the data
df = pd.read_csv('train.csv')

# Drop unrelated features first, then drop missing data
df = df.drop(columns=['title', 'author']).dropna()


# 2- (Tokenization) and (Text Cleaning) and (Vectorization)
# --------------------------------------------------------------------------------------------------
nltk.download('stopwords')
tokenizer=RegexpTokenizer('r\w+')
stopwords_english=set(stopwords.words('english'))

def CleanNews(news):
 news=news.replace("<br /><br />"," ")
 news=news.lower()
 news=news.split() 
 news= ''.join(p for p in news if p not in punctuation)
 # Tokenizing the text
 news_tokens=tokenizer.tokenize(news)
 news_tokens_without_stopwords=[token for token in news_tokens if token not in stopwords_english]
 stemmed_news_tokens_without_stopwords=[PorterStemmer().stem(token) for token in news_tokens_without_stopwords]
 cleaned_news=' '.join(stemmed_news_tokens_without_stopwords)
 return cleaned_news

# Clean the data 
# df['text'] = df['text'].apply(CleanNews)
# print(df)

# Removing Null
nulls = df.isnull().sum()
nulls[nulls > 0]
df    = df.fillna(0)
# print(df['text'])

# Defining X and y
X = df['text'].values.astype('U')
y = df['label'].values.astype('U')
#print(X.shape) 
#print(y.shape)


# 3- ML Algorithm
# --------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)#, shuffle=True)


# 4- (Transformer) TfidfTransformer (tf-idf) transformer
# --------------------------------------------------------------------------------------------------
# tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df=0.75)     # PAC Accuracy: 0.961015 0.874053
tfidf_vectorizer  = TfidfVectorizer(sublinear_tf=True, encoding='ISO-8859-1') # PAC Accuracy: 0.972768 0.875915

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test  = tfidf_vectorizer.transform(X_test)


# 5- Pipeline
# --------------------------------------------------------------------------------------------------
pipe         = Pipeline(steps = [('clf', PassiveAggressiveClassifier())])
                         
                         
search_space = [{'clf': [PassiveAggressiveClassifier()]},
                {'clf': [MultinomialNB()]},
                {'clf': [BernoulliNB()]},
                {'clf': [RidgeClassifier()]},
                {'clf': [SGDClassifier()]},
                {'clf': [Perceptron()]},
                {'clf': [RandomForestClassifier()]}]
                         
gridsearch   = GridSearchCV(estimator  = pipe,
                          param_grid = search_space,
                          scoring    = 'accuracy')
                         
best_model   = gridsearch.fit(tfidf_train, y_train)

# 6- (Evaluation)
# --------------------------------------------------------------------------------------------------
print('Best accuracy: %f using %s'%(best_model.best_score_, best_model.best_params_))

y_pred = gridsearch.predict(tfidf_test)

# Build confusion matrix. 1: unreliable, 0: reliable
# print(confusion_matrix(y_test, y_pred, labels=[1, 0]))

# scores  = cross_val_score(gridsearch, X, y, cv=5)
# print(f'PAC K Fold Accuracy: {round(scores.mean()*100,2)}%')

# 5- Test Data
# --------------------------------------------------------------------------------------------------
# 1- Read the data
test_data = pd.read_csv('test.csv')

# Assign ids to an object to use it later for Kaggle submission
test_id = test_data['id']

# 2- (Tokenization) and (Text Cleaning) and (Vectorization)
test_data         = test_data.drop(columns=['id','title', 'author']).fillna('fake and unreliable')
test_data         = test_data.fillna(0)
test_data['text'] = test_data['text'].values.astype('U')

# 3- (Transformer) TfidfTransformer (tf-idf) transformer
test_vectorized = tfidf_vectorizer.transform(test_data['text'])

# 4- Predict test data
test_predictions = gridsearch.predict(test_vectorized)

# Join test data's ids with their respective predicted labels
submission = pd.DataFrame({'id':test_id, 'label':test_predictions})
print(submission.shape)
print(submission.head())



















