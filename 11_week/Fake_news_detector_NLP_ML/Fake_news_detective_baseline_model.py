# ==================================================================
# ==================================================================
# Arwa Ashi -  Weekend Project -  Week 11 - Nov 20, 2020
# Saudi Digital Academy
# NLP Based Text Fake News Detector
# ==================================================================
# ==================================================================

# Packages
# --------------------------------------------------------------------------------------------------
# 1- (Raw Text) <<<<<  Reading and Exploring your Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 2- (Tokenization) and (Text Cleaning) and (Vectorization)
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
# 3- (ML Algorithm)
from sklearn.model_selection import train_test_split
# 4- (Transformer)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
# 5- Classifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
# 6- (Evaluation)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score


# Data
# 1- Reading and Exploring your Data
# ---------------------------------------------------------------------------------------------
# https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<
df_fake         = pd.read_csv('Fake.csv')
df_fake['label'] =  'Fake'
# print(df_fake.head(5))

df_real          = pd.read_csv('True.csv')
df_real['label'] =  'Real'
# print(df_real.head(5))

df = pd.concat([df_fake,df_real])
# df = pd.DataFrame(df)
# print(df.head(5))
# print(df.tail(5))

# checking consistancy
# df = df.dropna()
# print(df.label.value_counts())

# sns.countplot(x='label', data=df)
# plt.show()

# https://www.kaggle.com/c/fake-news/data <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
df_train = pd.read_csv('train.csv', index_col = 'id')
# df       = df_train
# print(df_train.head(5))

df_test = pd.read_csv('test.csv', index_col = 'id')
# print(df_test.head(5))

# checking consistancy
# print(df_train.label.value_counts())

# sns.countplot(x='label', data=df_train)
# plt.show()


# 2- (Tokenization) and (Text Cleaning) and (Vectorization)
# --------------------------------------------------------------------------------------------------
nltk.download('stopwords')
tokenizer=RegexpTokenizer('r\w+')
stopwords_english=set(stopwords.words('english'))

def cleanSms(sms):
 sms=sms.replace("<br /><br />"," ")
 sms=sms.lower()
 sms=sms.split() 
 sms= ''.join(p for p in sms if p not in punctuation)
 # Tokenizing the text
 sms_tokens=tokenizer.tokenize(sms)
 sms_tokens_without_stopwords=[token for token in sms_tokens if token not in stopwords_english]
 stemmed_sms_tokens_without_stopwords=[PorterStemmer().stem(token) for token in sms_tokens_without_stopwords]
 cleaned_sms=' '.join(stemmed_sms_tokens_without_stopwords)
 return cleaned_sms

# Clean the data & plot it on X & Y
df['title']   = df['title'].apply(cleanSms)
df['text']    = df['text'].apply(cleanSms)
df['subject'] = df['subject'].apply(cleanSms)
# print(df)

# Removing Null
nulls = df.isnull().sum()
nulls[nulls > 0]
df    = df.fillna(0)
# print(df['text'])

# Defining X and y
X = df['text'].values.astype('U')
y = df['label'].values.astype('U')
# print(X.shape) 
# print(y.shape) 


# 3- ML Algorithm
# --------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, shuffle=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)


# 4- (Transformer) TfidfTransformer (tf-idf) transformer
# --------------------------------------------------------------------------------------------------
# tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df=0.75)     # PAC Accuracy: 98.62
tfidf_vectorizer  = TfidfVectorizer(sublinear_tf=True, encoding='ISO-8859-1') # PAC Accuracy: 99.19

tfidf_vectorizer.fit(X_train)

vec_train = tfidf_vectorizer.transform(X_train)
vec_test  = tfidf_vectorizer.transform(X_test)

# print(vec_test.shape)
# print(y_test.shape)


# 5- Classifier
# --------------------------------------------------------------------------------------------------
# GLM
# PassiveAggressiveClassifier <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========================================================================
PAC = PassiveAggressiveClassifier(max_iter = 150)
PAC.fit(vec_train, y_train)
PAC_y_pred = PAC.predict(vec_test)

# LogisticRegression <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========================================================================
LR = LogisticRegression(solver='lbfgs')
LR.fit(vec_train, y_train)
LR_y_pred = LR.predict(vec_test)

# RidgeClassifier <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========================================================================
RC = RidgeClassifier()
RC.fit(vec_train, y_train)
RC_y_pred = RC.predict(vec_test)

# SGDClassifier <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========================================================================
SGDC = SGDClassifier()
SGDC.fit(vec_train, y_train)
SGDC_y_pred = SGDC.predict(vec_test)

# Perceptron <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========================================================================
PC = Perceptron()
PC.fit(vec_train, y_train)
PC_y_pred = PC.predict(vec_test)

# Ensemble Methods
# RandomForestClassifier <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========================================================================
RFC = RandomForestClassifier()
RFC.fit(vec_train, y_train)
RFC_y_pred = RFC.predict(vec_test)


# 6- (Evaluation)
# --------------------------------------------------------------------------------------------------
X           = tfidf_vectorizer.transform(X)

# PassiveAggressiveClassifier <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========================================================================
PAC_CMatrix = confusion_matrix(y_test, PAC_y_pred, labels=('Real','Fake'))
# print(PAC_CMatrix)
# sns.heatmap(PAC_CMatrix, annot=True, fmt='.0f')
# plt.show()

PAC_score   = accuracy_score(y_test, PAC_y_pred)
# print(f'PAC Accuracy: {round(PAC_score*100,2)}')

# Predict Spam
# print(PAC.predict(tfidf_vectorizer.transform(["you won $900 in the new lottery draw. Call +123456789."])))

PAC_scores  = cross_val_score(PAC, X, y, cv=5)
# print(f'PAC K Fold Accuracy: {round(PAC_scores.mean()*100,2)}%')


# LogisticRegression <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========================================================================
LR_CMatrix = confusion_matrix(y_test, LR_y_pred, labels=('Real','Fake'))
# print(LR_CMatrix)
# sns.heatmap(LR_CMatrix, annot=True, fmt='.0f')
# plt.show()

LR_score   = accuracy_score(y_test, LR_y_pred)
# print(f'LR Accuracy: {round(LR_score*100,2)}')

LR_scores  = cross_val_score(LR, X, y, cv=5)
# print(f'LR K Fold Accuracy: {round(LR_scores.mean()*100,2)}%')


# RidgeClassifier <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========================================================================
RC_CMatrix = confusion_matrix(y_test, RC_y_pred, labels=('Real','Fake'))
# print(RC_CMatrix)
# sns.heatmap(RC_CMatrix, annot=True, fmt='.0f')
# plt.show()

RC_score   = accuracy_score(y_test, RC_y_pred)
# print(f'RC Accuracy: {round(LR_score*100,2)}')

RC_scores  = cross_val_score(RC, X, y, cv=5)
# print(f'RC K Fold Accuracy: {round(RC_scores.mean()*100,2)}%')


# SGDClassifier <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========================================================================
SGDC_CMatrix = confusion_matrix(y_test, SGDC_y_pred, labels=('Real','Fake'))
# print(SGDC_CMatrix)
# sns.heatmap(SGDC_CMatrix, annot=True, fmt='.0f')
# plt.show()

SGDC_score   = accuracy_score(y_test, SGDC_y_pred)
# print(f'SGDC Accuracy: {round(SGDC_score*100,2)}')

SGDC_scores  = cross_val_score(SGDC, X, y, cv=5)
# print(f'SGDC K Fold Accuracy: {round(SGDC_scores.mean()*100,2)}%')


# Perceptron <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========================================================================
PC_CMatrix = confusion_matrix(y_test, PC_y_pred, labels=('Real','Fake'))
# print(PC_CMatrix)
# sns.heatmap(PC_CMatrix, annot=True, fmt='.0f')
# plt.show()

PC_score   = accuracy_score(y_test, PC_y_pred)
# print(f'PC Accuracy: {round(PC_score*100,2)}')

PC_scores  = cross_val_score(PC, X, y, cv=5)
# print(f'PC K Fold Accuracy: {round(PC_scores.mean()*100,2)}%')


# RandomForestClassifier <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========================================================================
RFC_CMatrix = confusion_matrix(y_test, RFC_y_pred, labels=('Real','Fake'))
# print(RFC_CMatrix)
# sns.heatmap(RFC_CMatrix, annot=True, fmt='.0f')
# plt.show()

RFC_score  = accuracy_score(y_test, RFC_y_pred)
# print(f'RFC Accuracy: {round(RFC_score*100,2)}')

RFC_scores  = cross_val_score(RFC, X, y, cv=5)
# print(f'RFC K Fold Accuracy: {round(RFC_scores.mean()*100,2)}%')


# DataFrame:
# What are the differences between the classifiers?
# --------------------------------------------------------------------------------------------------
ML_Compare_data = {'Classifier'      :['Passive Aggressive Classifier',
                                       'Logistic Regression'          ,
                                       'Ridge Classifier'             ,
                                       'SGD Classifier'               ,
                                       'Perceptron'                   ,
                                       'Random Forest Classifier'     ,
                                       ],
                   'accuracy_score'  :[round(PAC_score*100 ,2)       ,
                                       round(LR_score*100  ,2)       ,
                                       round(RC_score*100  ,2)       ,
                                       round(SGDC_score*100,2)       ,
                                       round(PC_score*100  ,2)       ,
                                       round(RFC_score*100 ,2)       ,
                                       ],
                   'K Fold Accuracy' :[round(PAC_scores.mean()*100 ,2),
                                       round(LR_scores.mean()*100  ,2),
                                       round(RC_scores.mean()*100  ,2),
                                       round(SGDC_scores.mean()*100,2),
                                       round(PC_scores.mean()*100  ,2),
                                       round(RFC_scores.mean()*100 ,2),
                                       ],
                   }
                   
df_ML_Compare_data = pd.DataFrame(ML_Compare_data) 
print('\n',df_ML_Compare_data)

sns.barplot(x='accuracy_score', y = 'Classifier', data = df_ML_Compare_data, color="b")#, hue='Classifier')

plt.title('Fake NEws Detective Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
plt.show()
                
                    
