# =================================================================================
# Arwa Ashi - HW2 - Week 12 - Nov 23, 2020
# Saudi Digital Academy
# Machine Learning
# =================================================================================
# Assignment
# Use PCA to reduce the number of features for the seattle weather data.
# =================================================================================

# Dimension reduction and principal component analysis (PCA)
# ---------------------------------------------------------------------------------

# Packages
# ---------------------------------------------------------------------------------
# 1- importing and preparing the Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 2- Normalize and center the data
from scipy.stats import boxcox
# 3- Standard scaling
from sklearn.preprocessing import StandardScaler
# 4- PCA
from sklearn.decomposition import PCA


# 1- importing and preparing the Data
# ---------------------------------------------------------------------------------
df = pd.read_csv('https://raw.githubusercontent.com/daniel-dc-cd/data_science/master/module_4_ML/data/seattle_weather_1948-2017.csv')

# Drop NAs
df = df.dropna()

#print(df.head())
#print(df.tail())
#print(df.info())

# Drop descriptive columns such as DATE and RAIN When we perform PCA, we want numerical features
df = df.drop(['DATE','RAIN'], axis=1)
print(df.head())
print(df.info())

# 2- Normalize and center the data (Our first transformation - Box-Cox)
# ---------------------------------------------------------------------------------
ax = df.hist(bins=50, xlabelsize=-1, ylabelsize=-1)
plt.show()

# We're going to start by trying the Box-Cox Transformation on the data, a popular
# transformation. It does require a strictly positive input, so we will add 1 to every
# value in each column.
df = df + 1

df_TF = pd.DataFrame(index=df.index)
for col in df.columns.values:
    df_TF['{}_TF'.format(col)] = boxcox(df.loc[:, col])[0]

ax = df_TF.hist(bins=50, xlabelsize=-1, ylabelsize=-1)
plt.show()

# 3- Standard scaling
# ---------------------------------------------------------------------------------
df_TF = StandardScaler().fit_transform(df_TF)
print(df_TF.shape)#(25548, 3)
print(df_TF)

'''
[[ 1.89415073 -0.63902519 -0.33821457]
 [ 1.9365653  -1.15868074 -0.98097635]
 [ 1.86352426 -1.15868074 -1.08303275]
 ...
 [-0.72051913 -1.06984799 -1.38013808]
 [-0.72051913 -0.89493243 -1.1835919 ]
 [-0.72051913 -0.72349953 -0.98097635]]
'''

df_1 = df_TF[np.logical_not(np.isnan(df_TF))]
print(df_1.reshape(-1, 1))

'''
[[ 1.89415073]
 [-0.63902519]
 [-0.33821457]
 ...
 [-0.72051913]
 [-0.72349953]
 [-0.98097635]]
'''

print("mean: ", np.round(df_TF.mean(), 2)) # mean:  -0.0
print("standard deviation: ", np.round(df_TF.std(), 2)) #standard deviation:  1.0

# 4- PCA 
# ---------------------------------------------------------------------------------
fit = PCA()
pca_data = fit.fit_transform(df_TF)
print(pca_data)

'''
[[ 1.28584908  1.5645957  -0.09582135]
 [ 2.07271799  1.31645734 -0.16100007]
 [ 2.11464215  1.21221352 -0.21691154]
 ...
 [ 1.38868571 -1.28024063 -0.03224251]
 [ 1.14140707 -1.19056744 -0.02609047]
 [ 0.89257956 -1.09910756 -0.01335246]]
'''

# calculate percentage variance for each component
per_var = np.round(fit.explained_variance_ratio_*100, decimals=1)
labels  = ['PC' + str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_df = pd.DataFrame(pca_data, columns=labels)

plt.scatter(pca_df.PC1, pca_df.PC2, alpha=0.2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
plt.show()

loading_scores          = pd.Series(fit.components_[0])
sorted_loloading_scores = loading_scores.abs().sort_values(ascending=False)
top_3                   = sorted_loloading_scores[0:3].index.values
print(sorted_loloading_scores[top_3])

