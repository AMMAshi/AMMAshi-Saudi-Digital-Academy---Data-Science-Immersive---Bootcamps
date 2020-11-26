# =================================================================================
# Arwa Ashi - HW4 - Week 12 - Nov 25, 2020
# Saudi Digital Academy
# Machine Learning
# =================================================================================
# Assignment
# Use the Seattle rain data an apply K-Means clustering to the dataset.

# Use K-Means on the Seattle Weather data
# =================================================================================

# Packages
# ---------------------------------------------------------------------------------
# importing and preparing the Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
# K-Means
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


# Data
# ---------------------------------------------------------------------------------
df      = pd.read_csv('https://raw.githubusercontent.com/daniel-dc-cd/data_science/master/module_4_ML/data/seattle_weather_1948-2017.csv')
numrows = 25549 

# Create an empty dataframe to hold values
Kmean_df = pd.DataFrame({'before_yesterday':[0.0]*numrows,
                         'yesterday'       :[0.0]*numrows,
                         'today'           :[0.0]*numrows,
                         'tomorrow'        :[0.0]*numrows})

# Sort columns for convience
seq      = ['before_yesterday', 'yesterday', 'today','tomorrow']
Kmean_df = Kmean_df.reindex(columns=seq)

for i in range(0,numrows):
    tomorrow          = df.iloc[i,1]
    today             = df.iloc[(i-1),1]
    yesterday         = df.iloc[(i-2),1]
    before_yesterday  = df.iloc[(i-3),1]
    Kmean_df.iat[i,3] = tomorrow
    Kmean_df.iat[i,2] = today
    Kmean_df.iat[i,1] = yesterday
    Kmean_df.iat[i,0] = before_yesterday

Kmean_df   = Kmean_df.dropna()


# Building the matrix and model, to give the data more Normalization we can take the log
# ---------------------------------------------------------------------------------
tomorrow_data           = np.array(Kmean_df['tomorrow'])
today_data              = np.array(Kmean_df['today'])
yesterday_data          = np.array(Kmean_df['yesterday'])
before_yesterday_data   = np.array(Kmean_df['before_yesterday'])

data = pd.DataFrame({'tomorrow'           : tomorrow_data,
                     'today'              : today_data,
                     'yesterday'          : yesterday_data,
                     'before_yesterday'   : before_yesterday_data
                     })
print(data.head())

scatter_matrix(data, alpha = 0.2, figsize = (11,5), diagonal = 'kde');
plt.scatter(tomorrow_data, today_data, c='black', s=7)
plt.show()

# Data Correlation
# ---------------------------------------------------------------------------------
sns.heatmap(data.corr())
plt.show()
print(data.corr())

# K-means Implementation
# ---------------------------------------------------------------------------------
matrix = data.values

n_clusters = 4
model      = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
model.fit(matrix)

Rain_clusters = model.predict(matrix) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< what we going to plot
silhouette_avg   = silhouette_score(matrix, Rain_clusters)
print('score de silhouette: {:<.3f}'.format(silhouette_avg)) 
print(model.inertia_)

# Visualization 
# ---------------------------------------------------------------------------------
# Clusters scatter plot
plt.scatter(matrix[:, 0], matrix[:, 1], c=Rain_clusters, s=50, cmap='viridis')
# select cluster centers
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Tomorrow will rain or not ??')  
plt.xlabel('Tomorrow')  
plt.ylabel('Todat')  
plt.show()

# -------------------------------------------------------------------------------------
from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

centers, labels = find_clusters(matrix, 4)
plt.scatter(matrix[:, 0], matrix[:, 1], c=labels,s=50, cmap='viridis');

centers, labels = find_clusters(matrix, 4, rseed=0)
plt.scatter(matrix[:, 0], matrix[:, 1], c=labels,s=50, cmap='viridis')
plt.show()

labels = KMeans(6, random_state=0).fit_predict(matrix)
plt.scatter(matrix[:, 0], matrix[:, 1], c=labels,s=50, cmap='viridis')
plt.show()

# -------------------------------------------------------------------------------------
from sklearn.datasets import make_moons
labels = KMeans(2, random_state=0).fit_predict(matrix)
plt.scatter(matrix[:, 0], matrix[:, 1], c=labels,s=50, cmap='viridis')
plt.show()

