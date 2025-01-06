import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:\\Users\\amitc\\OneDrive\\Desktop\\ML\\ml_project\\rolling_stones_spotify.csv")
pd.set_option('display.max_rows', None, 'display.max_columns', None)
print(data.head())

print(data.info())

#1. Initial data inspection and data cleaning: 
#Dropping Unnamed column since we already have sequence
data.drop('Unnamed: 0', axis=1, inplace=True)

print(data.head())
print(data.describe())
print(data.isnull().sum())

#Examine the data initially to identify duplicates, missing values, irrelevant 
#entries, or outliers. Check for any instances of erroneous entries and 
#rectify them as needed
#Refine the data
from scipy import stats
from sklearn.preprocessing import StandardScaler

#Drop Duplicate rows
data = data.drop_duplicates()

#Handle missing values
data = data.dropna()
print(data.shape)

#Convert release_date to datetime
data['release_date'] =  pd.to_datetime(data['release_date'])

#Extract release year from release date
data['release_year'] = data['release_date'].dt.year

#Normalize numerical values
numerical_cols = ['acousticness', 'danceability','energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity', 'duration_ms']
ss = StandardScaler()
data[numerical_cols] = ss.fit_transform(data[numerical_cols])

print(data.describe())

#find the outliers in the data
# we can use boxplot to find the outliers per columns or we can use zscore to find outliers
#Zscore = (data_point -mean) / std. deviation 
#n outlier threshold value is chosen which is generally 3.0. As 99.7% of the data points lie between +/- 3 standard deviation (using Gaussian Distribution approach).

# sns.boxplot(data['danceability'])
# plt.show()
z=  np.abs(stats.zscore(data[numerical_cols]))
print(z)

# threshold_value = 3
# outlier_indices =  np.where(z> threshold_value)[0]
# data =  data.drop(outlier_indices)

# print(data.shape)
data = data[(np.abs(stats.zscore(data[numerical_cols])) < 3).all(axis=1)]
# data = data[np.abs(stats.zscore(data[numerical_cols]) < 3).all(axis=1)]
print(data.shape)

#Create a feature for the decade release
data['release_decade'] = (data['release_year'] //10) * 10

# Ensure consistent capitalization for album names
data['album'] = data['album'].str.title()

print(data.info())
print(data.head())

#Step 3 : Perform exploratory data analysis and feature engineering¶
#a. Utilize suitable visualizations to identify the two albums that should be recommended to anyone based on the number of popular songs in each album

print(len(data.album.unique()))
print(data.groupby('album')['popularity'].mean())

album_popularity =  data.groupby('album')['popularity'].mean().sort_values(ascending=False)

#Plotting the avaerage popularity of albums
plt.figure(figsize=(20,16))
sns.barplot(x= album_popularity.values, y=album_popularity.index)
plt.title("Average popularity of album")
plt.xlabel("Average Popularity")
plt.ylabel("Album")
plt.show()

top_2_album =  album_popularity.head(2)
print("Top 2 Albums are--", top_2_album)


#b.  Conduct exploratory data analysis to delve into various features of songs, aiming to identify patterns
#Correlation matrix
#Select only numerical columns from dataframe for correlation
category_df = data.select_dtypes(include=float)
print(category_df)
corr_matrix = category_df.corr()
print(corr_matrix)

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, vmin=-1)
plt.title("Correlation Matrxi of Song Features")
plt.show()


#c. Examine the relationship between a song's popularity and various factors, exploring how this correlation has evolved
#Scatter plot showing relationship between popularity with other features

features = ['acousticness', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
for feature in features:
    plt.title(f"Popularity vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("Popularity")
    sns.scatterplot(data=data, x=feature, y='popularity')
    plt.show()
    
#d.  Provide insights on the significance of dimensionality reduction techniques. Share your ideas and elucidate your observations 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Selecting numerical features
print(data.head())
numerical_feat =  data.select_dtypes(include=['float64', 'int64']).drop(columns=['popularity', 'duration_ms'])

#Standirising features
ss= StandardScaler()
scaled_featues =  ss.fit_transform(numerical_feat)

#Applying PCA
pca= PCA(n_components=2)
principal_comp = pca.fit_transform(scaled_featues)
print("principal comp",principal_comp)

#Explained variance
exp_var=  pca.explained_variance_ratio_
print(f"Explained Variance: {exp_var}")

# Adding principal components to the dataframe
data['PC1'] = principal_comp[:, 0]
data['PC2'] = principal_comp[:, 1]

# Plotting the PCA results
plt.figure(figsize=(10,6))
sns.scatterplot(x='PC1', y='PC2', data=data, hue='popularity', palette='viridis')
plt.title('PCA Results')
plt.show()

# Observcation¶
# First Principal Component (PC1): The first component explains 30.68% of the variance in the data. This suggests that a significant portion, but not the majority, of the data's variability is captured by this single dimension.

# Second Principal Component (PC2): The second component explains an additional 15.39% of the variance. Together, the first two components explain 46.07% of the total variance in the dataset.
# Moderate Explained Variance by PC1 and PC2:

# The first principal component captures 30.68% of the variance, which is substantial but not dominant. This indicates that while there is some strong underlying structure, the data is not overwhelmingly dominated by a single factor.

# The second principal component adds 15.39% to the explained variance. Together, they capture about 46.07% of the variance, which is less than half of the total variance. This suggests that the dataset has multiple factors contributing to its variability, each relatively important.

# Cumulative Explained Variance:The cumulative explained variance of 46.07% by the first two components implies that while these components capture a significant portion of the information, more components would be necessary to capture the majority of the variance. This might suggest that the data is complex and multifaceted.


#4. Perform cluster analysis
# a. Identify the right number of clusters¶

#using elbow method to find right number of cluster

from sklearn.cluster import KMeans

# Elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_featues)
    sse.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(10,6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#B .Use appropriate clustering algorithm
# Select numeric columns for clustering
numeric_cols = [
    'acousticness', 'danceability', 'energy', 'instrumentalness', 
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 
    'popularity', 'duration_ms'
]
df_numeric = data[numeric_cols]

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)
# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(df_scaled)
print(data.head())

# C. Define each cluster based on the features
# Aggregate by cluster and calculate the mean of numeric features
cluster_summary = data.groupby('cluster')[numeric_cols].mean()

# Display cluster summary
print(cluster_summary)

# Get the most common album for each cluster
most_common_album = data.groupby('cluster')['album'].agg(lambda x: x.mode()[0])

# Combine numeric summary with the most common album
cluster_summary['most_common_album'] = most_common_album

# Display combined cluster summary
print(cluster_summary)

# Key Insights:¶
# Popular Albums: "Aftermath (Uk Version)" and "Voodoo Lounge Uncut (Live)" were identified as albums with the most popular songs, making them strong candidates for recommendation.

# Feature Patterns: Popular songs tended to have higher energy, moderate danceability, and lower acousticness. The correlation analysis revealed that popularity was positively correlated with energy and negatively correlated with acousticness.

# Dimensionality Reduction: PCA effectively reduced the dataset's complexity while retaining almost half of the total variance in just two components. This simplification helped in visualizing and understanding the data better.

# Cluster Characteristics:
# Cluster 0: Moderate acousticness and danceability, higher energy, and popularity. Common album: "Aftermath (Uk Version)."
# Cluster 1: Lower acousticness and danceability, higher energy and popularity. Common album: "Voodoo Lounge Uncut (Live)."
# Cluster 2: Higher acousticness and danceability, lower energy and popularity. Common album: "Honk (Deluxe)."
# Conclusion:
# The project successfully utilized exploratory data analysis and clustering techniques to create meaningful cohorts of songs. The insights gained from the analysis can help improve song recommendations on Spotify by understanding the key features that define song popularity and clustering similar songs together. The approach of combining EDA, PCA, and clustering provides a comprehensive methodology for analyzing and interpreting complex datasets in the music industry.