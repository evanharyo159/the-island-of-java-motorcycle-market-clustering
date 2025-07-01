# the-island-of-java-motorcycle-market-clustering
Clustering analysis of districts and cities in the Island of Java in Indonesia to identify optimal targets for the motorcycle industry using demographic data and partitional hard clustering.

# Clustering Analysis of Districts and Cities in The Island of Java as Targets of Motorcycle Industry

## Overview

This research addresses the shrinking motorcycle market on the Island of Java, Indonesia, where 87% of private vehicles are motorcycles and 80% of households already own one. To help the motorcycle industry find new growth opportunities, this study focuses on **clustering regencies and cities in Java based on their demographic characteristics.** The aim is to provide data-driven recommendations for motorcycle manufacturers to optimize their product focus and strategies.

## Problem Statement

The high saturation of motorcycles in Java (60% of Indonesia's total, with 80% household ownership) indicates a shrinking market for new motorcycle sales. This poses a long-term challenge for the motorcycle industry, which requires new strategies beyond traditional sales to sustain growth.

## Research Objectives

* To cluster regencies and cities in Java based on their demographic characteristics.
* To evaluate and determine the most suitable clustering method for this dataset.
* To provide actionable decision recommendations for the motorcycle industry based on the formed clusters.

## Methodology

This research employed a **partitional hard clustering method** using 12 demographic variables, categorized into:
1.  **Economic Conditions of Society**
2.  **Living Conditions of Society**
3.  **Demographic Conditions of the Region**

The key steps involved:
1.  **Dataset Creation:** Data scraping from trusted sources.
2.  **Exploratory Data Analysis (EDA):** Initial analysis of the dataset.
3.  **Clustering:** Implementation of:
    * K-Means Clustering
    * K-Medoids Clustering
4.  **Cluster Evaluation:** Using four metrics to determine the best clustering method:
    * Silhouette Index
    * Dunn Index
    * Davies Bouldin Index
    * Calinski Harabasz Index

## Key Findings

The research found that the **K-Medoids Clustering method with 5 clusters** provided the most suitable grouping of regencies and cities in Java.

## Decision Recommendations for the Motorcycle Industry

Based on the 5 formed clusters, four distinct decision recommendations are provided:

1.  **Spare Parts Distribution:** Focus on optimizing the supply and distribution of motorcycle spare parts.
2.  **Workshop Establishment:** Prioritize the establishment and improvement of motorcycle repair and maintenance workshops.
3.  **Sales of Mid- to High-End Motorcycles:** Target these regions for the sale of premium and higher-priced motorcycle models.
4.  **Sales of Mid-Range Motorcycles and Below:** Focus on these areas for the sale of more affordable and entry-level motorcycle models.

## Data & Technologies Used

* **Data Source:** Demographics data scraped from trusted sites, such as Statistics Indonesia (BPS Indonesia)
* **Programming Language:** Python (likely for EDA, clustering, and evaluation).
* **Libraries (Expected):** `pandas`, `numpy`, `scikit-learn` (for clustering), `matplotlib`, `seaborn` (for visualization), etc.

## How to Reproduce / Run the Analysis
Open Google Collaboratory and try to run the code below or find the ipynb files in the folder
# Region-Based Clustering for Upper Secondary School Recommendations in Indonesia

This project applies unsupervised learning (K-Means and K-Medoids) to cluster 514 cities and regencies in Indonesia using socioeconomic indicators. The goal is to support education policy by recommending whether regions are more suited for general (academic) or vocational secondary education.

## 📁 Dataset
The dataset used includes socioeconomic variables from 2019 to 2024.

---

## 🚀 Getting Started

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Load Data
```python
import pandas as pd
file_path = '/content/drive/MyDrive/SKRIPSI-TA/SKRIPSI/data used/final_NO COR data_used_1622_revisi_bismillah.xlsx'
df = pd.read_excel(file_path)
df.drop(columns='Unnamed: 0', inplace=True)
```

---

## 📊 Descriptive Statistics & Outlier Detection
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Boxplot for one variable
var = 'PDRB ADHK (Rp)'
data = df[var]
plt.boxplot(data)
plt.title(f"Boxplot of {var}")
plt.show()

# Detect outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(data < lower) | (data > upper)]
print(outliers[['provinsi', 'kota', var]])
```

---

## 🔍 K-Means Clustering

### Elbow Method
```python
from sklearn.cluster import KMeans
X = df.iloc[:, 3:]
distortions = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=99).fit(X)
    distortions.append(km.inertia_)

plt.plot(range(2, 10), distortions)
plt.title('Elbow Curve')
plt.grid(True)
plt.show()
```

### Apply K-Means
```python
k = 5
km = KMeans(n_clusters=k, random_state=99)
df['Cluster_KMeans'] = km.fit_predict(X)
```

### PCA Visualization
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster_KMeans', data=df, palette='Set1')
plt.title('K-Means PCA Clusters')
plt.show()
```

---

## 📊 K-Medoids Clustering

### Install & Apply
```bash
!pip install scikit-learn-extra
```
```python
from sklearn_extra.cluster import KMedoids
km_medoids = KMedoids(n_clusters=5, random_state=99)
df['Cluster_KMedoids'] = km_medoids.fit_predict(X)
```

### Visualization
```python
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster_KMedoids', data=df, palette='Set2')
plt.title('K-Medoids PCA Clusters')
plt.show()
```

---

## ✅ Cluster Evaluation Metrics

### Silhouette Score
```python
from sklearn.metrics import silhouette_score
silhouette_score(X, df['Cluster_KMeans'])
silhouette_score(X, df['Cluster_KMedoids'])
```

### Davies-Bouldin Index
```python
from sklearn.metrics import davies_bouldin_score
davies_bouldin_score(X, df['Cluster_KMeans'])
davies_bouldin_score(X, df['Cluster_KMedoids'])
```

### Calinski-Harabasz Index
```python
from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_score(X, df['Cluster_KMeans'])
calinski_harabasz_score(X, df['Cluster_KMedoids'])
```

---

## 🔬 Dunn Index (Custom Implementation)
```python
from sklearn.metrics import pairwise_distances
from itertools import combinations
import numpy as np

def dunn_index(clusters, distance_matrix):
    max_intra = max(np.max(distance_matrix[np.ix_(c, c)]) for c in clusters)
    min_inter = min(np.min(distance_matrix[np.ix_(c1, c2)]) for c1, c2 in combinations(clusters, 2))
    return min_inter / max_intra

# Example for KMeans
labels = df['Cluster_KMeans']
clusters = [np.where(labels == i)[0] for i in range(k)]
distance_matrix = pairwise_distances(X)
print("Dunn Index (KMeans):", dunn_index(clusters, distance_matrix))
```

---

## 🔢 Result Summary
- K-Means with **5 clusters** gave the best balance of silhouette score and interpretability.
- **Cluster 0 and 2** show higher tech and growth potential → fit for **general/academic** schools.
- **Cluster 1 and 3** show lower development → suited for **vocational** education.
- **Cluster 4** shows mixed characteristics.

---

## 💼 Future Work
- Integrate cultural/community variables.
- Combine spatial GIS data for deeper insights.
- Apply model to more recent post-2024 projections.

---

## 📃 License
This project is part of academic research and open for non-commercial use.

---

**Created by:** Evan Haryowidyatna | 2025


## Visualizations

* **Clustering results:**
* ![image](https://github.com/user-attachments/assets/6fc0f61a-16da-47c4-ae61-5531513d5269)
* ![image](https://github.com/user-attachments/assets/06506790-a304-43f0-af2d-09bcc56e8d89)
* <img width="457" alt="image" src="https://github.com/user-attachments/assets/7017412e-e911-4fe8-9e2c-59f283faabc2" />

* **Evaluation metrics:** 
<img width="726" alt="image" src="https://github.com/user-attachments/assets/97c96a43-7456-42e2-b924-efba8cd72438" />
<img width="759" alt="image" src="https://github.com/user-attachments/assets/4eb9262e-7c42-480b-a65c-d54c7ed8f3b8" />

## Contact

[Evan Haryowidyatna]
[e.haryowidyatna@gmail.com]
[https://github.com/evanharyo159]
