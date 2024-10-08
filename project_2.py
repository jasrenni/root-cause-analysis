# -*- coding: utf-8 -*-
"""bottle_graph.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Tqd2yo96AYRu6QdHLxTwKJwTVNlkJ1st
"""

import pandas as pd

# Load Jira dataset
jira_df = pd.read_csv('Jira (5).csv')
# Extract relevant features
jira_df = jira_df[['Issue id', 'Summary', 'Issue key', 'Issue Type', 'Status', 'Priority', 'Assignee']]

# Display the first few rows to check the data
jira_df.head()

pip install sentence-transformers

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Text cleaning function
def clean_text(text):
    # Remove punctuation and lowercase the text
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply the cleaning function to the 'Summary' column
jira_df['Cleaned Summary'] = jira_df['Summary'].apply(clean_text)

# Display the first few cleaned summaries
jira_df[['Summary', 'Cleaned Summary']].head()

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Text cleaning function
def clean_text(text):
    # Remove punctuation and lowercase the text
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply the cleaning function to the 'Summary' column
jira_df['Cleaned Summary'] = jira_df['Summary'].apply(clean_text)

# Display the first few cleaned summaries
jira_df[['Summary', 'Cleaned Summary']].head()

from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model for embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings for each cleaned summary
embeddings = model.encode(jira_df['Cleaned Summary'].tolist(), show_progress_bar=True)

# Check the shape of the embeddings
print(embeddings.shape)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Compute cosine similarities between all pairs of issue summaries
similarity_matrix = cosine_similarity(embeddings)

# Convert similarity matrix to a DataFrame for easier handling
similarity_df = pd.DataFrame(similarity_matrix, index=jira_df['Issue id'], columns=jira_df['Issue id'])

# Display the similarity matrix
similarity_df.head()

import networkx as nx

# Create a graph where nodes are issue IDs
G = nx.Graph()

# Add nodes
for issue_id in jira_df['Issue id']:
    G.add_node(issue_id)

# Add edges based on similarity threshold
threshold = 0.8  # Adjust the threshold as needed
for i in range(len(similarity_df)):
    for j in range(i+1, len(similarity_df)):
        if similarity_df.iloc[i, j] > threshold:
            G.add_edge(similarity_df.index[i], similarity_df.columns[j], weight=similarity_df.iloc[i, j])

# Identify bottlenecks (e.g., high-degree nodes)
bottlenecks = sorted(G.degree, key=lambda x: x[1], reverse=True)

# Display the top 5 bottlenecks
print("Top 5 bottlenecks (issue IDs):", bottlenecks[:5])

import networkx as nx

# Create a graph where nodes are issue IDs
G = nx.Graph()

# Add nodes
for issue_id in jira_df['Issue id']:
    G.add_node(issue_id)

# Add edges based on similarity threshold
threshold = 0.8  # Adjust the threshold as needed
for i in range(len(similarity_df)):
    for j in range(i+1, len(similarity_df)):
        if similarity_df.iloc[i, j] > threshold:
            G.add_edge(similarity_df.index[i], similarity_df.columns[j], weight=similarity_df.iloc[i, j])

# Identify bottlenecks (e.g., high-degree nodes)
bottlenecks = sorted(G.degree, key=lambda x: x[1], reverse=True)

# Display the top 5 bottlenecks
print("Top 5 bottlenecks (issue IDs):", bottlenecks[:5])

pip install faiss-cpu

import faiss

# Convert embeddings to float32
embeddings = np.array(embeddings).astype('float32')

# Build the FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # Using L2 distance
index.add(embeddings)  # Add embeddings to the index

# Search for nearest neighbors
D, I = index.search(embeddings, k=10)  # Find the 10 nearest neighbors for each issue

# Display the first few results
print(I[:5])

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans

# Example issue embeddings (replace with your actual embeddings)
# Suppose you have 100 issues and each is represented by a 768-dimensional embedding
np.random.seed(42)
issue_embeddings = np.random.rand(100, 768)  # Example with random data
issue_ids = [f"ISSUE_{i}" for i in range(100)]  # Replace with your actual issue IDs

# Cluster the issues based on their embeddings
n_clusters = 10  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(issue_embeddings)

# Create a new graph for the clusters
G_clustered = nx.Graph()

# Add nodes representing each cluster
for i in range(n_clusters):
    G_clustered.add_node(i, issues=[issue_ids[j] for j in range(len(issue_ids)) if labels[j] == i])

# Add edges between clusters based on bottlenecks (high degree in original graph)
for i in range(n_clusters):
    for j in range(i+1, n_clusters):
        # Assuming you have a method to determine if clusters should be connected
        G_clustered.add_edge(i, j, weight=np.random.randint(1, 10))  # Replace with actual logic

# Visualize the clustered graph
pos = nx.spring_layout(G_clustered, seed=42)

# Node color based on the size of the cluster (number of issues in the cluster)
node_color = [len(G_clustered.nodes[node]['issues']) for node in G_clustered.nodes]
nx.draw_networkx_nodes(G_clustered, pos, node_size=500, cmap=plt.cm.Blues, node_color=node_color)

# Draw edges
nx.draw_networkx_edges(G_clustered, pos, alpha=0.5)

# Draw node labels (e.g., cluster ID)
nx.draw_networkx_labels(G_clustered, pos, font_size=10, labels={i: f"Cluster {i}" for i in range(n_clusters)})

# Show plot
plt.title('Clustered Knowledge Graph Showing Issue Similarities and Bottlenecks')
plt.show()

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Jira CSV file
jira_df = pd.read_csv('Jira (5).csv')

# Extract issue summaries
summaries = jira_df['Summary'].dropna().tolist()

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Preprocess and encode summaries into embeddings
summary_embeddings = []
for summary in summaries:
    inputs = tokenizer(summary, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    summary_embeddings.append(embeddings[0])

summary_embeddings = np.array(summary_embeddings)

# Cluster summaries using Agglomerative Clustering
agglomerative = AgglomerativeClustering(n_clusters=9, linkage='ward')
cluster_labels = agglomerative.fit_predict(summary_embeddings)

# Calculate cluster weights
cluster_weights = np.bincount(cluster_labels)

# Identify bottleneck cluster
bottleneck_cluster = np.argmax(cluster_weights)

# Visualize clusters and bottleneck
plt.figure(figsize=(10, 6))
sns.barplot(x=np.arange(len(cluster_weights)), y=cluster_weights)
plt.title('Cluster Weights (Agglomerative Clustering)')
plt.xlabel('Cluster')
plt.ylabel('Weight')
plt.axhline(y=cluster_weights[bottleneck_cluster], color='r', linestyle='--')
plt.show()

# Output cluster summaries
for i in range(len(cluster_weights)):
    print(f"\nCluster {i} Summary:")
    cluster_summaries = [summaries[j] for j in range(len(summaries)) if cluster_labels[j] == i]
    for summary in cluster_summaries:
        print(summary)

# Report bottleneck
print(f"\nBottleneck Cluster: {bottleneck_cluster} with weight {cluster_weights[bottleneck_cluster]}")

from sklearn.cluster import AgglomerativeClustering

# Cluster summaries using Agglomerative Clustering
agglomerative = AgglomerativeClustering(n_clusters=9, linkage='ward')
cluster_labels = agglomerative.fit_predict(summary_embeddings)

# Calculate cluster weights
cluster_weights = np.bincount(cluster_labels)

# Identify bottleneck cluster
bottleneck_cluster = np.argmax(cluster_weights)

# Visualize clusters and bottleneck
plt.figure(figsize=(10, 6))
sns.barplot(x=np.arange(len(cluster_weights)), y=cluster_weights)
plt.title('Cluster Weights (Agglomerative Clustering)')
plt.xlabel('Cluster')
plt.ylabel('Weight')
plt.axhline(y=cluster_weights[bottleneck_cluster], color='r', linestyle='--')
plt.show()

# Output cluster summaries
for i in range(len(cluster_weights)):
    print(f"\nCluster {i} Summary:")
    cluster_summaries = [summaries[j] for j in range(len(summaries)) if cluster_labels[j] == i]
    for summary in cluster_summaries:
        print(summary)

# Report bottleneck
print(f"\nBottleneck Cluster: {bottleneck_cluster} with weight {cluster_weights[bottleneck_cluster]}")

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Jira CSV file
jira_df = pd.read_csv('Jira (5).csv')

# Extract issue summaries
summaries = jira_df['Summary'].dropna().tolist()

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Preprocess and encode summaries into embeddings
summary_embeddings = []
for summary in summaries:
    inputs = tokenizer(summary, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    summary_embeddings.append(embeddings[0])

summary_embeddings = np.array(summary_embeddings)

# Cluster summaries using KMeans
n_clusters = 9
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(summary_embeddings)

# Calculate cluster weights
cluster_weights = np.bincount(cluster_labels)
cluster_centers = kmeans.cluster_centers_

# Calculate sum of distances of samples to their closest cluster center
_, distances = pairwise_distances_argmin_min(summary_embeddings, cluster_centers)
cluster_distances = [np.sum(distances[cluster_labels == i]) for i in range(n_clusters)]

# Identify bottleneck clusters (with highest weight)
bottleneck_cluster = np.argmax(cluster_weights)

# Visualize clusters and bottleneck
plt.figure(figsize=(10, 6))
sns.barplot(x=np.arange(n_clusters), y=cluster_weights)
plt.title('Cluster Weights')
plt.xlabel('Cluster')
plt.ylabel('Weight')
plt.axhline(y=cluster_weights[bottleneck_cluster], color='r', linestyle='--')
plt.show()

# Output cluster summaries
for i in range(n_clusters):
    print(f"\nCluster {i} Summary:")
    cluster_summaries = [summaries[j] for j in range(len(summaries)) if cluster_labels[j] == i]
    for summary in cluster_summaries:
        print(summary)

# Report bottleneck
print(f"\nBottleneck Cluster: {bottleneck_cluster} with weight {cluster_weights[bottleneck_cluster]}")

from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

# Define the number of clusters for KMeans (you can adjust this)
n_clusters = 10

# K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

kmeans_labels = kmeans.fit_predict(summary_embeddings)
kmeans_silhouette = silhouette_score(summary_embeddings, kmeans_labels)
print(f"K-Means Silhouette Score: {kmeans_silhouette}")

# Mean Shift Clustering
mean_shift = MeanShift()
mean_shift_labels = mean_shift.fit_predict(summary_embeddings)
mean_shift_silhouette = silhouette_score(summary_embeddings, mean_shift_labels)
print(f"Mean Shift Silhouette Score: {mean_shift_silhouette}")



# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=5)  # Adjust these parameters as needed
dbscan_labels = dbscan.fit_predict(summary_embeddings)

# Check the unique labels
unique_labels = set(dbscan_labels)
print(f"Unique labels from DBSCAN: {unique_labels}")

# Filter out noise points
if len(unique_labels) > 1:
    # Exclude noise points (-1)
    valid_labels = [label for label in dbscan_labels if label != -1]
    valid_embeddings = summary_embeddings[np.isin(dbscan_labels, valid_labels)]
    valid_dbscan_labels = [label for label in dbscan_labels if label != -1]
    dbscan_silhouette = silhouette_score(valid_embeddings, valid_dbscan_labels)
else:
    dbscan_silhouette = 'Not enough clusters for silhouette score'

print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
reduced_embeddings = pca.fit_transform(summary_embeddings)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)  # Set n_init to avoid FutureWarning
kmeans_labels = kmeans.fit_predict(reduced_embeddings)

# Mean Shift Clustering
mean_shift = MeanShift()
mean_shift_labels = mean_shift.fit_predict(reduced_embeddings)

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=5)  # Adjust number of clusters as needed
agg_labels = agg_clustering.fit_predict(reduced_embeddings)

# Plot K-Means clusters
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title('K-Means Clusters')
plt.colorbar()

# Plot Mean Shift clusters
plt.subplot(1, 3, 2)
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=mean_shift_labels, cmap='viridis', s=50)
plt.title('Mean Shift Clusters')
plt.colorbar()

# Plot Agglomerative Clustering
plt.subplot(1, 3, 3)
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=agg_labels, cmap='viridis', s=50)
plt.title('Agglomerative Clustering')
plt.colorbar()

plt.show()

# Plot Dendrogram for Agglomerative Clustering
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(summary_embeddings, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

import pandas as pd

# Load the first 100 rows of the Jira CSV file
df = pd.read_csv('Jira (5).csv', nrows=100)

# Print the column names to identify the correct text column
print(df.columns)

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np

# Load the first 100 rows of the Jira CSV file
df = pd.read_csv('Jira (5).csv', nrows=100)

# Inspect column names
print(df.columns)

# Use 'Summary' or 'Description' based on your preference
texts = df['Summary'].fillna('')  # Replace 'Summary' with 'Description' if needed

# Convert text to embeddings using a pre-trained model (e.g., BERT, Sentence-BERT)
# For demonstration, this step is skipped here. Assume `summary_embeddings` is obtained.
# summary_embeddings = ...

# For demonstration, create random embeddings if actual embeddings are not available
summary_embeddings = np.random.rand(len(df), 100)  # Replace with actual embeddings

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(summary_embeddings)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(reduced_embeddings)

# Mean Shift Clustering
mean_shift = MeanShift()
mean_shift_labels = mean_shift.fit_predict(reduced_embeddings)

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=5)
agg_labels = agg_clustering.fit_predict(reduced_embeddings)

# Plot K-Means clusters
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title('K-Means Clusters')
plt.colorbar()

# Plot Mean Shift clusters
plt.subplot(1, 3, 2)
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=mean_shift_labels, cmap='viridis', s=50)
plt.title('Mean Shift Clusters')
plt.colorbar()

# Plot Agglomerative Clustering
plt.subplot(1, 3, 3)
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=agg_labels, cmap='viridis', s=50)
plt.title('Agglomerative Clustering')
plt.colorbar()

plt.show()

# Plot Dendrogram for Agglomerative Clustering
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(summary_embeddings, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Load the first 100 rows of the Jira CSV file
df = pd.read_csv('Jira (5).csv', nrows=100)

# Use 'Summary' or 'Description' based on your preference
texts = df['Summary'].fillna('')  # Replace 'Summary' with 'Description' if needed

# Convert text to embeddings using a pre-trained model (e.g., BERT, Sentence-BERT)
# For demonstration, this step is skipped here. Assume `summary_embeddings` is obtained.
# summary_embeddings = ...

# For demonstration, create random embeddings if actual embeddings are not available
summary_embeddings = np.random.rand(len(df), 100)  # Replace with actual embeddings

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(summary_embeddings)

# Apply GMM
n_components = 5  # Number of clusters; adjust as needed
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm_labels = gmm.fit_predict(reduced_embeddings)

# Plot GMM clusters
plt.figure(figsize=(10, 5))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=gmm_labels, cmap='viridis', s=50)
plt.title('GMM Clusters')
plt.colorbar()
plt.show()

from sklearn.mixture import GaussianMixture

n_components_range = range(1, 11)  # Test different numbers of components
bic_scores = []
aic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(reduced_embeddings)
    bic_scores.append(gmm.bic(reduced_embeddings))
    aic_scores.append(gmm.aic(reduced_embeddings))

# Plot BIC and AIC scores
plt.figure(figsize=(12, 6))
plt.plot(n_components_range, bic_scores, label='BIC')
plt.plot(n_components_range, aic_scores, label='AIC')
plt.xlabel('Number of Components')
plt.ylabel('Score')
plt.title('Model Selection Criteria')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# Load the first 100 rows of the Jira CSV file
df = pd.read_csv('Jira (5).csv', nrows=100)

# Use 'Summary' or 'Description' based on your preference
texts = df['Summary'].fillna('')  # Replace 'Summary' with 'Description' if needed

# Convert text to embeddings using a pre-trained model (e.g., BERT, Sentence-BERT)
# For demonstration, this step is skipped here. Assume `summary_embeddings` is obtained.
# summary_embeddings = ...

# For demonstration, create random embeddings if actual embeddings are not available
summary_embeddings = np.random.rand(len(df), 100)  # Replace with actual embeddings

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(summary_embeddings)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(reduced_embeddings)
kmeans_silhouette = silhouette_score(reduced_embeddings, kmeans_labels)

# Mean Shift Clustering
mean_shift = MeanShift()
mean_shift_labels = mean_shift.fit_predict(reduced_embeddings)
mean_shift_silhouette = silhouette_score(reduced_embeddings, mean_shift_labels)

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=5)
agg_labels = agg_clustering.fit_predict(reduced_embeddings)
agg_silhouette = silhouette_score(reduced_embeddings, agg_labels)

# GMM Clustering
n_components = 5
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm_labels = gmm.fit_predict(reduced_embeddings)
gmm_silhouette = silhouette_score(reduced_embeddings, gmm_labels)
gmm_bic = gmm.bic(reduced_embeddings)
gmm_aic = gmm.aic(reduced_embeddings)

# Plot K-Means clusters
plt.figure(figsize=(20, 15))

plt.subplot(2, 3, 1)
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title(f'K-Means Clusters\nSilhouette Score: {kmeans_silhouette:.2f}')
plt.colorbar()

# Plot Mean Shift clusters
plt.subplot(2, 3, 2)
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=mean_shift_labels, cmap='viridis', s=50)
plt.title(f'Mean Shift Clusters\nSilhouette Score: {mean_shift_silhouette:.2f}')
plt.colorbar()

# Plot Agglomerative Clustering
plt.subplot(2, 3, 3)
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=agg_labels, cmap='viridis', s=50)
plt.title(f'Agglomerative Clustering\nSilhouette Score: {agg_silhouette:.2f}')
plt.colorbar()

# Plot GMM clusters
plt.subplot(2, 3, 4)
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=gmm_labels, cmap='viridis', s=50)
plt.title(f'GMM Clusters\nSilhouette Score: {gmm_silhouette:.2f}')
plt.colorbar()

# Plot Dendrogram for Agglomerative Clustering
plt.subplot(2, 3, 5)
plt.title('Dendrogram')
dendrogram = sch.dendrogram(sch.linkage(summary_embeddings, method='ward'))

plt.tight_layout()
plt.show()

# Print scores
print(f'K-Means Silhouette Score: {kmeans_silhouette:.2f}')
print(f'Mean Shift Silhouette Score: {mean_shift_silhouette:.2f}')
print(f'Agglomerative Clustering Silhouette Score: {agg_silhouette:.2f}')
print(f'GMM Silhouette Score: {gmm_silhouette:.2f}')
print(f'GMM BIC: {gmm_bic:.2f}')
print(f'GMM AIC: {gmm_aic:.2f}')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Vectorize the summaries
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(summaries)

# Fit LDA model
lda = LatentDirichletAllocation(n_components=n_clusters, random_state=42)
lda_topics = lda.fit_transform(X)

# Display topics
for topic_idx, topic in enumerate(lda.components_):
    print(f"\nTopic {topic_idx}:")
    print(" ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]))

pip install umap-learn

from sklearn.manifold import TSNE
import umap

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(summary_embeddings)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=cluster_labels, palette='viridis')
plt.title('t-SNE Visualization of Clusters')
plt.show()

# UMAP visualization
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_results = umap_model.fit_transform(summary_embeddings)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=umap_results[:, 0], y=umap_results[:, 1], hue=cluster_labels, palette='viridis')
plt.title('UMAP Visualization of Clusters')
plt.show()

from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(summary_embeddings)
distances, indices = neighbors_fit.kneighbors(summary_embeddings)

# Plot histogram of distances to the nearest neighbor
plt.hist(distances[:, 1], bins=50)
plt.title('Histogram of Distances to Nearest Neighbor')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()

from transformers import BartTokenizer, BartForConditionalGeneration

# Load BART model and tokenizer
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

def generate_summary(text):
    inputs = bart_tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Generate summaries for each cluster
for i in range(n_clusters):
    cluster_texts = " ".join([summaries[j] for j in range(len(summaries)) if cluster_labels[j] == i])
    print(f"\nCluster {i} Generalized Summary:")
    print(generate_summary(cluster_texts))

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the Jira CSV file
df = pd.read_csv('Jira (5).csv')

# Convert date columns to datetime
df['create'] = pd.to_datetime(df['Created'])
df['update'] = pd.to_datetime(df['Updated'])
df['last view'] = pd.to_datetime(df['Last Viewed'])

# Calculate time to resolve in hours
df['time_to_resolve'] = (df['update'] - df['create']).dt.total_seconds() / (60 * 60)

# Ensure text data is present for clustering
summaries = df['Summary'].dropna().tolist()

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(summaries)

# Perform KMeans clustering
n_clusters = 9
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Add cluster labels to DataFrame
df['Cluster'] = cluster_labels

# Calculate cluster metrics
cluster_metrics = df.groupby('Cluster').agg(
    issue_count=('Summary', 'size'),
    avg_resolution_time=('time_to_resolve', 'mean')
).reset_index()

# Identify bottleneck clusters
bottleneck_cluster = cluster_metrics.sort_values(by='avg_resolution_time', ascending=False).iloc[0]

print(f"Bottleneck Cluster:")
print(f"Cluster ID: {bottleneck_cluster['Cluster']}")
print(f"Issue Count: {bottleneck_cluster['issue_count']}")
print(f"Average Resolution Time (hours): {bottleneck_cluster['avg_resolution_time']}")

# Output details of bottleneck cluster
bottleneck_issues = df[df['Cluster'] == bottleneck_cluster['Cluster']]
print("\nDetails of Bottleneck Issues:")
print(bottleneck_issues[['Summary', 'time_to_resolve']])

pip install --upgrade networkx plotly

pip install networkx plotly

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('Jira (5).csv')


df['create'] = pd.to_datetime(df['Created'])
df['update'] = pd.to_datetime(df['Updated'])
df['time_to_resolve'] = (df['update'] - df['create']).dt.total_seconds() / (60 * 60)


summaries = df['Summary'].dropna().tolist()

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(summaries)


n_clusters = 9
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)


df['Cluster'] = clusters


print(df.head())


cluster_metrics = df.groupby('Cluster').agg(
    issue_count=('Summary', 'size'),
    avg_resolution_time=('time_to_resolve', 'mean')
).reset_index()

# Display the metrics
print(cluster_metrics)

# Find the bottleneck cluster
bottleneck_cluster = cluster_metrics.sort_values(by='avg_resolution_time', ascending=False).iloc[0]['Cluster']
print(f"Bottleneck Cluster: {bottleneck_cluster}")

# Output details of the bottleneck cluster
bottleneck_details = df[df['Cluster'] == bottleneck_cluster]
print("\nDetails of Bottleneck Cluster:")
print(bottleneck_details[['Summary', 'time_to_resolve']])

import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# Load the Jira CSV file
df = pd.read_csv('Jira (5).csv')

# Convert date columns to datetime
df['create'] = pd.to_datetime(df['Created'])
df['update'] = pd.to_datetime(df['Updated'])
df['time_to_resolve'] = (df['update'] - df['create']).dt.total_seconds() / (60 * 60)

# Ensure text data is present for clustering
summaries = df['Summary'].dropna().tolist()

# Convert text data into numerical features using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(summaries)

#  KMeans clustering
from sklearn.cluster import KMeans
n_clusters = 9
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Find bottleneck cluster
bottleneck_cluster = df.groupby('Cluster').agg(
    avg_resolution_time=('time_to_resolve', 'mean')
).reset_index().sort_values(by='avg_resolution_time', ascending=False).iloc[0]['Cluster']

print(f"Bottleneck Cluster: {bottleneck_cluster}")

# Output details of the bottleneck cluster
print("\nDetails of Bottleneck Cluster:")
bottleneck_details = df[df['Cluster'] == bottleneck_cluster]
print(bottleneck_details[['Summary', 'time_to_resolve', 'Status', 'Priority']])

# Create a graph for bottleneck cluster
G = nx.Graph()

# Add nodes for issues in the bottleneck cluster
for _, issue in bottleneck_details.iterrows():
    G.add_node(issue['Issue key'], label=issue['Summary'], time_to_resolve=issue['time_to_resolve'])

# Add edges (customize based on available relationships)
# Here we add dummy edges between all nodes in the bottleneck cluster
for i, node1 in bottleneck_details.iterrows():
    for j, node2 in bottleneck_details.iterrows():
        if i != j:
            G.add_edge(node1['Issue key'], node2['Issue key'])


pos = nx.spring_layout(G, seed=42)

#  for Plotly
node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_y.append(y0)
    edge_y.append(y1)

# Create the Plotly figure
fig = go.Figure()

# Add edges
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    mode='lines',
    line=dict(width=0.5, color='#888')
))

# Add nodes
fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=[G.nodes[node]['label'] for node in G.nodes()],
    marker=dict(size=10, color='blue')
))

fig.update_layout(
    title='Knowledge Graph of Bottleneck Issues',
    showlegend=False
)

fig.show()

# Additional steps to review dependencies and priorities
# For demonstration, assuming there are columns for issue links in your dataset
# You would need to adjust this part based on actual data
if 'Linked Issues' in df.columns:
    # Extract and analyze issue linkages within the bottleneck cluster
    linkages = df[df['Cluster'] == bottleneck_cluster]
    # Example: create and analyze linkage graph or other dependency-related analysis
    # This part will depend on how linkages are stored and should be customized accordingly
    pass
