import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers = pd.read_csv('DataFile/Customers.csv', parse_dates=['SignupDate'])
transactions = pd.read_csv('DataFile/Transactions.csv', parse_dates=['TransactionDate'])

# Merge customers and transactions
merged_data = pd.merge(customers, transactions, on='CustomerID', how='inner')

# Aggregate transaction data for each customer
customer_features = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',  # Total spending
    'TransactionID': 'count',  # Number of transactions
    'Price': 'mean',  # Average transaction value
    'Region': 'first',  # Region of customer
    'SignupDate': 'first'  # Signup date
}).reset_index()

# Convert SignupDate to numerical format (days since minimum date)
customer_features['SignupDate'] = (customer_features['SignupDate'] - customer_features['SignupDate'].min()).dt.days

# One-hot encode categorical features (Region)
encoder = OneHotEncoder()
region_encoded = encoder.fit_transform(customer_features[['Region']]).toarray()

# Combine numerical and one-hot encoded features
numerical_features = customer_features[['TotalValue', 'TransactionID', 'Price', 'SignupDate']]
feature_matrix = np.hstack([numerical_features.values, region_encoded])

# Normalize features
scaler = MinMaxScaler()
feature_matrix_scaled = scaler.fit_transform(feature_matrix)

# Perform clustering (K-Means)
optimal_clusters = None
db_index_values = []
silhouette_values = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_matrix_scaled)

    # Evaluate clustering
    db_index = davies_bouldin_score(feature_matrix_scaled, cluster_labels)
    silhouette_avg = silhouette_score(feature_matrix_scaled, cluster_labels)

    db_index_values.append(db_index)
    silhouette_values.append(silhouette_avg)

    print(f"Clusters: {k}, DB Index: {db_index:.4f}, Silhouette Score: {silhouette_avg:.4f}")

    if optimal_clusters is None or db_index < min(db_index_values):
        optimal_clusters = (k, db_index)

# Choose optimal number of clusters
k_optimal, db_optimal = optimal_clusters
print(f"Optimal Clusters: {k_optimal}, DB Index: {db_optimal:.4f}")

# Fit final model
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
final_labels = kmeans.fit_predict(feature_matrix_scaled)

# Add cluster labels to customer data
customer_features['Cluster'] = final_labels

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(feature_matrix_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=final_labels, palette='viridis', s=100)
plt.title('Customer Segmentation (PCA Projection)')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend(title='Cluster')
plt.show()

# Save clustering results
# output_path = '/mnt/data/Customer_Clusters.csv'
output_path = "Data/Customer_Clusters.csv"

customer_features.to_csv(output_path, index=False)

print(f"Clustering results saved to {output_path}")
