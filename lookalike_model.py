from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Load datasets
customers = pd.read_csv('DataFile/Customers.csv', parse_dates=['SignupDate'])
products = pd.read_csv('DataFile/Products.csv')
transactions = pd.read_csv('DataFile/Transactions.csv', parse_dates=['TransactionDate'])

# Merge datasets
transactions_with_products = pd.merge(transactions, products, on='ProductID', how='inner')
merged_data = pd.merge(customers, transactions_with_products, on='CustomerID', how='inner')

# Filter for the first 20 customers (C0001 - C0020)
target_customers = customers[customers['CustomerID'].isin([f"C{str(i).zfill(4)}" for i in range(1, 21)])]

# Prepare feature set by merging datasets
customer_features = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',  # Total spending
    'TransactionID': 'count',  # Number of transactions
    'Category': lambda x: ','.join(x),  # Categories purchased
    'Region': 'first',  # Region of customer
    'SignupDate': 'first'  # Signup date
}).reset_index()

# One-hot encode categorical features (Region and Categories)
encoder = OneHotEncoder()
region_encoded = encoder.fit_transform(customer_features[['Region']]).toarray()

# Convert SignupDate to numerical format (days since minimum date)
customer_features['SignupDate'] = (customer_features['SignupDate'] - customer_features['SignupDate'].min()).dt.days

# Create a binary matrix for categories purchased
categories = customer_features['Category'].str.get_dummies(sep=',')
customer_features = customer_features.drop('Category', axis=1)
feature_matrix = np.hstack([
    customer_features[['TotalValue', 'TransactionID', 'SignupDate']].values,  # Numeric features
    region_encoded,  # Region features
    categories.values  # Categories binary matrix
])

# Scale feature matrix
scaler = MinMaxScaler()
feature_matrix_scaled = scaler.fit_transform(feature_matrix)

# Compute cosine similarity
similarity_matrix = cosine_similarity(feature_matrix_scaled)

# Generate recommendations for the first 20 customers
recommendations = {}
for idx, customer_id in enumerate(target_customers['CustomerID']):
    # Get similarity scores for the current customer
    scores = similarity_matrix[idx]
    similar_customers = sorted(
        [(other_id, score) for other_id, score in zip(customer_features['CustomerID'], scores) if other_id != customer_id],
        key=lambda x: x[1],
        reverse=True
    )[:3]  # Top 3 similar customers
    recommendations[customer_id] = similar_customers

# Create Lookalike.csv
lookalike_df = pd.DataFrame({
    "CustomerID": recommendations.keys(),
    "Recommendations": [str(rec) for rec in recommendations.values()]
})

output_path = 'Data/Lookalike.csv'
lookalike_df.to_csv(output_path, index=False)

print(f"Lookalike.csv has been saved at {output_path}")
