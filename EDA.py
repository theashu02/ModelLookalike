import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers = pd.read_csv('DataFile/Customers.csv', parse_dates=['SignupDate'])
products = pd.read_csv('DataFile/Products.csv')
transactions = pd.read_csv('DataFile/Transactions.csv', parse_dates=['TransactionDate'])

# Merge datasets
merged_data = pd.merge(transactions, customers, on='CustomerID', how='inner')
merged_data = pd.merge(merged_data, products, on='ProductID', how='inner')

# 1. Summary statistics
print("\nSummary Statistics:")
print(merged_data.describe())

# 2. Data distributions
plt.figure(figsize=(10, 6))
sns.histplot(merged_data['TotalValue'], bins=30, kde=True)
plt.title('Distribution of Total Transaction Values')
plt.xlabel('Total Value')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Region', data=customers, order=customers['Region'].value_counts().index)
plt.title('Customer Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Number of Customers')
plt.show()

# 3. Top-performing products
top_products = merged_data.groupby('ProductName')['TotalValue'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Performing Products:")
print(top_products)

plt.figure(figsize=(10, 6))
top_products.plot(kind='bar', color='skyblue')
plt.title('Top 10 Products by Total Value')
plt.xlabel('Product Name')
plt.ylabel('Total Value')
plt.show()

# 4. Monthly trends in transactions
merged_data['Month'] = merged_data['TransactionDate'].dt.to_period('M')
monthly_trends = merged_data.groupby('Month')['TotalValue'].sum()

plt.figure(figsize=(12, 6))
monthly_trends.plot(kind='line', marker='o')
plt.title('Monthly Transaction Value Trends')
plt.xlabel('Month')
plt.ylabel('Total Transaction Value')
plt.show()

# 5. Average spending by customer region
avg_spending_region = merged_data.groupby('Region')['TotalValue'].mean()
print("\nAverage Spending by Region:")
print(avg_spending_region)

plt.figure(figsize=(10, 6))
avg_spending_region.plot(kind='bar', color='lightgreen')
plt.title('Average Spending by Region')
plt.xlabel('Region')
plt.ylabel('Average Spending')
plt.show()

# Save plots and findings to a report file
output_path = 'Data/EDA_Report.pdf'
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(output_path) as pdf:
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_data['TotalValue'], bins=30, kde=True)
    plt.title('Distribution of Total Transaction Values')
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Region', data=customers, order=customers['Region'].value_counts().index)
    plt.title('Customer Distribution by Region')
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 6))
    top_products.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Products by Total Value')
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(12, 6))
    monthly_trends.plot(kind='line', marker='o')
    plt.title('Monthly Transaction Value Trends')
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 6))
    avg_spending_region.plot(kind='bar', color='lightgreen')
    plt.title('Average Spending by Region')
    pdf.savefig()
    plt.close()

print(f"EDA Report saved to {output_path}")
