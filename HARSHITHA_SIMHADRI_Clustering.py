import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

customers_file = "Customers.csv"
transactions_file = "Transactions.csv"
products_file = "Products.csv"

customers_df = pd.read_csv(customers_file)
transactions_df = pd.read_csv(transactions_file)
products_df = pd.read_csv(products_file)


transactions_products = transactions_df.merge(products_df, on="ProductID", how="left")

customer_features = transactions_products.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'TransactionID': 'count',
    'Category': lambda x: x.value_counts().idxmax()  
}).rename(columns={
    'TotalValue': 'TotalSpent',
    'TransactionID': 'TransactionCount',
    'Category': 'PreferredCategory'
})

customer_features = customer_features.merge(customers_df[['CustomerID', 'Region']], on='CustomerID', how='left')

customer_features = pd.get_dummies(customer_features, columns=['PreferredCategory', 'Region'], drop_first=True)

scaler = MinMaxScaler()
numerical_features = ['TotalSpent', 'TransactionCount']
customer_features[numerical_features] = scaler.fit_transform(customer_features[numerical_features])

clustering_data = customer_features.drop('CustomerID', axis=1)

inertia = []
K = range(2, 11)  
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(clustering_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters', fontsize=16)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.grid(True)
plt.show()

optimal_k = 4  
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_features['Cluster'] = kmeans.fit_predict(clustering_data)

db_index = davies_bouldin_score(clustering_data, customer_features['Cluster'])
print(f"Davies-Bouldin Index: {db_index:.4f}")

clustered_data = customer_features.merge(customers_df, on='CustomerID', how='left')

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=customer_features, x='TotalSpent', y='TransactionCount', hue='Cluster', palette='viridis', s=100
)
plt.title('Customer Segmentation Based on Spending and Transactions', fontsize=16)
plt.xlabel('Total Spent (Normalized)', fontsize=12)
plt.ylabel('Transaction Count (Normalized)', fontsize=12)
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

cluster_summary = clustered_data.groupby('Cluster').agg({
    'TotalSpent': ['mean', 'sum'],
    'TransactionCount': ['mean', 'sum'],
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'CustomerCount'})

print("Cluster Summary:\n", cluster_summary)
