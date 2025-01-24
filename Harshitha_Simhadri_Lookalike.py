import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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
    'Category': lambda x: x.value_counts().idxmax()  # Most frequent category
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

similarity_matrix = cosine_similarity(customer_features.drop('CustomerID', axis=1))
similarity_df = pd.DataFrame(similarity_matrix, index=customer_features['CustomerID'], columns=customer_features['CustomerID'])

lookalike_data = {}
for customer_id in customer_features['CustomerID'][:20]:
    similar_customers = similarity_df[customer_id].sort_values(ascending=False)[1:4]  # Exclude self-similarity
    lookalike_data[customer_id] = [(idx, round(score, 4)) for idx, score in similar_customers.items()]

lookalike_df = pd.DataFrame({
    'CustomerID': lookalike_data.keys(),
    'Lookalikes': [str(value) for value in lookalike_data.values()]
})

lookalike_df.to_csv("Lookalike.csv", index=False)

print("Lookalike recommendations for first 20 customers saved to 'Lookalike.csv'")
