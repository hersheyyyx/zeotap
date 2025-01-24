import pandas as pd
import matplotlib.pyplot as plt

customers_file = "Customers.csv"
products_file = "Products.csv"
transactions_file = "Transactions.csv"

customers_df = pd.read_csv(customers_file)
products_df = pd.read_csv(products_file)
transactions_df = pd.read_csv(transactions_file)

customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

print("Customers Dataset Preview:\n", customers_df.head())
print("Products Dataset Preview:\n", products_df.head())
print("Transactions Dataset Preview:\n", transactions_df.head())

merged_df = transactions_df.merge(customers_df, on='CustomerID', how='left')
merged_df = merged_df.merge(products_df, on='ProductID', how='left')

print("Merged Dataset Preview:\n", merged_df.head())


region_counts = merged_df['Region'].value_counts()
plt.figure(figsize=(8, 6))
region_counts.plot(kind='bar')
plt.title('Customer Distribution by Region', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

top_products = merged_df.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
top_products.plot(kind='barh', color='skyblue')
plt.title('Top 10 Best-Selling Products by Quantity', fontsize=16)
plt.xlabel('Total Quantity Sold', fontsize=12)
plt.ylabel('Product Name', fontsize=12)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

category_sales = merged_df.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 6))
category_sales.plot(kind='bar', color='orange')
plt.title('Total Sales by Product Category', fontsize=16)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Total Sales (USD)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

merged_df['MonthYear'] = merged_df['TransactionDate'].dt.to_period('M')
monthly_sales = merged_df.groupby('MonthYear')['TotalValue'].sum()
plt.figure(figsize=(12, 6))
monthly_sales.plot(marker='o', linestyle='-')
plt.title('Monthly Sales Trends', fontsize=16)
plt.xlabel('Month-Year', fontsize=12)
plt.ylabel('Total Sales (USD)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

customer_clv = merged_df.groupby('CustomerID')['TotalValue'].sum().sort_values(ascending=False)
customer_clv = customer_clv.reset_index().merge(customers_df[['CustomerID', 'CustomerName']], on='CustomerID')
top_customers = customer_clv.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_customers['CustomerName'], top_customers['TotalValue'], color='teal')
plt.title('Top 10 Customers by Lifetime Value', fontsize=16)
plt.xlabel('Total Revenue (USD)', fontsize=12)
plt.ylabel('Customer Name', fontsize=12)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

signup_trends = customers_df['SignupDate'].dt.to_period('M').value_counts().sort_index()
plt.figure(figsize=(12, 6))
signup_trends.plot(marker='o', linestyle='-')
plt.title('Customer Signup Trends Over Time', fontsize=16)
plt.xlabel('Month-Year', fontsize=12)
plt.ylabel('Number of Signups', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

regional_sales = merged_df.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)
regional_category_preferences = merged_df.groupby(['Region', 'Category'])['Quantity'].sum().unstack().idxmax(axis=1)
plt.figure(figsize=(10, 6))
regional_sales.plot(kind='bar', color='coral')
plt.title('Total Sales by Region', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Total Sales (USD)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Regional Category Preferences:\n", regional_category_preferences)
