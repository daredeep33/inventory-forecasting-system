# Test.py
import pandas as pd
from modules.data_loader import load_sales_data, load_current_stock

sales = load_sales_data('./data/sales.xml')
print(f"Loaded {len(sales)} sales records")

stock = load_current_stock('./data/godowns')
print(f"Loaded {len(stock)} inventory entries")

print("\n=== Sales Data ===")
print(sales.info())
print("Missing values:\n", sales.isnull().sum())

print("\n=== Inventory Data ===")
print(f"Unique products: {len(set(p for p,_ in stock))}")
print(f"Godowns: {len(set(g for _,g in stock))}")

top_products = sales['Product'].value_counts().head(10)
top_products.plot(kind='barh', title='Top 10 Selling Products')

# Convert inventory dict to DataFrame
stock_df = pd.Series(stock).reset_index()
stock_df.columns = ['Product', 'Godown', 'Quantity']

# Merge with sales data
merged = sales.merge(stock_df, on=['Product', 'Godown'], suffixes=('_sold', '_current'))
merged['Stock_Throughput'] = merged['Quantity_sold'] / merged['Quantity_current']