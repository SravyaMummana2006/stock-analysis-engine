from data_fetch import fetch_stock_data
import os

os.makedirs("data", exist_ok=True)

df = fetch_stock_data("RELIANCE.NS")
df.to_csv("data/RELIANCE_NS.csv")

print("✅ Saved successfully")