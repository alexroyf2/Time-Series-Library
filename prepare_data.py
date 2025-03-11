import pandas as pd
import os

# Read the data
df = pd.read_csv('Historical_Data.csv')

# Create a date column by combining Year and Month
df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
df['date'] = df['date'].dt.strftime('%Y-%m-%d')  # Format date as string

# Drop Year and Month columns as they'll be encoded by the model
df = df.drop(['Year', 'Month'], axis=1)

# Reorder columns to have date first, then features, then target (Quantity_Sold)
cols = ['date'] + [col for col in df.columns if col not in ['date', 'Quantity_Sold']] + ['Quantity_Sold']
df = df[cols]

# Handle any missing data
df = df.dropna()

# Ensure the dataset directory exists
os.makedirs('./dataset/custom', exist_ok=True)

# Save to the dataset directory
df.to_csv('./dataset/custom/sales_data.csv', index=False)

print(f"Data prepared and saved to './dataset/custom/sales_data.csv'")
print(f"Number of records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}") 