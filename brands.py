import pandas as pd
import os

# You already have this
import kagglehub
path = kagglehub.dataset_download("openfoodfacts/world-food-facts")

# List files to find correct one
print(os.listdir(path))

# Load the main file (usually products.csv or similar)
file_path = os.path.join(path, "en.openfoodfacts.org.products.tsv")  # Adjust if needed
df = pd.read_csv(file_path, sep='\t', low_memory=False)

print(df.columns)  # preview available columns
# Drop rows with missing 'countries_en'
df = df[df['countries_en'].notna()]

# Keep only products sold in Israel
df_israel = df[df['countries_en'].str.contains("Israel", case=False)]

# Preview brands
brands = df_israel['brands'].dropna().unique()
print("Sample brands in Israel:", brands[:20])
df_brands = pd.DataFrame({"brand": brands})
df_brands.to_csv("israeli_food_brands_from_openfoodfacts.csv", index=False)
print("✅ Saved to israeli_food_brands_from_openfoodfacts.csv")
