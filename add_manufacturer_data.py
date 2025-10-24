import pandas as pd
import numpy as np

# Load existing CSV
df = pd.read_csv("fixed_pas_forecasting_data.csv")

# Add manufacturer data
np.random.seed(42)  # For consistent results
manufacturers = ["Pfizer", "Johnson & Johnson", "Merck", "Novartis", "Roche"]

# Assign manufacturers randomly but consistently per entity_id
df['manufacturer_id'] = df['entity_id'] % 5 + 1
df['manufacturer_name'] = df['manufacturer_id'].map({
    1: "Pfizer",
    2: "Johnson & Johnson", 
    3: "Merck",
    4: "Novartis",
    5: "Roche"
})

# Save updated CSV
df.to_csv("fixed_pas_forecasting_data.csv", index=False)
print("✅ Added manufacturer columns to CSV")
print(f"Manufacturers: {df['manufacturer_name'].unique()}")