import pandas as pd
import os

# Get absolute path to the raw data
current_dir = os.path.dirname(__file__)
raw_path = os.path.join(current_dir, 'data', 'raw', 'diabetes.csv')

# Load and clean data
df = pd.read_csv(raw_path)
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, df[cols].median())

# Ensure processed directory exists
os.makedirs(os.path.join(current_dir, 'data', 'processed'), exist_ok=True)

# Save cleaned data
clean_path = os.path.join(current_dir, 'data', 'processed', 'diabetes_clean.csv')
df.to_csv(clean_path, index=False)
print(f"Clean data saved to: {clean_path}")