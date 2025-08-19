import pandas as pd
df = pd.read_csv('../data/symptoms_raw/diabetes_symptoms.csv')
print("Columns in dataset:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))