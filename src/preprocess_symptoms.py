import pandas as pd
import os
from pathlib import Path

def preprocess_data():
    # Define paths using Path for cross-platform compatibility
    raw_path = Path('../data/symptoms_raw/diabetes_symptoms.csv')
    processed_dir = Path('../data/symptoms_processed/')
    processed_path = processed_dir / 'diabetes_symptoms_clean.csv'
    
    try:
        # 1. Create processed directory if it doesn't exist
        processed_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {processed_dir}")

        # 2. Load data
        df = pd.read_csv(raw_path)
        print("✓ Data loaded successfully")

        # 3. Preprocessing steps (same as before)
        symptom_columns = [
            'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
            'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
            'Irritability', 'delayed healing', 'partial paresis',
            'muscle stiffness', 'Alopecia', 'Obesity'
        ]
        
        for col in symptom_columns:
            df[col] = df[col].str.strip().str.lower().map({'yes': 1, 'no': 0})

        df['class'] = df['class'].str.strip().str.lower().map({
            'positive': 1, 'pos': 1, '1': 1, 'yes': 1,
            'negative': 0, 'neg': 0, '0': 0, 'no': 0
        })

        df['Gender'] = df['Gender'].str.strip().str.lower().map({'male': 1, 'female': 0})
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median())

        # 4. Save processed data
        df.to_csv(processed_path, index=False)
        print(f"✓ Cleaned data saved to: {processed_path}")
        
        return True

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

if __name__ == '__main__':
    if preprocess_data():
        print("✅ Preprocessing completed successfully!")
    else:
        print("❌ Preprocessing failed")
        exit(1)