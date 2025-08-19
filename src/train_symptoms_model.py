import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
import sys

def train_symptoms_model():
    try:
        # ===== Configuration =====
        processed_path = Path('../data/symptoms_processed/diabetes_symptoms_clean.csv')
        model_path = Path('../models/symptoms_model.pkl')
        RANDOM_STATE = 42
        TEST_SIZE = 0.2

        # ===== Explicit List of All Symptoms/Features =====
        EXPECTED_SYMPTOMS = [
            'Age',                      # Patient age in years
            'Gender',                   # Biological sex (0: Female, 1: Male)
            'Polyuria',                 # Excessive urination (0: No, 1: Yes)
            'Polydipsia',               # Excessive thirst (0: No, 1: Yes)
            'Sudden weight loss',       # Unexplained weight loss (0: No, 1: Yes)
            'Weakness',                 # General weakness (0: No, 1: Yes)
            'Polyphagia',               # Excessive hunger (0: No, 1: Yes)
            'Genital thrush',           # Yeast infection (0: No, 1: Yes)
            'Visual blurring',          # Blurred vision (0: No, 1: Yes)
            'Itching',                  # Persistent itching (0: No, 1: Yes)
            'Irritability',             # Mood irritability (0: No, 1: Yes)
            'Delayed healing',          # Slow wound healing (0: No, 1: Yes)
            'Partial paresis',          # Muscle weakness (0: No, 1: Yes)
            'Muscle stiffness',         # Stiff muscles (0: No, 1: Yes)
            'Alopecia',                 # Hair loss (0: No, 1: Yes)
            'Obesity'                   # BMI ≥ 30 (0: No, 1: Yes)
        ]

        # ===== Data Loading =====
        print("\nLoading data...")
        try:
            df = pd.read_csv(processed_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            
            # Standardize column names (strip whitespace and capitalize first letters)
            df.columns = df.columns.str.strip().str.title().str.replace(' ', '')
            print("\nStandardized columns:", df.columns.tolist())
            
            # Verify all expected symptoms are present (case insensitive)
            missing_symptoms = [
                symptom for symptom in EXPECTED_SYMPTOMS 
                if symptom.replace(' ', '').lower() not in [
                    col.replace(' ', '').lower() for col in df.columns
                ]
            ]
            
            if missing_symptoms:
                raise ValueError(
                    f"Missing expected symptoms in data: {missing_symptoms}\n"
                    f"Data contains: {df.columns.tolist()}"
                )
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {processed_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

        # ===== Data Preparation =====
        print("\nPreparing data...")
        
        # Map columns to expected names (case insensitive)
        column_mapping = {
            col: next(
                s for s in EXPECTED_SYMPTOMS 
                if s.replace(' ', '').lower() == col.replace(' ', '').lower()
            )
            for col in df.columns
            if col.replace(' ', '').lower() in [
                s.replace(' ', '').lower() for s in EXPECTED_SYMPTOMS
            ]
        }
        
        df = df.rename(columns=column_mapping)
        X = df[EXPECTED_SYMPTOMS]  # Ensure correct column order
        y = df['Class']
        
        # Ensure we have features and targets
        if X.empty or y.empty:
            raise ValueError("Empty features or target after data preparation")

        # ===== Train-Test Split =====
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            stratify=y
        )

        # ===== Model Training =====
        print("\nTraining model...")
        model = RandomForestClassifier(
            n_estimators=200,           # Increased from 100 for better performance
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1,
            max_depth=10,               # Added to prevent overfitting
            min_samples_split=5         # Added to prevent overfitting
        )
        
        model.fit(X_train, y_train)

        # ===== Evaluation =====
        print("\nEvaluating model...")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        print("\n=== Training Metrics ===")
        print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
        print(classification_report(y_train, y_train_pred))

        print("\n=== Test Metrics ===")
        print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print(classification_report(y_test, y_test_pred))

        # ===== Save Model =====
        print("\nSaving model...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': model,
            'feature_names': EXPECTED_SYMPTOMS,
            'feature_descriptions': {
                'Age': 'Patient age in years (continuous)',
                'Gender': 'Biological sex (0: Female, 1: Male)',
                'Polyuria': 'Excessive urination (0: No, 1: Yes)',
                'Polydipsia': 'Excessive thirst (0: No, 1: Yes)',
                'Sudden weight loss': 'Unexplained weight loss (0: No, 1: Yes)',
                'Weakness': 'General weakness (0: No, 1: Yes)',
                'Polyphagia': 'Excessive hunger (0: No, 1: Yes)',
                'Genital thrush': 'Yeast infection (0: No, 1: Yes)',
                'Visual blurring': 'Blurred vision (0: No, 1: Yes)',
                'Itching': 'Persistent itching (0: No, 1: Yes)',
                'Irritability': 'Mood irritability (0: No, 1: Yes)',
                'Delayed healing': 'Slow wound healing (0: No, 1: Yes)',
                'Partial paresis': 'Muscle weakness (0: No, 1: Yes)',
                'Muscle stiffness': 'Stiff muscles (0: No, 1: Yes)',
                'Alopecia': 'Hair loss (0: No, 1: Yes)',
                'Obesity': 'BMI ≥ 30 (0: No, 1: Yes)'
            },
            'target_names': ['No Diabetes', 'Diabetes'],
            'required_features': EXPECTED_SYMPTOMS,
            'training_metrics': classification_report(y_train, y_train_pred, output_dict=True),
            'test_metrics': classification_report(y_test, y_test_pred, output_dict=True),
            'random_state': RANDOM_STATE,
            'model_type': 'RandomForestClassifier',
            'model_parameters': model.get_params()
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model successfully saved to {model_path}")
        
        # Print feature importance
        print("\n=== Feature Importance ===")
        for name, importance in sorted(zip(EXPECTED_SYMPTOMS, model.feature_importances_), 
                                     key=lambda x: x[1], reverse=True):
            print(f"{name}: {importance:.4f}")
        
        return model_data

    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    train_symptoms_model()