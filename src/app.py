from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import traceback
import shap
import lime
import lime.lime_tabular
import plotly.express as px
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
# Add this after your imports
import numpy as np
np.bool = bool  # Map np.bool to Python's built-in bool

app = Flask(__name__)

# Load models and data
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR.parent / 'models'
DATA_DIR = BASE_DIR.parent / 'data'

# Load models
clinical_model = joblib.load(MODELS_DIR / 'clinical_model.pkl')
symptoms_model = joblib.load(MODELS_DIR / 'symptoms_model.pkl')


# Update your CLINICAL_FEATURE_MAP to include pregnancies
CLINICAL_FEATURE_MAP = {
    'pregnancies': 'Pregnancies',
    'age': 'Age',
    'gender': 'Gender',
    'polydipsia': 'Polydipsia',
    'polyuria': 'Polyuria',
    'sudden_weight_loss': 'Sudden weight loss',
    'weakness': 'Weakness',
    'polyphagia': 'Polyphagia',
    'genital_thrush': 'Genital thrush',
    'visual_blurring': 'Visual blurring',
    'itching': 'Itching',
    'irritability': 'Irritability',
    'delayed_healing': 'Delayed healing',
    'partial_paresis': 'Partial paresis',
    'muscle_stiffness': 'Muscle stiffness',
    'alopecia': 'Alopecia',
    'obesity': 'Obesity',
    # Add any other features your form sends
}
SYMPTOMS_FEATURE_MAP = {
    'gender': 'Gender',
    'polyuria': 'Polyuria',
    'polydipsia': 'Polydipsia',
    'sudden_weight_loss': 'Sudden weight loss',
    'weakness': 'Weakness',
    'polyphagia': 'Polyphagia',
    'genital_thrush': 'Genital thrush',
    'visual_blurring': 'Visual blurring',
    'itching': 'Itching',
    'irritability': 'Irritability',
    'delayed_healing': 'Delayed healing',
    'partial_paresis': 'Partial paresis',
    'muscle_stiffness': 'Muscle stiffness',
    'alopecia': 'Alopecia',
    'obesity': 'Obesity',
    # Add any other symptoms features your form sends
}

def prepare_input(form_data, feature_mapping, expected_features):
    """Convert form data to model-ready input"""
    model_input = {}
    for form_feature, value in form_data.items():
        model_feature = feature_mapping.get(form_feature, form_feature)
        if model_feature not in expected_features:
            raise ValueError(f"Unexpected feature: {model_feature}")
        model_input[model_feature] = float(value)
    
    # Return array in correct feature order
    return [model_input[feature] for feature in expected_features]



# Load training data for explainers
train_data = pd.read_csv(DATA_DIR / 'processed/diabetes_clean.csv')
X_train = train_data.drop('Outcome', axis=1)

# Initialize explainers
clinical_explainer = shap.TreeExplainer(clinical_model)
symptoms_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['No Diabetes', 'Diabetes'],
    mode='classification'
)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/clinical')
def clinical_form():
    return render_template('clinical_form.html')

@app.route('/symptoms')
def symptoms_form():
    return render_template('symptoms_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = clinical_model.predict([data])[0]
    probability = clinical_model.predict_proba([data])[0][1]
    
    explanation = generate_explanations(data, 'clinical')
    
    return render_template('clinical_result.html',
                        prediction=prediction,
                        probability=f"{probability:.0%}",
                        explanation=explanation)


@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    try:
        # Debug: Print raw form data and model expectations
        print("\n=== DEBUG INFO ===")
        print("Model expects features:", symptoms_model.feature_names_in_)
        print("Raw form data:", request.form)
        
        # Get form data with proper error handling
        form_data = {}
        for key, value in request.form.items():
            try:
                # Clean and convert values
                cleaned_value = value.strip()
                if not cleaned_value:  # Handle empty strings
                    form_data[key] = 0.0
                    print(f"Warning: Empty value for {key}, using 0.0")
                else:
                    form_data[key] = float(cleaned_value)
            except ValueError as ve:
                print(f"Error converting value '{value}' for field '{key}': {ve}")
                form_data[key] = 0.0  # Default value if conversion fails
        
        print("\nProcessed form data:", form_data)
        
        # Prepare input data with feature mapping
        input_data = []
        missing_features = []
        invalid_features = []
        
        for feature in symptoms_model.feature_names_in_:
            # Find matching form field with case-insensitive comparison
            matched_field = None
            for form_field in form_data:
                if (form_field.lower() == feature.lower() or 
                    SYMPTOMS_FEATURE_MAP.get(form_field, "").lower() == feature.lower()):
                    matched_field = form_field
                    break
            
            if matched_field:
                input_data.append(form_data[matched_field])
            else:
                missing_features.append(feature)
                input_data.append(0.0)  # Default value
                print(f"Warning: No match found for model feature '{feature}'")
        
        # Log any issues
        if missing_features:
            print("\nMissing features (using defaults):", missing_features)
        
        # Create DataFrame ensuring correct feature order
        input_df = pd.DataFrame([input_data], columns=symptoms_model.feature_names_in_)
        print("\nFinal input data for model:")
        print(input_df)
        
        # Make prediction
        prediction = symptoms_model.predict(input_df)[0]
        probability = symptoms_model.predict_proba(input_df)[0][1]
        
        # Generate explanations with error handling
        try:
            explanation = generate_explanations(input_df.values[0], 'symptoms')
        except Exception as e:
            print(f"\nExplanation generation failed: {str(e)}")
            explanation = {
                'shap': None,
                'lime': "<div class='alert alert-warning'>Explanation unavailable</div>",
                'importance': "<div class='alert alert-warning'>Feature importance unavailable</div>"
            }
        
        print("\n=== PREDICTION RESULTS ===")
        print(f"Prediction: {prediction}, Probability: {probability:.2%}")
        
        return render_template('symptoms_result.html',
                            prediction=prediction,
                            probability=f"{probability:.0%}",
                            explanation=explanation)
        
    except Exception as e:
        print(f"\n!!! CRITICAL ERROR !!!")
        print(f"Error details: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return render_template('error.html', 
                            error_message="We encountered an error processing your request. Please ensure all fields are filled correctly and try again."), 400
                          

def generate_explanations(input_data, model_type):
    if model_type == 'clinical':
        # SHAP values
        shap_values = clinical_explainer.shap_values(np.array(input_data).reshape(1, -1))
        plt.figure()
        shap.summary_plot(shap_values, feature_names=X_train.columns, plot_type='bar', show=False)
        plt.tight_layout()
        shap_path = 'static/shap_explanation.png'
        plt.savefig(shap_path)
        plt.close()
        
        # LIME explanation
        exp = symptoms_explainer.explain_instance(
            np.array(input_data),
            clinical_model.predict_proba,
            num_features=5
        )
        lime_html = exp.as_html()
    else:
        # SHAP values for symptoms model
        shap_values = clinical_explainer.shap_values(np.array(input_data).reshape(1, -1))
        plt.figure()
        shap.summary_plot(shap_values, feature_names=X_train.columns, plot_type='bar', show=False)
        plt.tight_layout()
        shap_path = 'static/shap_explanation.png'
        plt.savefig(shap_path)
        plt.close()
        
        # LIME explanation
        exp = symptoms_explainer.explain_instance(
            np.array(input_data),
            symptoms_model.predict_proba,
            num_features=5
        )
        lime_html = exp.as_html()
    
    # Feature importance
    importance = clinical_model.feature_importances_ if model_type == 'clinical' else symptoms_model.feature_importances_
    
    fig = px.bar(
        x=X_train.columns,
        y=importance,
        labels={'x': 'Feature', 'y': 'Importance'},
        title='Feature Importance'
    )
    importance_html = fig.to_html(full_html=False)
    
    return {
        'shap': shap_path,
        'lime': lime_html,
        'importance': importance_html
    }

@app.route('/what_if', methods=['POST'])
def what_if_analysis():
    data = request.get_json()
    new_data = np.array([float(x) for x in data['values']])
    
    original_pred = clinical_model.predict_proba([new_data])[0][1]
    modified_data = new_data.copy()
    modified_data[data['feature_idx']] = data['new_value']
    modified_pred = clinical_model.predict_proba([modified_data])[0][1]
    
    return jsonify({
        'original': original_pred,
        'modified': modified_pred,
        'difference': modified_pred - original_pred
    })

@app.route('/debug-routes')
def debug_routes():
    return str([rule.rule for rule in app.url_map.iter_rules()])

if __name__ == '__main__':
    app.run(debug=True)  