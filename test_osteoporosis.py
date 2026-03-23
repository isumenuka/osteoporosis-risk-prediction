import warnings
import pytest
import pandas as pd
import os
import numpy as np
from Osteoporosis import load_model_assets, make_prediction, get_feature_names

try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:
    class InconsistentVersionWarning(UserWarning):
        pass

# Suppress scikit-learn version mismatch and streamlit context warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
pytestmark = [
    pytest.mark.filterwarnings("ignore::sklearn.exceptions.InconsistentVersionWarning"),
    pytest.mark.filterwarnings("ignore:.*missing ScriptRunContext.*")
]

# ==============================================================================
# PYTEST CONFIGURATION & FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def model_assets():
    """Load models, encoders, and scaler once for the session."""
    all_models, label_encoders, scaler = load_model_assets()
    return all_models, label_encoders, scaler

# ==============================================================================
# PART 1: MANUAL TEST CASES (17 cases from original test.py)
# ==============================================================================

MANUAL_CASES = [
    ("High Risk Female (Postmenopausal, Smoker, Low Calcium)", {
        'Age': 65, 'Gender': 'Female', 'Hormonal Changes': 'Postmenopausal',
        'Family History': 'Yes', 'Race/Ethnicity': 'Caucasian', 'Body Weight': 'Underweight',
        'Calcium Intake': 'Low', 'Vitamin D Intake': 'Insufficient', 'Physical Activity': 'Sedentary',
        'Smoking': 'Yes', 'Alcohol Consumption': 'Heavy', 'Medical Conditions': 'Rheumatoid Arthritis',
        'Medications': 'Corticosteroids', 'Prior Fractures': 'Yes'
    }, 1),
    ("Low Risk Young Male (Good Nutrition)", {
        'Age': 22, 'Gender': 'Male', 'Hormonal Changes': 'Normal',
        'Family History': 'No', 'Race/Ethnicity': 'Asian', 'Body Weight': 'Normal',
        'Calcium Intake': 'Adequate', 'Vitamin D Intake': 'Sufficient', 'Physical Activity': 'Active',
        'Smoking': 'No', 'Alcohol Consumption': 'None', 'Medical Conditions': 'None',
        'Medications': 'None', 'Prior Fractures': 'No'
    }, 0),
    ("Moderate Risk Female (Elderly but Healthy Habits)", {
        'Age': 70, 'Gender': 'Female', 'Hormonal Changes': 'Postmenopausal',
        'Family History': 'No', 'Race/Ethnicity': 'Asian', 'Body Weight': 'Normal',
        'Calcium Intake': 'Adequate', 'Vitamin D Intake': 'Sufficient', 'Physical Activity': 'Active',
        'Smoking': 'No', 'Alcohol Consumption': 'None', 'Medical Conditions': 'None',
        'Medications': 'None', 'Prior Fractures': 'No'
    }, 1),
    ("High Risk Male (Smoker, Heavy Drinker, Medical Issues)", {
        'Age': 60, 'Gender': 'Male', 'Hormonal Changes': 'Low Testosterone',
        'Family History': 'Yes', 'Race/Ethnicity': 'Caucasian', 'Body Weight': 'Underweight',
        'Calcium Intake': 'Low', 'Vitamin D Intake': 'Insufficient', 'Physical Activity': 'Sedentary',
        'Smoking': 'Yes', 'Alcohol Consumption': 'Heavy', 'Medical Conditions': 'Thyroid Disorders',
        'Medications': 'Corticosteroids', 'Prior Fractures': 'Yes'
    }, 1),
    ("Low Risk Young Female Athlete", {
        'Age': 25, 'Gender': 'Female', 'Hormonal Changes': 'Normal',
        'Family History': 'No', 'Race/Ethnicity': 'Caucasian', 'Body Weight': 'Normal',
        'Calcium Intake': 'Adequate', 'Vitamin D Intake': 'Sufficient', 'Physical Activity': 'Active',
        'Smoking': 'No', 'Alcohol Consumption': 'None', 'Medical Conditions': 'None',
        'Medications': 'None', 'Prior Fractures': 'No'
    }, 0)
]

@pytest.mark.parametrize("case_name, inputs, expected_label", MANUAL_CASES)
def test_manual_prediction(model_assets, case_name, inputs, expected_label):
    """Test manual edge cases for prediction consistency."""
    all_models, label_encoders, scaler = model_assets
    
    test_inputs = inputs.copy()
    gender = test_inputs['Gender']
    test_inputs['_selected_model_key'] = 'male_rf' if gender == 'Male' else 'female_rf'
    
    prediction, risk_score = make_prediction(test_inputs, all_models, label_encoders, scaler)
    
    # If expected_label is None, we just check if it's a valid prediction
    if expected_label is not None:
        error_msg = f"FAILED: {case_name} | Expected {expected_label}, got {prediction} (Risk: {risk_score:.4f})"
        assert prediction == expected_label, error_msg
    else:
        assert prediction in [0, 1]

# ==============================================================================
# PART 2: REAL DATA SUCCESS RATE
# ==============================================================================

def test_real_data_accuracy(model_assets):
    """Calculate accuracy against real training data samples."""
    all_models, label_encoders, scaler = model_assets
    
    csv_path = 'data/osteoporosis_data.csv'
    if not os.path.exists(csv_path):
        pytest.skip("data/osteoporosis_data.csv not found")
        
    df = pd.read_csv(csv_path)
    
    # Take 5 samples (3 negative, 2 positive)
    samples_neg = df[df['Osteoporosis'] == 0].head(3)
    samples_pos = df[df['Osteoporosis'] == 1].head(2)
    samples = pd.concat([samples_neg, samples_pos])
    
    correct_count = 0
    total_count = len(samples)
    wrong_cases = []

    for _, row in samples.iterrows():
        inputs = {
            'Age': int(row['Age']), 'Gender': row['Gender'], 'Hormonal Changes': row['Hormonal Changes'],
            'Family History': row['Family History'], 'Race/Ethnicity': row['Race/Ethnicity'],
            'Body Weight': row['Body Weight'], 'Calcium Intake': row['Calcium Intake'],
            'Vitamin D Intake': row['Vitamin D Intake'], 'Physical Activity': row['Physical Activity'],
            'Smoking': row['Smoking'],
            'Alcohol Consumption': row['Alcohol Consumption'] if pd.notna(row['Alcohol Consumption']) else 'None',
            'Medical Conditions': row['Medical Conditions'] if pd.notna(row['Medical Conditions']) else 'None',
            'Medications': row['Medications'] if pd.notna(row['Medications']) else 'None',
            'Prior Fractures': row['Prior Fractures']
        }
        actual_label = int(row['Osteoporosis'])
        
        inputs['_selected_model_key'] = 'male_rf' if inputs['Gender'] == 'Male' else 'female_rf'
        prediction, risk_score = make_prediction(inputs, all_models, label_encoders, scaler)
        
        if prediction == actual_label:
            correct_count += 1
        else:
            wrong_cases.append({
                'id': row.get('Id', 'Unknown'),
                'actual': actual_label,
                'pred': prediction,
                'risk': risk_score
            })

    accuracy = (correct_count / total_count) * 100
    print(f"\n--- REAL DATA VALIDATION SUMMARY ---")
    print(f"Total Samples Tested: {total_count}")
    print(f"Correct Predictions:  {correct_count}")
    print(f"Final Success Rate:   {accuracy:.2f}%")
    
    if wrong_cases:
        print("\nWRONG CASES DETAILS:")
        for wc in wrong_cases:
            print(f"  - Sample ID {wc['id']}: Expected {wc['actual']}, predicted {wc['pred']} (Risk: {wc['risk']:.4f})")
    
    assert accuracy > 80, f"Accuracy too low: {accuracy:.2f}%"

# ==============================================================================
# PART 3: MODEL ASSETS INTEGRITY
# ==============================================================================

def test_asset_loading(model_assets):
    """Verify all required model files are present and loadable."""
    all_models, label_encoders, scaler = model_assets
    assert all_models is not None
    assert label_encoders is not None
    assert scaler is not None
    assert all(k in all_models for k in ['male_rf', 'female_rf', 'male_ada', 'female_et'])

def test_feature_consistency():
    """Verify feature names match model expectation."""
    features = get_feature_names()
    assert len(features) == 14
    assert 'Age' in features
    assert 'Prior Fractures' in features

if __name__ == "__main__":
    # If run directly as a script, provide a quick summary
    print("This file is designed to be run with 'pytest'.")
    print("Example: pytest test.py -v -s")
