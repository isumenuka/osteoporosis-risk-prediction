# Osteoporosis Risk Assessment Application 🦴

A machine learning-powered web application for predicting osteoporosis risk based on personal health data and lifestyle factors. The application offers **gender-specific model selection**, allowing users to choose between two trained models per gender for personalized and comparative risk assessments.

---

## 🚀 Features

- **Personalized Risk Assessment**: Input age, gender, lifestyle habits, and medical history to get an instant risk prediction.
- **4 Gender-Specific Models**: Separate, optimized models for males and females — each with two algorithm choices for flexible prediction.
- **Model Selection Dropdown**: After selecting gender, users can choose which trained model to use for prediction.
- **Actionable Recommendations**: Tailored advice on calcium intake, Vitamin D, exercise, and lifestyle changes based on your specific risk profile.
- **Interactive UI**: Clean, dark-themed interface built with Streamlit for a premium user experience.

---

## 🧠 Model Information

The application uses **4 pre-trained models** located in the `models/` directory, split by gender:

### 👨 Male Models

| Model | File | Algorithm |
|-------|------|-----------|
| 1st Model | `osteoporosis_male_random_forest_model.pkl` | Random Forest Classifier |
| 2nd Model | `osteoporosis_male_adaboost_model_2nd.pkl` | AdaBoost Classifier |

### 👩 Female Models

| Model | File | Algorithm |
|-------|------|-----------|
| 1st Model | `osteoporosis_female_random_forest_model.pkl` | Random Forest Classifier |
| 2nd Model | `osteoporosis_female_extra_trees_model_2nd.pkl` | Extra Trees Classifier |

### Shared Assets

| File | Purpose |
|------|---------|
| `label_encoders.pkl` | Label encoders for categorical features |
| `scaler.pkl` | Feature scaler applied before prediction |

> **Note on Age Sensitivity**: All models are highly sensitive to **Age**. Individuals over **45 years old** may frequently receive a "High Risk" prediction. This reflects the patterns learned from training data and is intentionally conservative for older demographics.

---

## 🛠️ Installation & Usage

1. **Clone the Repository** (if applicable) or download the project files.

2. **Install Dependencies**:
   Ensure you have Python installed. Install the required libraries using pip:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
   ```

3. **Run the Application**:
   Navigate to the project directory in your terminal and run:
   ```bash
   streamlit run Osteoporosis.py
   ```

4. **Access the App**:
   The application will open automatically in your default web browser (usually at `http://localhost:8501`).

5. **Select Your Model**:
   - Choose your **Gender** from the dropdown.
   - A second dropdown will appear to select between the **1st Model** (Random Forest) or the **2nd Model** (AdaBoost for males / Extra Trees for females).
   - Fill in remaining health details and click **Calculate Osteoporosis Risk**.

---

## 📂 Project Structure

```
osteoporosis-risk-prediction/
│
├── Osteoporosis.py          # Main application script (UI, logic, prediction pipeline)
│
├── models/                  # Pre-trained model files
│   ├── osteoporosis_male_random_forest_model.pkl
│   ├── osteoporosis_male_adaboost_model_2nd.pkl
│   ├── osteoporosis_female_random_forest_model.pkl
│   ├── osteoporosis_female_extra_trees_model_2nd.pkl
│   ├── label_encoders.pkl
│   └── scaler.pkl
│
├── data/                    # Dataset used for training
│   └── osteoporosis_data.csv
│
├── notebooks/               # Jupyter notebooks for analysis & training
│   └── MASTER_Complete_Pipeline.ipynb
│
└── docs/                    # Detailed documentation and guides
```

---

## 📋 Input Features

| Feature | Description |
|---------|-------------|
| Age | Current age in years |
| Gender | Biological sex (Male / Female) |
| Hormonal Status | Normal, Postmenopausal, Perimenopausal, Low Testosterone |
| Family History | Parent or sibling with osteoporosis |
| Race/Ethnicity | Caucasian, Asian, African American, Hispanic, Other |
| Body Weight | Normal, Underweight, Overweight |
| Calcium Intake | Adequate, Low, or High daily calcium |
| Vitamin D Intake | Sufficient or Insufficient |
| Physical Activity | Active, Moderate, or Sedentary |
| Smoking | Yes / No |
| Alcohol Consumption | None, Moderate, or Heavy |
| Medical Conditions | Rheumatoid Arthritis, Thyroid Disorders, etc. |
| Medications | Corticosteroids, Anticonvulsants, etc. |
| Prior Fractures | History of fractures after age 50 |

---

**DSGP Group 40** | 2026
