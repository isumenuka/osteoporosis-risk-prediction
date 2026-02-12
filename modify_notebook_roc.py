import json
import os

nb_path = r'c:\Users\Isum Enuka\Downloads\osteoporosis-risk-prediction\notebooks\MASTER_Complete_Pipeline.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error loading notebook: {e}")
    exit(1)

# Define new content with ROC Curves
new_male_execution_with_roc = [
    "# ============================================================================\n",
    "# SECTION 5.3: TUNE, TRAIN & EVALUATE MALE MODELS\n",
    "# ============================================================================\n",
    "\n",
    "male_results, male_models = train_evaluate_gender_models(X_train_m, y_train_m, X_test_m, y_test_m, 'Male')\n",
    "\n",
    "# Identify Best Male Model (Should be XGBoost as we filtered for it)\n",
    "best_male_name = 'XGBoost'\n",
    "if best_male_name in male_results:\n",
    "    best_male_model = male_results[best_male_name]['model_obj']\n",
    "    history = male_results[best_male_name]['history']\n",
    "\n",
    "    print(f'\\n✨ Best Male Model: {best_male_name}')\n",
    "    print(f'   ROC-AUC: {male_results[best_male_name][\"roc_auc\"]:.4f}')\n",
    "    print(f'   Overfitting Gap: {male_results[best_male_name][\"overfit_gap\"]:.4f}')\n",
    "\n",
    "    # 1. PLOT LOSS CURVE\n",
    "    if history:\n",
    "        training_loss = history['validation_0']['logloss']\n",
    "        validation_loss = history['validation_1']['logloss']\n",
    "\n",
    "        plt.figure(figsize=(10, 5))\n",
    "        plt.plot(training_loss, label='Training Loss')\n",
    "        plt.plot(validation_loss, label='Validation Loss')\n",
    "        plt.xlabel('Iterations')\n",
    "        plt.ylabel('Log Loss')\n",
    "        plt.title('Male XGBoost: Training vs Validation Loss')\n",
    "        plt.legend()\n",
    "        plt.grid(True)\n",
    "        plt.show()\n",
    "\n",
    "    # 2. PLOT ROC CURVE (ROG)\n",
    "    from sklearn.metrics import roc_curve, auc\n",
    "    y_pred_proba = best_male_model.predict_proba(X_test_m)[:, 1]\n",
    "    fpr, tpr, _ = roc_curve(y_test_m, y_pred_proba)\n",
    "    roc_auc_val = auc(fpr, tpr)\n",
    "\n",
    "    plt.figure(figsize=(10, 5))\n",
    "    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.4f})')\n",
    "    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
    "    plt.xlim([0.0, 1.0])\n",
    "    plt.ylim([0.0, 1.05])\n",
    "    plt.xlabel('False Positive Rate')\n",
    "    plt.ylabel('True Positive Rate')\n",
    "    plt.title('Male Model: ROC Curve')\n",
    "    plt.legend(loc=\"lower right\")\n",
    "    plt.grid(True)\n",
    "    plt.show()\n"
]

new_female_execution_with_roc = [
    "# ============================================================================\n",
    "# SECTION 5.4: TUNE, TRAIN & EVALUATE FEMALE MODELS\n",
    "# ============================================================================\n",
    "\n",
    "female_results, female_models = train_evaluate_gender_models(X_train_f, y_train_f, X_test_f, y_test_f, 'Female')\n",
    "\n",
    "# Identify Best Female Model\n",
    "best_female_name = 'XGBoost'\n",
    "if best_female_name in female_results:\n",
    "    best_female_model = female_results[best_female_name]['model_obj']\n",
    "    history = female_results[best_female_name]['history']\n",
    "\n",
    "    print(f'\\n✨ Best Female Model: {best_female_name}')\n",
    "    print(f'   ROC-AUC: {female_results[best_female_name][\"roc_auc\"]:.4f}')\n",
    "    print(f'   Overfitting Gap: {female_results[best_female_name][\"overfit_gap\"]:.4f}')\n",
    "\n",
    "    # 1. PLOT LOSS CURVE\n",
    "    if history:\n",
    "        training_loss = history['validation_0']['logloss']\n",
    "        validation_loss = history['validation_1']['logloss']\n",
    "\n",
    "        plt.figure(figsize=(10, 5))\n",
    "        plt.plot(training_loss, label='Training Loss')\n",
    "        plt.plot(validation_loss, label='Validation Loss')\n",
    "        plt.xlabel('Iterations')\n",
    "        plt.ylabel('Log Loss')\n",
    "        plt.title('Female XGBoost: Training vs Validation Loss')\n",
    "        plt.legend()\n",
    "        plt.grid(True)\n",
    "        plt.show()\n",
    "\n",
    "    # 2. PLOT ROC CURVE (ROG)\n",
    "    from sklearn.metrics import roc_curve, auc\n",
    "    y_pred_proba = best_female_model.predict_proba(X_test_f)[:, 1]\n",
    "    fpr, tpr, _ = roc_curve(y_test_f, y_pred_proba)\n",
    "    roc_auc_val = auc(fpr, tpr)\n",
    "\n",
    "    plt.figure(figsize=(10, 5))\n",
    "    plt.plot(fpr, tpr, color='purple', lw=2, label=f'ROC curve (area = {roc_auc_val:.4f})')\n",
    "    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
    "    plt.xlim([0.0, 1.0])\n",
    "    plt.ylim([0.0, 1.05])\n",
    "    plt.xlabel('False Positive Rate')\n",
    "    plt.ylabel('True Positive Rate')\n",
    "    plt.title('Female Model: ROC Curve')\n",
    "    plt.legend(loc=\"lower right\")\n",
    "    plt.grid(True)\n",
    "    plt.show()\n"
]

updated_count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        if "SECTION 5.3: TUNE, TRAIN & EVALUATE MALE MODELS" in source_str:
            cell['source'] = new_male_execution_with_roc
            updated_count += 1
            print("Updated Section 5.3 Male loop with ROC")
            
        elif "SECTION 5.4: TUNE, TRAIN & EVALUATE FEMALE MODELS" in source_str:
            cell['source'] = new_female_execution_with_roc
            updated_count += 1
            print("Updated Section 5.4 Female loop with ROC")

if updated_count == 0:
    print("WARNING: No cells were updated. Check matching strings.")
else:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Notebook updated successfully. {updated_count} cells modified.")
