import json
import os

nb_path = r'c:\Users\Isum Enuka\Downloads\osteoporosis-risk-prediction\notebooks\MASTER_Complete_Pipeline.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error loading notebook: {e}")
    exit(1)

# Define new content
new_train_eval_func = [
    "# ============================================================================\n",
    "# SECTION 5.1: DEFINE MODEL TRAINING FUNCTIONS & HYPERPARAMETERS\n",
    "# ============================================================================\n",
    "\n",
    "from scipy.stats import randint, uniform\n",
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "\n",
    "def get_models_and_params():\n",
    "    # Returns tuple of (models_dict, params_dict)\n",
    "    models = {\n",
    "        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),\n",
    "        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),\n",
    "        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),\n",
    "        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),\n",
    "        'XGBoost': XGBClassifier(random_state=RANDOM_STATE, verbosity=0, eval_metric='logloss'),\n",
    "        'AdaBoost': AdaBoostClassifier(random_state=RANDOM_STATE),\n",
    "        'Bagging': BaggingClassifier(random_state=RANDOM_STATE),\n",
    "        'KNN': KNeighborsClassifier(),\n",
    "        'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),\n",
    "        'Neural Network': 'NN_SPECIAL', # Handled separately\n",
    "        'Stacking': StackingClassifier(\n",
    "            estimators=[\n",
    "                ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),\n",
    "                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE))\n",
    "            ],\n",
    "            final_estimator=LogisticRegression()\n",
    "        ),\n",
    "        'Extra Trees': ExtraTreesClassifier(random_state=RANDOM_STATE)\n",
    "    }\n",
    "\n",
    "    params = {\n",
    "        'Logistic Regression': {'C': uniform(0.1, 10), 'solver': ['liblinear', 'lbfgs']},\n",
    "        'Decision Tree': {'max_depth': randint(3, 20), 'min_samples_split': randint(2, 20)},\n",
    "        'Random Forest': {'n_estimators': randint(50, 300), 'max_depth': randint(3, 20), 'min_samples_split': randint(2, 10)},\n",
    "        'Gradient Boosting': {'n_estimators': randint(50, 300), 'learning_rate': uniform(0.01, 0.3), 'max_depth': randint(3, 10)},\n",
    "        'XGBoost': {'n_estimators': randint(50, 300), 'learning_rate': uniform(0.01, 0.3), 'max_depth': randint(3, 10), 'subsample': uniform(0.5, 0.5)},\n",
    "        'AdaBoost': {'n_estimators': randint(50, 300), 'learning_rate': uniform(0.01, 1.0)},\n",
    "        'Bagging': {'n_estimators': randint(10, 100)},\n",
    "        'KNN': {'n_neighbors': randint(3, 20), 'weights': ['uniform', 'distance']},\n",
    "        'SVM': {'C': uniform(0.1, 10), 'gamma': ['scale', 'auto']},\n",
    "        'Stacking': {}, # Usually not tuned in this simple loop\n",
    "        'Extra Trees': {'n_estimators': randint(50, 300), 'max_depth': randint(3, 20)}\n",
    "    }\n",
    "    return models, params\n",
    "\n",
    "def train_evaluate_gender_models(X_tr, y_tr, X_te, y_te, gender_name):\n",
    "    print(f'\\n' + '='*60)\n",
    "    print(f'⚙️ TUNING & TRAINING MODELS FOR: {gender_name.upper()}')\n",
    "    print('='*60)\n",
    "\n",
    "    models, params = get_models_and_params()\n",
    "    gender_results = {}\n",
    "    gender_trained_models = {}\n",
    "\n",
    "    for name, model in models.items():\n",
    "        # Optimization: Only process XGBoost as requested for 'best perfomance' and saving\n",
    "        if name != 'XGBoost':\n",
    "            continue\n",
    "\n",
    "        print(f'   Processing {name}...')\n",
    "        try:\n",
    "            final_model = model\n",
    "            training_history = None\n",
    "\n",
    "            # 1. Hyperparameter Tuning (RandomizedSearchCV)\n",
    "            if name in params and params[name]:\n",
    "                print(f'      -> Tuning hyperparameters...')\n",
    "                search = RandomizedSearchCV(\n",
    "                    estimator=model,\n",
    "                    param_distributions=params[name],\n",
    "                    n_iter=10,\n",
    "                    cv=3,\n",
    "                    scoring='roc_auc',\n",
    "                    random_state=RANDOM_STATE,\n",
    "                    n_jobs=-1\n",
    "                )\n",
    "                search.fit(X_tr, y_tr)\n",
    "                final_model = search.best_estimator_\n",
    "                print(f'      -> Best Score: {search.best_score_:.4f}')\n",
    "\n",
    "                # REFIT WITH EVAL_SET FOR LOSS CURVES (XGBoost specific)\n",
    "                if name == 'XGBoost':\n",
    "                    print(f'      -> Refitting with eval_set for Loss Graphs...')\n",
    "                    final_model.fit(\n",
    "                        X_tr, y_tr,\n",
    "                        eval_set=[(X_tr, y_tr), (X_te, y_te)],\n",
    "                        verbose=False\n",
    "                    )\n",
    "                    training_history = final_model.evals_result()\n",
    "\n",
    "            else:\n",
    "                final_model.fit(X_tr, y_tr)\n",
    "\n",
    "            # 2. Evaluation\n",
    "            y_pred = final_model.predict(X_te)\n",
    "            y_pred_proba = final_model.predict_proba(X_te)[:, 1]\n",
    "\n",
    "            # 3. Overfitting Check\n",
    "            y_train_pred = final_model.predict(X_tr)\n",
    "            train_acc = accuracy_score(y_tr, y_train_pred)\n",
    "            test_acc = accuracy_score(y_te, y_pred)\n",
    "            overfit_gap = train_acc - test_acc\n",
    "\n",
    "            # Metrics\n",
    "            roc = roc_auc_score(y_te, y_pred_proba)\n",
    "            f1 = f1_score(y_te, y_pred)\n",
    "\n",
    "            gender_results[name] = {\n",
    "                'accuracy': test_acc,\n",
    "                'roc_auc': roc,\n",
    "                'f1_score': f1,\n",
    "                'train_accuracy': train_acc,\n",
    "                'overfit_gap': overfit_gap,\n",
    "                'model_obj': final_model,\n",
    "                'history': training_history\n",
    "            }\n",
    "            gender_trained_models[name] = final_model\n",
    "\n",
    "            print(f'      -> Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Gap: {overfit_gap:.4f}')\n",
    "\n",
    "        except Exception as e:\n",
    "            print(f'   ⚠️ Error training {name}: {str(e)}')\n",
    "\n",
    "    return gender_results, gender_trained_models\n"
]

new_male_execution = [
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
    "    # PLOT LOSS CURVE\n",
    "    if history:\n",
    "        training_loss = history['validation_0']['logloss']\n",
    "        validation_loss = history['validation_1']['logloss']\n",
    "\n",
    "        plt.figure(figsize=(10, 6))\n",
    "        plt.plot(training_loss, label='Training Loss')\n",
    "        plt.plot(validation_loss, label='Validation Loss')\n",
    "        plt.xlabel('Iterations')\n",
    "        plt.ylabel('Log Loss')\n",
    "        plt.title('Male XGBoost Training vs Validation Loss')\n",
    "        plt.legend()\n",
    "        plt.grid(True)\n",
    "        plt.show()\n"
]

new_female_execution = [
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
    "    # PLOT LOSS CURVE\n",
    "    if history:\n",
    "        training_loss = history['validation_0']['logloss']\n",
    "        validation_loss = history['validation_1']['logloss']\n",
    "\n",
    "        plt.figure(figsize=(10, 6))\n",
    "        plt.plot(training_loss, label='Training Loss')\n",
    "        plt.plot(validation_loss, label='Validation Loss')\n",
    "        plt.xlabel('Iterations')\n",
    "        plt.ylabel('Log Loss')\n",
    "        plt.title('Female XGBoost Training vs Validation Loss')\n",
    "        plt.legend()\n",
    "        plt.grid(True)\n",
    "        plt.show()\n"
]

new_section_6_saving = [
    "# ============================================================================\n",
    "# SECTION 6.3: TUNED MODEL LEADERBOARD & SELECTION\n",
    "# ============================================================================\n",
    "\n",
    "tuned_df = pd.DataFrame(tuned_results).T.drop('model_obj', axis=1)\n",
    "tuned_df = tuned_df.sort_values('roc_auc', ascending=False)\n",
    "\n",
    "print('\\n🏆 FINAL TUNED MODEL LEADERBOARD:')\n",
    "print(tuned_df)\n",
    "\n",
    "# Select Overall Best\n",
    "best_tuned_name = tuned_df.index[0]\n",
    "best_tuned_model = tuned_results[best_tuned_name]['model_obj']\n",
    "\n",
    "print(f'\\n✨ OVERALL BEST TUNED MODEL: {best_tuned_name}')\n",
    "print(f'   ROC-AUC: {tuned_df.iloc[0][\"roc_auc\"]:.4f}')\n",
    "\n",
    "# Check if Neural Network is best and needs saving differently if complex\n",
    "# But for pickle, Keras models might need 'model.save'. Baseline code used pickle.\n",
    "# For safety with Keras in pickle list:\n",
    "if best_tuned_name == 'Neural Network':\n",
    "    # best_tuned_model.save('models/best_tuned_neural_network.h5')\n",
    "    print('   (Skipping save of general NN model as per user request)')\n",
    "else:\n",
    "    # with open('models/best_tuned_model.pkl', 'wb') as f:\n",
    "    #     pickle.dump(best_tuned_model, f)\n",
    "    print('   (Skipping save of general best model as per user request)')\n"
]

new_section_10_saving = [
    "# ============================================================================\n",
    "# SECTION 10.2: SAVE BEST MODEL AS .PKL\n",
    "# ============================================================================\n",
    "\n",
    "print(\"=\"*80)\n",
    "print(\"📦 MODEL ARTIFACTS SAVED\")\n",
    "print(\"=\"*80)\n",
    "\n",
    "# Save the scaler for deployment\n",
    "scaler_path = \"models/scaler.pkl\"\n",
    "with open(scaler_path, 'wb') as f:\n",
    "    pickle.dump(scaler, f)\n",
    "\n",
    "print(f\"\\n✅ Scaler saved to: {scaler_path}\")\n",
    "\n",
    "# Save label encoders dictionary\n",
    "encoders_path = \"models/label_encoders.pkl\"\n",
    "with open(encoders_path, 'wb') as f:\n",
    "    pickle.dump(le_dict, f)\n",
    "\n",
    "print(f\"✅ Label encoders saved to: {encoders_path}\")\n",
    "\n",
    "print(\"\\n\" + \"=\"*80)\n",
    "print(\"NOTE: As per request, only Gender-Specific (Male/Female) models were saved in Part 5.\")\n",
    "print(\"General models ('best_tuned.pkl', etc.) were skipped to keep artifacts clean.\")\n",
    "print(\"=\"*80)\n"
]

updated_count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        if "SECTION 5.1: DEFINE MODEL TRAINING FUNCTIONS" in source_str:
            cell['source'] = new_train_eval_func
            updated_count += 1
            print("Updated Section 5.1 function definition")
            
        elif "SECTION 5.3: TUNE, TRAIN & EVALUATE MALE MODELS" in source_str:
            cell['source'] = new_male_execution
            updated_count += 1
            print("Updated Section 5.3 Male loop")
            
        elif "SECTION 5.4: TUNE, TRAIN & EVALUATE FEMALE MODELS" in source_str:
            cell['source'] = new_female_execution
            updated_count += 1
            print("Updated Section 5.4 Female loop")
            
        elif "SECTION 6.3: TUNED MODEL LEADERBOARD" in source_str:
            cell['source'] = new_section_6_saving
            updated_count += 1
            print("Updated Section 6.3 Saving logic")
            
        elif "SECTION 10.2: SAVE BEST MODEL AS .PKL" in source_str:
            cell['source'] = new_section_10_saving
            updated_count += 1
            print("Updated Section 10.2 Saving logic")

if updated_count == 0:
    print("WARNING: No cells were updated. Check matching strings.")
else:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Notebook updated successfully. {updated_count} cells modified.")
