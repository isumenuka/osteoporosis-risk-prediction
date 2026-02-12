# ============================================================================
# SECTION 10.2: SAVE BEST MODEL AS .PKL (UPDATED - FIXED IMPORT ERROR)
# ============================================================================
# Copy this code and replace your Section 10.2 cell in the notebook

# Import joblib for saving sklearn/xgboost models (better than pickle)
import joblib

print("="*80)
print("📦 MODEL ARTIFACTS SAVED")
print("="*80)

# Save the scaler for deployment
scaler_path = "models/scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"\n✅ Scaler saved to: {scaler_path}")

# Save label encoders dictionary
encoders_path = "models/label_encoders.pkl"
joblib.dump(le_dict, encoders_path)
print(f"✅ Label encoders saved to: {encoders_path}")

print("\n" + "="*80)
print("💾 FINAL SAVING: BEST GENDER-SPECIFIC MODELS")
print("="*80)

# Save Male
if 'GLOBAL_BEST_MALE_MODEL' in globals():
    path = 'models/osteoporosis_male_model.pkl'
    joblib.dump(GLOBAL_BEST_MALE_MODEL, path)
    print(f"✅ Saved MALE Model: {path}")
else:
    print("⚠️ GLOBAL_BEST_MALE_MODEL not found. Did you run Part 5?")

# Save Female
if 'GLOBAL_BEST_FEMALE_MODEL' in globals():
    path = 'models/osteoporosis_female_model.pkl'
    joblib.dump(GLOBAL_BEST_FEMALE_MODEL, path)
    print(f"✅ Saved FEMALE Model: {path}")
else:
    print("⚠️ GLOBAL_BEST_FEMALE_MODEL not found. Did you run Part 5?")

print("\n" + "="*80)
print("✅ ALL MODELS AND ARTIFACTS SAVED SUCCESSFULLY!")
print("="*80)
