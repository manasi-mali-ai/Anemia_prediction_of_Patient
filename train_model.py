import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'Arial'

print("==================================================")
print("  ANEMIA AI MODEL RE-TRAINING & EVALUATION PIPELINE")
print("==================================================")

# 1. Load Data
dataset_path = 'AI_Driven_Cell_Counter_Clean_Training_Dataset.csv'
df = pd.read_csv(dataset_path)
print(f"\n[+] Loaded dataset: '{dataset_path}'")
print(f"    Total rows: {len(df)}, Total columns: {len(df.columns)}")

# 2. Features and Target Preparation
feature_cols = [
    'Age_years', 'Gender', 'RBC_count_10^12_L', 'Hemoglobin_g_dL', 
    'HCT_percent', 'MCV_fL', 'MCH_pg', 'MCHC_g_dL', 'RDW_CV_percent'
]

X = df[feature_cols].copy()
# Normalize and encode Gender: Male=0, Female=1
X['Gender'] = X['Gender'].astype(str).str.strip().str.upper().map({'MALE': 0, 'FEMALE': 1, 'M': 0, 'F': 1})

# Rename features to exact match expected by app.py
EXPECTED_FEATURES = [
    "Age", "Gender", "RBC(10^12/L)", "HGB(g/dl)", 
    "HCT(%)", "MCV(fl)", "MCH(pg)", "MCHC(g/dl)", "RDW-CV(%)"
]
X.columns = EXPECTED_FEATURES

# Harmonize class labels clinically (consolidating HS/AIHA spherocytic hemolytic cases)
label_mapping = {
    'Chronic disease / normocytic': 'Chronic Disease / Normocytic Anemia',
    'IDA': 'Iron Deficiency Anemia (IDA)',
    'HS': 'Hereditary Spherocytosis (HS)',
    'HS/AIHA': 'Hereditary Spherocytosis (HS)',
    'No anemia': 'No Anemia (Healthy Reference)',
    'B12/Folate deficiency': 'B12 / Folate Deficiency Anemia',
    'Mixed deficiency': 'Mixed Deficiency Anemia (IDA + Folate/B12)',
    'Thalassemia trait': 'Thalassemia Trait'
}

y = df['Suggested_Label_or_Source_Label'].map(label_mapping)
print("\n[+] Target class distribution:")
for cls, cnt in y.value_counts().items():
    print(f"    - {cls:45s}: {cnt:3d} cases ({cnt/len(y)*100:.1f}%)")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Stratified 5-Fold Cross Validation Across Multiple Models
print("\n[+] Running 5-Fold Stratified Cross-Validation across candidate models...")
candidate_models = {
    "Random Forest (200 trees)": RandomForestClassifier(n_estimators=200, random_state=42),
    "Extra Trees (200 trees)": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    "Logistic Regression (scaled)": Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=42))]),
    "SVM (RBF, scaled)": Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', probability=True, random_state=42))]),
    "K-Nearest Neighbors (k=5)": Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in candidate_models.items():
    acc_scores = cross_val_score(model, X, y_encoded, cv=skf, scoring='accuracy')
    f1_macro = cross_val_score(model, X, y_encoded, cv=skf, scoring='f1_macro')
    cv_results.append({
        'Model': name,
        'CV Accuracy Mean': acc_scores.mean(),
        'CV Accuracy Std': acc_scores.std(),
        'CV Macro F1': f1_macro.mean()
    })
    print(f"    {name:30s} | Accuracy: {acc_scores.mean():.4f} +/- {acc_scores.std():.4f} | Macro F1: {f1_macro.mean():.4f}")

cv_df = pd.DataFrame(cv_results)

# 4. Train/Test Holdout Evaluation
print("\n[+] Performing 80/20 Stratified Holdout Split for In-Depth Validation...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# Imputer & Scaler
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Best Model: Production Pipeline with RandomForestClassifier
best_rf = RandomForestClassifier(
    n_estimators=200,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Build Production Pipeline
production_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', best_rf)
])

production_pipeline.fit(X_train, y_train)

# Predictions on holdout test set
y_test_pred = production_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')
test_weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"\n[+] Holdout Test Evaluation Results:")
print(f"    Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"    Macro F1:    {test_macro_f1:.4f}")
print(f"    Weighted F1: {test_weighted_f1:.4f}")

report = classification_report(y_test, y_test_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# 5. Feature Importances
clf_in_pipe = production_pipeline.named_steps['classifier']
feature_importances = pd.Series(clf_in_pipe.feature_importances_, index=EXPECTED_FEATURES).sort_values(ascending=False)
print("\nFeature Importances:")
for feat, imp in feature_importances.items():
    print(f"    {feat:15s}: {imp:.4f} ({imp*100:.2f}%)")

# 6. Generate Evaluation Plots
os.makedirs("evaluation_plots", exist_ok=True)

# Plot 1: Model Comparison
plt.figure(figsize=(10, 5))
bars = plt.barh(cv_df['Model'], cv_df['CV Accuracy Mean'], xerr=cv_df['CV Accuracy Std'], color='#2b5c8f', capsize=4, alpha=0.85)
plt.xlim(0.80, 1.0)
plt.xlabel('Cross-Validation Accuracy', fontsize=12)
plt.title('Candidate Model Performance (5-Fold Stratified CV)', fontsize=14, fontweight='bold', pad=12)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w*100:.2f}%", va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('evaluation_plots/model_comparison.png', dpi=300)
plt.close()

# Plot 2: Confusion Matrix
plt.figure(figsize=(9, 7))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, cbar=False)
plt.title('Test Set Confusion Matrix (Random Forest)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('evaluation_plots/confusion_matrix.png', dpi=300)
plt.close()

# Plot 3: Feature Importances
plt.figure(figsize=(9, 5))
bars = plt.barh(feature_importances.index[::-1], feature_importances.values[::-1], color='#e05d44', alpha=0.85)
plt.xlabel('Importance Score', fontsize=12)
plt.title('Random Forest Feature Importance Analysis', fontsize=14, fontweight='bold', pad=12)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.003, bar.get_y() + bar.get_height()/2, f"{w*100:.2f}%", va='center', fontsize=10, fontweight='bold')
plt.xlim(0, max(feature_importances.values)*1.18)
plt.tight_layout()
plt.savefig('evaluation_plots/feature_importances.png', dpi=300)
plt.close()

print("\n[+] Evaluation plots saved to 'evaluation_plots/'")

# 7. Final Training on Full Clean Dataset for Production
print("\n[+] Retraining final production pipeline on full clean dataset (277 samples)...")
full_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ))
])
full_pipeline.fit(X, y_encoded)

# Also fit standalone scaler and imputer for legacy notebook / app compatibility
full_imputer = SimpleImputer(strategy='median')
full_X_imp = full_imputer.fit_transform(X)
full_scaler = StandardScaler()
full_scaler.fit(full_X_imp)

# Save artifacts to repo/
joblib.dump(full_pipeline, 'repo/anemia_pipeline.pkl')
joblib.dump(le, 'repo/label_encoder.pkl')
joblib.dump(full_imputer, 'repo/imputer.pkl')
joblib.dump(full_scaler, 'repo/scaler.pkl')

# Save local backup in root
joblib.dump(full_pipeline, 'anemia_pipeline.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(full_imputer, 'imputer.pkl')
joblib.dump(full_scaler, 'scaler.pkl')

print("[+] Successfully saved updated model artifacts:")
print("    - repo/anemia_pipeline.pkl")
print("    - repo/label_encoder.pkl")
print("    - repo/imputer.pkl")
print("    - repo/scaler.pkl")

# Save cv_df and report_df to CSV for reporting
cv_df.to_csv('evaluation_plots/cv_results.csv', index=False)
report_df.to_csv('evaluation_plots/test_classification_report.csv')
feature_importances.to_frame('Importance').to_csv('evaluation_plots/feature_importances.csv')
print("\n[+] Re-training completed successfully!")
