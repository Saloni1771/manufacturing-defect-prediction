"""
Manufacturing Defect Prediction — SECOM sensor data
Dataset: UCI SECOM (https://archive.ics.uci.edu/dataset/179/secom)

1,567 units, 590 sensor readings each, pass(-1)/fail(1) label.
~6.6% fail rate, so this is genuinely imbalanced, not a toy split.
Trying to flag likely failures before they ship.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              roc_curve, precision_recall_curve, average_precision_score,
                              f1_score, recall_score, precision_score)
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

# ============================================================
# 1. LOAD DATA
# ============================================================
data = pd.read_csv('data/secom.data', sep=' ', header=None)
labels = pd.read_csv('data/secom_labels.data', sep=' ', header=None, names=['label', 'timestamp'])

data.columns = [f'sensor_{i}' for i in range(data.shape[1])]
labels['timestamp'] = labels['timestamp'].str.strip('"')
labels['timestamp'] = pd.to_datetime(labels['timestamp'], format='%d/%m/%Y %H:%M:%S')
labels['fail'] = (labels['label'] == 1).astype(int)  # 1 = fail, 0 = pass

print(f"Data shape: {data.shape}")
print(f"Fail rate: {labels['fail'].mean()*100:.2f}% ({labels['fail'].sum()} fails / {len(labels)} total)")

# ============================================================
# 2. EDA
# ============================================================

# 2a. Class imbalance
plt.figure(figsize=(6, 5))
counts = labels['fail'].value_counts().sort_index()
colors = ['#2C5F7C', '#C0392B']
bars = plt.bar(['Pass', 'Fail'], counts.values, color=colors)
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 15, str(val), ha='center', fontweight='bold')
plt.title(f'Class Distribution — {labels["fail"].mean()*100:.1f}% Fail Rate')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/01_class_imbalance.png', bbox_inches='tight')
plt.close()

# 2b. Missing data
missing_pct = (data.isnull().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
print(f"\nColumns with missing data: {len(missing_pct)} / {data.shape[1]}")
print(f"Columns >50% missing: {(missing_pct > 50).sum()}")

plt.figure(figsize=(9, 5))
plt.hist(missing_pct, bins=40, color='#C9A961', edgecolor='white')
plt.xlabel('% Missing')
plt.ylabel('Number of sensor columns')
plt.title('Distribution of Missingness Across 590 Sensor Features')
plt.tight_layout()
plt.savefig('outputs/02_missing_distribution.png', bbox_inches='tight')
plt.close()

# 2c. Zero/near-zero variance features (dead sensors)
variances = data.var(numeric_only=True)
near_zero_var = (variances < 0.01).sum()
print(f"Near-zero variance sensors (likely dead/constant): {near_zero_var}")

# 2d. Failures over time
labels_sorted = labels.sort_values('timestamp')
labels_sorted['date'] = labels_sorted['timestamp'].dt.date
daily = labels_sorted.groupby('date')['fail'].agg(['sum', 'count'])
daily['fail_rate'] = daily['sum'] / daily['count']

plt.figure(figsize=(11, 4.5))
plt.plot(daily.index, daily['fail_rate'] * 100, color='#C0392B', linewidth=1.2)
plt.fill_between(daily.index, daily['fail_rate'] * 100, alpha=0.15, color='#C0392B')
plt.ylabel('Daily fail rate (%)')
plt.title('Failure Rate Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/03_failure_rate_over_time.png', bbox_inches='tight')
plt.close()

# ============================================================
# 3. FEATURE ENGINEERING / CLEANING
# ============================================================

# Drop columns >50% missing — too sparse to trust
drop_cols = missing_pct[missing_pct > 50].index.tolist()
data_clean = data.drop(columns=drop_cols)
print(f"\nDropped {len(drop_cols)} columns >50% missing")

# Drop near-zero variance columns (dead sensors carry no signal)
variances_clean = data_clean.var(numeric_only=True)
nzv_cols = variances_clean[variances_clean < 0.01].index.tolist()
data_clean = data_clean.drop(columns=nzv_cols)
print(f"Dropped {len(nzv_cols)} near-zero-variance columns")
print(f"Remaining features: {data_clean.shape[1]}")

# Impute remaining missing values with median (robust to outliers, common for sensor data)
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data_clean), columns=data_clean.columns)

# Drop highly correlated features (redundant sensors) — keep one from each pair >0.95
corr_matrix = data_imputed.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
data_final = data_imputed.drop(columns=high_corr_cols)
print(f"Dropped {len(high_corr_cols)} highly correlated (redundant) sensor columns")
print(f"Final feature count: {data_final.shape[1]}")

y = labels['fail'].values
X = data_final

# ============================================================
# 4. MODELING (with imbalance handling via class weights)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set: {X_train.shape[0]} rows ({y_train.sum()} fails)")
print(f"Test set: {X_test.shape[0]} rows ({y_test.sum()} fails)")

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
}

results = []
roc_data = {}
pr_data = {}
fitted_models = {}

for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    fitted_models[name] = model
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, roc_auc)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_data[name] = (prec_curve, rec_curve, avg_prec)

    results.append({
        'Model': name,
        'ROC-AUC': round(roc_auc, 4),
        'Avg Precision (PR-AUC)': round(avg_prec, 4),
        'Recall (catch defects)': round(recall, 4),
        'Precision': round(precision, 4),
        'F1': round(f1, 4)
    })

results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
print("\n" + "="*75)
print("MODEL COMPARISON (imbalanced classification — ROC-AUC is the key metric)")
print("="*75)
print(results_df.to_string(index=False))
results_df.to_csv('outputs/model_comparison.csv', index=False)

# ROC curves
plt.figure(figsize=(7, 6))
colors_map = {'Logistic Regression': '#7C3AED', 'Random Forest': '#16A34A', 'Gradient Boosting': '#2C5F7C'}
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=colors_map[name], linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.6, label='Random baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — Defect Detection Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('outputs/04_roc_curves.png', bbox_inches='tight')
plt.close()

# Precision-Recall curves (more informative than ROC for rare-event/imbalanced problems)
plt.figure(figsize=(7, 6))
for name, (prec, rec, ap) in pr_data.items():
    plt.plot(rec, prec, label=f'{name} (AP = {ap:.3f})', color=colors_map[name], linewidth=2)
baseline = y_test.mean()
plt.axhline(baseline, linestyle='--', color='grey', alpha=0.6, label=f'Baseline ({baseline:.3f})')
plt.xlabel('Recall (share of true defects caught)')
plt.ylabel('Precision (share of flags that are real defects)')
plt.title('Precision-Recall Curves — More Informative for Rare Defects')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('outputs/05_precision_recall_curves.png', bbox_inches='tight')
plt.close()

# Confusion matrix for best model (by ROC-AUC)
best_name = results_df.iloc[0]['Model']
best_model = fitted_models[best_name]
if best_name == 'Logistic Regression':
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Pass', 'Predicted Fail'],
            yticklabels=['Actual Pass', 'Actual Fail'])
plt.title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.savefig('outputs/06_confusion_matrix.png', bbox_inches='tight')
plt.close()

# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
elif best_name == 'Logistic Regression':
    importances = pd.Series(np.abs(best_model.coef_[0]), index=X.columns).sort_values(ascending=False).head(15)

plt.figure(figsize=(9, 7))
sns.barplot(x=importances.values, y=importances.index, color='#2C5F7C')
plt.title(f'Top 15 Most Predictive Sensors — {best_name}')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('outputs/07_feature_importance.png', bbox_inches='tight')
plt.close()

print(f"\nBest model: {best_name}")
print(f"Top 10 predictive sensors:\n{importances.head(10)}")

# ============================================================
# 6. BUSINESS FRAMING — cost-based threshold analysis
# ============================================================
# Simulate: cost of missing a real defect (false negative) vs. cost of
# unnecessary inspection (false positive). Typical semiconductor context:
# missing a defect that ships is far more costly than an extra inspection.
if best_name == 'Logistic Regression':
    y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    y_prob_best = best_model.predict_proba(X_test)[:, 1]

cost_fn = 50   # relative cost of a missed defect (ships to customer)
cost_fp = 1    # relative cost of an unnecessary inspection

thresholds = np.arange(0.05, 0.95, 0.02)
total_costs = []
for t in thresholds:
    preds = (y_prob_best >= t).astype(int)
    fn = ((preds == 0) & (y_test == 1)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
    total_costs.append(fn * cost_fn + fp * cost_fp)

optimal_idx = np.argmin(total_costs)
optimal_threshold = thresholds[optimal_idx]

plt.figure(figsize=(9, 5))
plt.plot(thresholds, total_costs, color='#C0392B', linewidth=2)
plt.axvline(optimal_threshold, linestyle='--', color='#16A34A',
            label=f'Optimal threshold ≈ {optimal_threshold:.2f}')
plt.axvline(0.5, linestyle=':', color='grey', alpha=0.7, label='Default threshold (0.5)')
plt.xlabel('Classification threshold')
plt.ylabel(f'Total relative cost (missed defect = {cost_fn}x an unnecessary check)')
plt.title('Cost-Optimal Decision Threshold')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/08_cost_threshold_analysis.png', bbox_inches='tight')
plt.close()

print(f"\nAt default 0.5 threshold vs cost-optimal {optimal_threshold:.2f} threshold:")
default_preds = (y_prob_best >= 0.5).astype(int)
optimal_preds = (y_prob_best >= optimal_threshold).astype(int)
print(f"Default — Recall: {recall_score(y_test, default_preds):.3f}, Precision: {precision_score(y_test, default_preds, zero_division=0):.3f}")
print(f"Optimal — Recall: {recall_score(y_test, optimal_preds):.3f}, Precision: {precision_score(y_test, optimal_preds, zero_division=0):.3f}")

print("\nDone. All outputs saved to outputs/")
