"""
@author: Assoc. Prof. Elif Kartal
@title: Feature Selection Methods
@dataset: Wine Data Set https://archive.ics.uci.edu/dataset/109/wine
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.feature_selection import chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml
cifar10 = fetch_openml("CIFAR_10", version=1)

# 1. Load the Wine dataset
data = load_wine()
X = data.data
y = data.target
feature_names = data.feature_names

# Create a DataFrame for easier analysis
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
# Basic dataset info
print("Dataset shape:", df.shape)
print("Target classes:", np.unique(y))

# 2. Correlation Analysis

# Compute correlation matrix and look at correlation with the target variable
correlation_matrix = df.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Sort features by their correlation with the target
target_correlations = correlation_matrix['target'].sort_values(ascending=False)
print("\nCorrelation with target (sorted):\n", target_correlations)

# For example, select features with absolute correlation > 0.6 to the target
high_corr_features = target_correlations[abs(target_correlations) >
0.6].index.drop('target')
print("\nFeatures with |corr| > 0.5 to target:", list(high_corr_features))
if len(high_corr_features) > 0: hc_indices = [list(feature_names).index(f) for f in high_corr_features];

# 3. Chi-Squared Test

# Chi-Square test requires non-negative features, so we scale the data to [0,1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
chi_scores, p_values = chi2(X_scaled, y)
chi2_results = pd.DataFrame({"Feature": feature_names, "Chi2_Score":
chi_scores, "p_value": p_values})
chi2_results_sorted = chi2_results.sort_values("Chi2_Score", ascending=False)
chi2_results_sorted

# Select the top 5 features with the highest Chi2 scores
top_chi2_features = chi2_results_sorted.head(5)['Feature'].values
top_chi2_features

# 4. RFE (Recursive Feature Elimination) using Random Forest as the base model
rf = RandomForestClassifier(n_estimators=100, random_state=1903)
rfe = RFE(rf, n_features_to_select=5)
rfe.fit(X, y)
rfe_selected = np.array(feature_names)[rfe.support_]
rfe_selected

# 5. Apply the results and compare model performance with Random Forest

#Stratified Sampling(Tabakalı Örnekleme) bil sınavda çıkabilir.

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
random_state=1903, test_size=0.3)

# a) Using all features
full_rf = RandomForestClassifier(n_estimators=100, random_state=1903)
full_rf.fit(X_train, y_train)
print("\nRF accuracy with all features:", full_rf.score(X_test, y_test))

# b) Using RFE-selected features
X_train_rfe, X_test_rfe = X_train[:, rfe.support_], X_test[:, rfe.support_]
rfe_rf = RandomForestClassifier(n_estimators=100, random_state=1903)
rfe_rf.fit(X_train_rfe, y_train)
print("RF accuracy with RFE-selected features:", rfe_rf.score(X_test_rfe,
y_test))

# c) Using the top 5 Chi2-selected features
top_chi2_indices = [list(feature_names).index(f) for f in top_chi2_features]
X_train_chi, X_test_chi = X_train[:, top_chi2_indices], X_test[:,
top_chi2_indices]
chi_rf = RandomForestClassifier(n_estimators=100, random_state=1903)
chi_rf.fit(X_train_chi, y_train)
print("RF accuracy with Chi2-selected features:", chi_rf.score(X_test_chi,
y_test))

# d) Using correlation-based features (|corr| > 0.5)
X_train_hc, X_test_hc = X_train[:, hc_indices], X_test[:, hc_indices]
hc_rf = RandomForestClassifier(n_estimators=100, random_state=1903)
hc_rf.fit(X_train_hc, y_train)
print("RF accuracy with correlation-based features:", hc_rf.score(X_test_hc,
y_test))
