# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 18:39:09 2024

@author: Kaan
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# DataFrame creation

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("DataFrame shape = ", df.shape)
print("Target Classes =", np.unique(y))


# Correlation Analysis

correlation_matrix = df.corr()

target_correlations = correlation_matrix['target'].sort_values(ascending = False)

high_corr_features = target_correlations[abs(target_correlations) > 0.6].index.drop('target')
if len(high_corr_features) > 0:
    hc_indices = [list(feature_names).index(f) for f in high_corr_features]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
chi_scores, p_values = chi2(X_scaled, y)
chi2_results = pd.DataFrame({"Feature  ": feature_names, "Chi2_Score" : chi_scores, "p_value": p_values})
chi2_results_sorted = chi2_results.sort_values("Chi2_Score", ascending=False)

rf = RandomForestClassifier(n_estimators=100, random_state=1903)
rfe = RFE(rf, n_features_to_select = 5)
rfe.fit(X,y)
rfe_selected = np.array(feature_names)[rfe.support_]
rfe_selected

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 1903, test_size=0.3)


full_rf = RandomForestClassifier(n_estimators = 100, random_state = 1903)
full_rf.fit(X_train,y_train)
print("RF accuracy with all features = ", full_rf.score(X_test, y_test))

# b) Using RFE-selected features
X_train_rfe, X_test_rfe = X_train[:, rfe.support_], X_test[:, rfe.support_]
rfe_rf = RandomForestClassifier(n_estimators=100, random_state=1903)
rfe_rf.fit(X_train_rfe, y_train)
print("RF accuracy with RFE-selected features:", rfe_rf.score(X_test_rfe, y_test))

# c) Using the top 5 Chi2-selected features
top_chi2_indices = [list(feature_names).index(f) for f in top_chi2_features]
X_train_chi, X_test_chi = X_train[:, top_chi2_indices], X_test[:, top_chi2_indices]
chi_rf = RandomForestClassifier(n_estimators=100, random_state=1903)
chi_rf.fit(X_train_chi, y_train)
print("RF accuracy with Chi2-selected features:", chi_rf.score(X_test_chi, y_test))

# d) Using correlation-based features (|corr| > 0.6)
X_train_hc, X_test_hc = X_train[:, hc_indices], X_test[:, hc_indices]
hc_rf = RandomForestClassifier(n_estimators=100, random_state=1903)
hc_rf.fit(X_train_hc, y_train)
print("RF accuracy with correlation-based features:", hc_rf.score(X_test_hc, y_test))
