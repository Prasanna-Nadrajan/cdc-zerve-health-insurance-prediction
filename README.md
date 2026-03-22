# CDC x Zerve Hackathon: Health Insurance Claim Prediction

This repository contains the full Machine Learning pipeline and final submission for the Chennai Data Circle (CDC) x Zerve AI Hackathon 2026.

The goal of this project was to predict the probability of a customer filing a significant health insurance claim based on an anonymized dataset of 50 engineered features.

## 🚀 Project Overview

Problem: Binary Classification (Imbalanced)

Target: Probability (0.0 to 1.0) of a health insurance claim (target = 1)

Primary Metric: Precision-Recall AUC (PR-AUC)

Platform: Developed natively on Zerve AI

## 🏗️ Technical Methodology

### 1. Data Preprocessing & Cleaning

Missing Value Handling: Implemented a robust imputation strategy using Median (numerical) and Mode (categorical) values.

Magic Number Detection: Identified and treated anonymized "magic numbers" (e.g., -1, 999) as missing data to prevent model bias.

Feature Engineering: Generated a missing_count feature per row to capture potential signals in data sparsity.

### 2. Dimensionality Reduction (PCA)

With 50 anonymized features, multicollinearity was a significant risk.

Scaling: Applied StandardScaler to normalize feature distributions.

PCA Integration: Employed Principal Component Analysis to retain 95% of cumulative variance, effectively reducing the feature space to 36 principal components. This ensured the model focused on core variance while mitigating overfitting.

### 3. Modeling Strategy

Algorithm: XGBoost (Extreme Gradient Boosting) Classifier.

Class Imbalance: Addressed the 26:1 imbalance ratio using the scale_pos_weight parameter, forcing the model to prioritize minority class (claims) recognition.

Validation: Utilized Stratified 5-Fold Cross-Validation to ensure stable performance and consistent PR-AUC scores across data splits.

Output: Generated raw probabilities to provide granular risk ranking, specifically optimized for the PR-AUC metric.

## 📊 Results

The model achieved a stable PR-AUC across all folds, significantly exceeding the random baseline of ~0.036. The use of probability-based outputs allowed for a sophisticated ranking of customer risk profiles.

## 🛠️ Tools Used

Zerve AI: For canvas-based iterative development and AI-assisted pipeline orchestration.

Python Libraries: pandas, numpy, scikit-learn, XGBoost, matplotlib.

## 📜 Submission Requirements

Prediction File: final_submission_prasanna.csv (contains id and target probabilities).

Architecture: End-to-end automated pipeline from raw CSV to inference.

Developed for the CDC x Zerve AI Hackathon - March 2026
