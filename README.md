# DA5401 Assignment 7: Multi-Class Model Selection using ROC and Precision-Recall Curves

- **Name**: Rishabh Gupta
- **Roll Number**: DA25M024
---

## Project Overview

This assignment focuses on **multi-class model selection** using **Receiver Operating Characteristic (ROC) curves** and **Precision-Recall Curves (PRC)** for land cover classification from satellite imagery. The primary goal is to compare diverse classifiers and identify the best-performing model for classifying six different land cover types using spectral data from the Landsat Satellite dataset.

---

## Dataset

- **Dataset Name**: UCI Landsat Satellite Dataset
- **Source**: UCI Machine Learning Repository - Statlog Landsat Satellite
- **Description**: Contains multi-spectral values of pixels in 3x3 neighborhoods in satellite images with 6 land cover classes
- **Key Statistics**:
  - 36 spectral features per instance
  - 6 land cover classes (after removing class 7 - "all types present")
  - 6,339 total instances after preprocessing
- **Preprocessing Steps**:
  - Removed class 7 ("all types present") as per instructions
  - Standardized features using StandardScaler
  - Train-test split (70-30) with stratification

---

## Essential Files for Evaluation

1. **Assignment 7.ipynb**  
   – Main Jupyter Notebook containing:
     - Data loading and preprocessing
     - Six classifier implementations and training
     - Multi-class ROC and Precision-Recall analysis
     - Comprehensive model comparison and evaluation
     - Brownie points tasks (RandomForest, XGBoost, and AUC < 0.5 models)

2. **README.md**  
   – This documentation file.

---

## Implementation Details

### Key Libraries Used
- **pandas**, **numpy** → Data manipulation and numerical computations
- **matplotlib**, **seaborn** → Performance visualization and curve plotting
- **scikit-learn** → Machine learning models, metrics, and preprocessing
- **xgboost** → Gradient boosting implementation

### Models Compared
1. **K-Nearest Neighbors (KNN)** - Instance-based learning
2. **Decision Tree Classifier** - Non-parametric tree-based model
3. **Dummy Classifier (Prior)** - Baseline using class probabilities
4. **Logistic Regression** - Linear model benchmark
5. **Naive Bayes (Gaussian)** - Probabilistic classifier with independence assumptions
6. **Support Vector Classifier (SVC)** - Kernel-based method with probability estimates

---

## Approach

### Part A: Data Preparation and Baseline
- Loaded and standardized the Landsat Satellite dataset
- Implemented train-test split with stratification
- Trained all six specified classifiers
- Established baseline performance using Accuracy and Weighted F1-Score

### Part B: ROC Analysis for Model Selection
- Implemented One-vs-Rest (OvR) strategy for multi-class ROC curves
- Calculated macro-averaged AUC across all classes
- Generated comprehensive ROC visualizations
- Identified best and worst performing models based on AUC

### Part C: Precision-Recall Curve Analysis
- Explained importance of PRC for imbalanced classification
- Implemented multi-class PRC using OvR approach
- Calculated Average Precision (AP) metrics
- Analyzed precision-recall trade-offs across models

### Part D: Final Recommendation
- Synthesized rankings from F1-Score, ROC-AUC, and PRC-AP
- Provided comprehensive model comparison
- Recommended optimal model with performance justification

### Brownie Points Tasks
- **Task 1**: Implemented and evaluated RandomForest and XGBoost classifiers
- **Task 2**: Created models with AUC < 0.5 using random probability method

---

## Key Findings

### Baseline Performance Summary:
| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| KNN | 0.9114 | 0.9094 |
| SVC | 0.8928 | 0.8913 |
| Logistic Regression | 0.8493 | 0.8421 |
| Decision Tree | 0.8469 | 0.8481 |
| Naive Bayes | 0.7832 | 0.7901 |
| Dummy (Prior) | 0.2385 | 0.0919 |

### ROC Analysis Results:
| Model | Macro AUC | Min Class AUC | Max Class AUC |
|-------|-----------|---------------|---------------|
| KNN | 0.9802 | 0.9336 | 0.9959 |
| SVC | 0.9797 | 0.9232 | 0.9996 |
| Logistic Regression | 0.9720 | 0.9044 | 0.9981 |
| Naive Bayes | 0.9473 | 0.8837 | 0.9923 |
| Decision Tree | 0.8952 | 0.7303 | 0.9730 |
| Dummy (Prior) | 0.5000 | 0.5000 | 0.5000 |

### Main Insights:
- **KNN emerged as the top performer** across all evaluation metrics
- **All competent models achieved AUC > 0.89**, indicating strong discriminative power
- **ROC and PRC rankings showed perfect alignment** for top models
- **Ensemble methods (RandomForest, XGBoost) demonstrated competitive performance**
- **Successfully created models with AUC < 0.5** using random probability method

### Final Recommendation:
**K-Nearest Neighbors (KNN)** is recommended as the optimal model due to:
- Consistent #1 ranking across Accuracy, F1-Score, ROC-AUC, and Average Precision
- Excellent discriminative ability (AUC: 0.9802)
- Strong precision-recall balance (AP: 0.945)
- Robust performance across all six land cover types

---

## Brownie Points Achievement

✅ **Task 1**: RandomForest and XGBoost Implementation  
- Both ensemble methods showed competitive performance with original top models
- RandomForest achieved AUC ~0.985, competing closely with KNN

✅ **Task 2**: Models with AUC < 0.5  
- Successfully implemented random probability model achieving AUC = 0.493
- Demonstrated theoretical baseline for worst-case performance

---

## Technical Contributions

- Comprehensive implementation of multi-class ROC and PRC analysis
- Detailed comparison of six diverse classifier types
- Proper handling of non-standard class labels [1,2,3,4,5,7] for XGBoost
- Clear visualization and interpretation of performance metrics
- Robust experimental methodology with reproducible results

---

## How to Run

1. Ensure all required libraries are installed (scikit-learn, xgboost, matplotlib, seaborn, pandas, numpy)
2. Run the Jupyter notebook `Assignment 7.ipynb` sequentially
3. The notebook will automatically download the Landsat Satellite dataset
4. All results, visualizations, and analysis will be generated automatically

