# 24S1_imdb_movie_rating_prediction

This repository contains a machine learning project for predicting IMDb movie ratings based on multiple features, including social media influence, genre, and content-related attributes. The work involves preprocessing, feature selection, and training multiple supervised learning models.

## Repository Structure
```text
.
├── COMP30027_code.ipynb       # Main Jupyter Notebook with full preprocessing, training, and evaluation
├── output_KNN.csv             # Test set predictions from K-Nearest Neighbors model
├── output_LR.csv              # Test set predictions from Logistic Regression model
├── output_MLP.csv             # Test set predictions from Multilayer Perceptron model
├── output_rf.csv              # Test set predictions from Random Forest model
├── output_SVM.csv             # Test set predictions from Support Vector Machine model
└── COMP30027_Project2.pdf     # Full project report with methodology, results, and discussion
```

## Objective
The goal of this project is to analyze how various aspects of a film—such as the social media influence of its creators, genre, and content—affect its IMDb rating (0–4 scale) and to predict ratings using supervised learning models.

## Dataset
- **Training set:** 3004 instances with features and IMDb score categories.
- **Test set:** 759 instances with features only.
- Class distribution is imbalanced (score 2 dominates with ~61% of samples).

## Data Preprocessing
- **Numerical Data:** Log transformation, outlier removal (±2×IQR), standardization, and removal of highly correlated features.
- **Nominal Data:**  
  - Country → Binary label (USA / Non-USA)  
  - Language → Binary label (English / Non-English)  
  - Content Rating → One-hot encoding  
  - Genres → Multi-hot encoding  
- **Text Data:** All text-based features removed to reduce complexity and avoid overfitting.

## Model Selection
We implemented and evaluated the following models:
1. **Baseline:** Zero-Rule classifier  
2. **K-Nearest Neighbors (KNN)** — Weighted and unweighted versions  
3. **Support Vector Machine (SVM)** — Soft margin, hard margin, and RBF kernel  
4. **Logistic Regression** — Tuned regularization strength  
5. **Multilayer Perceptron (MLP)** — Fully connected neural network  
6. **Random Forest** — Ensemble of decision trees with bootstrap sampling  

## Evaluation
- **Validation set:** 5-fold cross-validation on training data  
- **Metrics:** Accuracy, precision, recall, F1 score  
- **Key Results:**  
  - **MLP** achieved highest validation accuracy (0.7625) but lower test performance (0.6463) due to overfitting  
  - **Logistic Regression** achieved the highest Kaggle test accuracy (0.70744)  
  - **Random Forest** performed robustly on both validation (0.7407) and test (0.7048) sets  
  - All models struggled to predict class 0 due to severe class imbalance  

## How to Run
1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```
2. Open `COMP30027_code.ipynb` in Jupyter Notebook.
3. Run cells in order to:
   - Load and preprocess the data
   - Train and validate models
   - Generate predictions for the test set
4. Submission CSV files are generated for Kaggle evaluation.

## Key Insights
- Data preprocessing and feature selection significantly improved performance.
- Class imbalance caused models to perform poorly on minority classes (especially score 0).
- Logistic Regression’s simplicity and interpretability made it the most reliable for this dataset.
- MLP showed overfitting tendencies; careful regularization or more balanced data is needed.

## Limitations
- Strong label imbalance; more diverse samples for minority classes are needed.
- Feature engineering could explore non-linear relationships more deeply.
- Some outliers may still influence model performance.

