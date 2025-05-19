# Guest Satisfaction Prediction

A machine learning project that predicts guest satisfaction ratings and levels for Airbnb listings using advanced regression and classification techniques. This project includes thorough preprocessing, feature engineering, and model evaluation, culminating in a Streamlit web application for interactive use.

## Overview

Guest satisfaction is a crucial metric for Airbnb hosts and potential guests. Using a dataset of Airbnb listings, this project aims to:

- Predict **review scores rating** (regression)
- Predict **guest satisfaction level** (classification: Very High, High, Average)

## Dataset

- **Source**: Airbnb  
- **Rows**: 8,724  
- **Features**: 68 + 1 target  
- **Target Columns**:  
  - `review_scores_rating` (Regression)  
  - `guest_satisfaction` (Classification - engineered)  

## Preprocessing Steps

- Duplicate & null handling
- Numerical fixes and type casting
- Flooring fractional numeric values (e.g. bathrooms, beds)
- Text processing (cleaning, tokenization, lemmatization, TF-IDF, SVD)
- Sentiment analysis with TextBlob
- Encoding (label, binary, rank mapping)
- Handling outliers using IQR

## Feature Engineering

- **Web scraping**:  
  - `host_rating` using BeautifulSoup  
  - `guest_favorite` using Selenium  
- **Text-derived features**: TF-IDF + SVD (reduced to 35 for regression, 50 for classification)
- **Amenities categorization** (Luxury, Safety, Essentials, etc.)
- **Derived features**: `years_active`, `total_cost`, `reviews_per_day`
- **Location breakdown**: `host_location` → `host_country`, `host_state`, etc.

## Feature Selection

- **Regression**: Correlation (top 35 features)
- **Classification**:  
  - Discrete: Chi-Squared  
  - Continuous: ANOVA  

## Models Used

### Regression

- **Linear Models**: Linear, Lasso, Ridge, ElasticNet
- **SVR** (RBF kernel)
- **Ensemble**:  
  - Random Forest  
  - XGBoost  
  - CatBoost  
  - LightGBM  
  - **Stacking** (Best, **R² = 0.3044**)

### Classification

- Logistic Regression
- SVC (RBF kernel)
- **Ensemble**:  
  - Random Forest  
  - XGBoost  
  - CatBoost  
  - LightGBM  
  - Voting  
  - **Stacking** (Best, **Accuracy = 0.6470**)

> SMOTE was used to handle class imbalance in classification.

## Evaluation Metrics

- **Regression**: R², MAE, MSE, RMSE, MAPE
- **Classification**: Accuracy, Precision, Recall, F1-Score

## Deployment

Built and deployed with **Streamlit** on [Streamlit Cloud](https://guest-satisfaction-project.streamlit.app/).

### App Features

- CSV Upload (Batch predictions)
- Form-based input (Single prediction)
- Metrics visualization (bar charts, gauges)
- Project overview, dataset preview, documentation

🔗 [Launch the App](https://guest-satisfaction-project.streamlit.app/)

---

