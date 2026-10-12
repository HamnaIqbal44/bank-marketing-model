# Credit Risk Decision Studio

An interactive statistical modelling and machine-learning
project built with Python and Streamlit.

## Research Question

How do interpretable statistical classification models compare
with machine-learning models when predicting historical
credit outcomes?

## Dataset

South German Credit — UCI Machine Learning Repository.

The dataset contains 1,000 historical credit records with
20 predictor variables.

## Models

- Logistic Regression
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis
- Naive Bayes
- K-Nearest Neighbors
- Decision Tree
- Random Forest

## Evaluation

Models are compared using:

- Accuracy
- Precision
- Recall
- F1
- ROC AUC
- 5-fold cross-validation ROC AUC

## Responsible Modelling

This is an educational project only.

Age, personal-status/sex and foreign-worker status are
excluded from predictive modelling.

The application must not be used for real lending decisions.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- UCI ML Repository

## Run Locally

```bash

pip install -r requirements.txt
streamlit run app.py 
