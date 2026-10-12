# BankLens — Bank Marketing Model Lab

**BankLens** is an interactive machine-learning and statistical analysis application for exploring bank marketing campaign data and predicting whether a customer is likely to subscribe to a term deposit.

The project combines **data preprocessing, statistical analysis, classification models, model evaluation, interpretability, and interactive visualization** in a Streamlit dashboard.

Instead of training only one classifier, BankLens allows multiple machine-learning models to be trained, compared, evaluated, and interpreted from a single interface.

---

## 🎯 Project Objective

Banks regularly contact customers through marketing campaigns to promote products such as term deposits.

The goal of this project is to answer the following question:

> **Given information about a customer and previous marketing interactions, can we estimate whether the customer will subscribe to a term deposit?**

BankLens solves this as a **binary classification problem**:

```text
Customer subscribes     → 1
Customer does not       → 0
```

The project focuses not only on prediction accuracy, but also on understanding:

- How customer characteristics affect predictions
- How different classification algorithms behave
- Which evaluation metrics matter for imbalanced datasets
- Which features influence model decisions
- How preprocessing affects machine-learning models

---

# ✨ Main Features

## 📊 Interactive Dataset Overview

The dashboard provides an immediate summary of the bank marketing dataset.

The Overview workspace includes:

- Total customer records
- Subscription rate
- Median account balance
- Previous campaign contacts
- Campaign-response trends
- Customer behaviour visualizations
- Model readiness summary
- Model comparison preview
- Confusion matrix
- ROC curve

---

## 🗂️ Data Explorer

The **Data Explorer** allows users to inspect individual variables before model training.

For numerical variables, BankLens displays:

- Distributions
- Histograms
- Variable ranges
- Customer behaviour patterns

For categorical variables, it displays:

- Category frequencies
- Subscription rate by category
- Customer-response differences

This helps identify useful relationships before building predictive models.

---

# 🧹 Data Preprocessing

The dataset contains both:

```text
Numerical features
+
Categorical features
```

Different preprocessing methods are therefore applied to each type.

---

## Numerical Features

Numerical variables are processed using:

```text
StandardScaler
```

Standardization transforms numerical variables approximately as:

```text
z = (x - mean) / standard deviation
```

This is especially important for distance-based models such as **K-Nearest Neighbors**.

---

## Categorical Features

Categorical features are transformed using:

```text
OneHotEncoder
```

Example:

```text
contact
----------------
cellular
telephone
unknown
```

may become:

```text
contact_cellular
contact_telephone
contact_unknown
```

Unknown categories are ignored during prediction so that unseen values do not crash the application.

---

# ⚠️ Leakage Prevention

One important preprocessing decision in the project is removing:

```text
duration
```

The `duration` feature represents the duration of the marketing call.

The problem is that call duration is only known **after the call has already occurred**.

If the goal is to predict subscription before or during campaign planning, using this feature would provide information that would not actually be available at prediction time.

Therefore:

```text
duration → removed
```

This reduces **data leakage** and makes the experiment more realistic.

---

# 🔀 Train/Test Split

The dataset is divided into:

```text
80% Training Data
20% Test Data
```

using:

```python
random_state=42
stratify=y
```

Stratification preserves approximately the same proportion of subscribers and non-subscribers in both datasets.

This is particularly useful because the dataset is **imbalanced**, with substantially fewer customers subscribing than not subscribing.

---

# 🤖 Models Implemented

BankLens compares six classification algorithms.

---

## 1. Logistic Regression — L2

```text
Logistic Regression
Regularization: L2
Solver: lbfgs
```

Logistic regression provides an interpretable statistical baseline.

It models the probability that:

```text
P(y = 1 | x)
```

using the logistic function:

```text
p = 1 / (1 + e^(-z))
```

where `z` is a linear combination of the input features.

---

## 2. Lasso Logistic Regression — L1

```text
Logistic Regression
Regularization: L1
Solver: liblinear
```

L1 regularization can reduce some model coefficients exactly to zero.

This makes the model useful for:

- Feature selection
- Sparse models
- Understanding important predictors

---

## 3. K-Nearest Neighbors

```text
KNN
k = 15
```

KNN classifies a customer by examining similar customers in the training dataset.

Because KNN relies on distance calculations, numerical scaling is especially important.

---

## 4. Gaussian Naive Bayes

Naive Bayes predicts class probabilities using Bayes' theorem.

It assumes that features are conditionally independent given the class.

Although this assumption is strong, the model provides a useful probabilistic baseline.

---

## 5. Decision Tree

The Decision Tree learns a hierarchy of rules such as:

```text
previous campaign success?
        ↓
account balance?
        ↓
contact type?
        ↓
subscription prediction
```

Tree depth and leaf size are constrained to reduce overfitting.

---

## 6. Random Forest

Random Forest combines many randomized decision trees.

The project uses approximately:

```text
250 trees
```

The ensemble usually provides more stable predictions than a single decision tree.

---

# 🧪 Model Lab

The **Model Lab** allows users to select and train an individual model.

After training, BankLens displays:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
```

along with:

- Confusion matrix
- ROC curve
- Model interpretability
- Influential features

This makes it possible to study each model independently.

---

# 📏 Evaluation Metrics

Accuracy alone is not enough for this dataset because the classes are imbalanced.

BankLens therefore reports several metrics.

---

## Accuracy

```text
Accuracy =
(TP + TN) / Total Observations
```

Measures the percentage of all correct predictions.

However, accuracy can be misleading when one class strongly dominates the dataset.

---

## Precision

```text
Precision =
TP / (TP + FP)
```

Answers:

> Of the customers predicted to subscribe, how many actually subscribed?

High precision means fewer false-positive predictions.

---

## Recall

```text
Recall =
TP / (TP + FN)
```

Answers:

> Of all customers who actually subscribed, how many did the model identify?

Recall is particularly useful when missing potential subscribers is costly.

---

## F1 Score

```text
F1 =
2 × (Precision × Recall)
    ---------------------
    Precision + Recall
```

The F1 score balances precision and recall.

---

## ROC-AUC

ROC-AUC measures how well the model separates:

```text
Subscribers
vs
Non-subscribers
```

Interpretation:

```text
0.50 → approximately random classification
0.70 → useful discrimination
0.80 → strong discrimination
0.90 → excellent discrimination
1.00 → perfect separation
```

---

# 🧮 Confusion Matrix

The confusion matrix separates predictions into:

```text
                Predicted No    Predicted Yes

Actual No            TN              FP

Actual Yes           FN              TP
```

Where:

```text
TN = True Negative
FP = False Positive
FN = False Negative
TP = True Positive
```

This provides more detail than accuracy alone.

---

# 📈 ROC Curve

BankLens displays an interactive ROC curve for trained models.

The graph compares:

```text
True Positive Rate
vs
False Positive Rate
```

A random classifier follows approximately the diagonal line:

```text
y = x
```

Models whose curves move further toward the upper-left corner generally provide stronger class separation.

---

# 📊 Compare Models

The **Compare Models** workspace evaluates all available models side by side.

Metrics compared include:

```text
Precision
Recall
F1 Score
ROC-AUC
```

Interactive grouped bar charts make it easy to see which algorithms perform best according to different evaluation criteria.

This demonstrates an important machine-learning principle:

> There is not always one universally best model.

For example:

```text
Model A → better precision
Model B → better recall
Model C → better ROC-AUC
```

The preferred model depends on the business objective.

---

# 🔍 Model Interpretability

Prediction performance alone is not enough.

BankLens also attempts to explain:

> Which features influence the model's decisions?

---

## Logistic Models

For logistic regression models, BankLens displays:

```text
Model coefficients
```

A positive coefficient pushes the prediction more toward:

```text
Subscription = Yes
```

while a negative coefficient pushes it toward:

```text
Subscription = No
```

---

## Tree-Based Models

For compatible tree models, BankLens displays:

```text
Feature Importance
```

This indicates which transformed variables contribute most strongly to prediction decisions.

---

## Models Without Native Feature Importance

Some algorithms, such as:

```text
KNN
Naive Bayes
```

do not expose model coefficients or tree-style feature importance in the same way.

For such models, the interface can use a fallback interpretation strategy rather than expecting native feature importance.

---

# 👤 Customer Prediction

BankLens includes an interactive customer prediction interface.

The user enters customer information through the frontend.

The trained pipeline then performs:

```text
Customer Information
        ↓
Preprocessing
        ↓
Trained Model
        ↓
Subscription Probability
        ↓
Prediction
```

The application displays:

- Subscription probability
- Prediction decision
- Probability visualization

This demonstrates how a trained machine-learning pipeline can be used for new observations.

---

# 🏗️ Project Architecture

BankLens follows a modular architecture.

```text
                    ┌─────────────────────────┐
                    │      Raw Bank Data      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Data Layer        │
                    │      src/data.py        │
                    │                         │
                    │ Loading                 │
                    │ Cleaning                │
                    │ Leakage removal         │
                    │ Target preparation      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Preprocessing Layer   │
                    │ src/preprocessing.py    │
                    │                         │
                    │ StandardScaler          │
                    │ OneHotEncoder           │
                    │ ColumnTransformer       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Model Layer       │
                    │     src/models.py       │
                    │                         │
                    │ Logistic Regression     │
                    │ Lasso Logistic          │
                    │ KNN                     │
                    │ Naive Bayes             │
                    │ Decision Tree           │
                    │ Random Forest           │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Evaluation Layer     │
                    │   src/evaluation.py     │
                    │                         │
                    │ Train/Test Split        │
                    │ Metrics                 │
                    │ Confusion Matrix        │
                    │ ROC / AUC               │
                    │ Model Comparison        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Interpretation Layer    │
                    │ src/interpretation.py   │
                    │                         │
                    │ Coefficients            │
                    │ Feature Importance      │
                    └────────────┬────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │       Streamlit Presentation        │
              │              app.py                 │
              │                                     │
              │ Overview                            │
              │ Data Explorer                       │
              │ Model Lab                           │
              │ Compare Models                      │
              │ Customer Prediction                 │
              └─────────────────────────────────────┘
```

---


# 📄 Module Responsibilities

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit frontend, navigation, dashboards, charts, model controls and predictions |
| `src/data.py` | Dataset loading, cleaning, target conversion and leakage removal |
| `src/preprocessing.py` | Numerical scaling, categorical encoding and `ColumnTransformer` creation |
| `src/models.py` | Defines the classification algorithms |
| `src/evaluation.py` | Train/test split, model fitting, evaluation metrics, ROC and model comparison |
| `src/interpretation.py` | Model coefficients and feature-importance extraction |
| `data/bank-full.csv` | Local Bank Marketing dataset |

---

# 🔁 Machine-Learning Pipeline

A major design decision in BankLens is keeping preprocessing and the classifier together inside a machine-learning pipeline.

Conceptually:

```text
Raw Customer
     ↓
Numerical Features
     ↓
StandardScaler
     ↓
             ┐
             ├── ColumnTransformer
             │
Categorical  │
Features     │
     ↓       │
OneHotEncoder
             ┘
     ↓
Classifier
     ↓
Prediction
```

This ensures the exact same transformations are used during:

```text
Training
Testing
New customer prediction
```

---

# 🛡️ Why the Pipeline Matters

Consider a new customer:

```text
age = 43
job = technician
contact = cellular
balance = 2200
```

The classifier should not manually receive a mixture of raw text and unscaled values.

Instead:

```text
Raw customer
     ↓
Pipeline
     ↓
Scale numerical variables
     ↓
Encode categorical variables
     ↓
Apply trained classifier
     ↓
Prediction
```

This reduces preprocessing inconsistencies and makes prediction more reliable.

---

# 📚 Dataset

The project uses the **Bank Marketing dataset**.

The application can work with a local copy:

```text
data/bank-full.csv
```

and the project loader can also retrieve the Bank Marketing dataset from the UCI repository when the local dataset is unavailable.

The prediction target represents whether the customer subscribed to a term deposit.

---

# 💻 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| scikit-learn | Preprocessing, machine-learning models and evaluation |
| Streamlit | Interactive web application |
| Plotly | Interactive visualizations |
| UCI ML Repository | Bank Marketing dataset source |

---


# ▶️ Run the Application

From the project root:

```bash
streamlit run app.py
```

Streamlit will normally start the application at:

```text
http://localhost:8501
```

---

# 🧭 How to Use BankLens

## Step 1 — Explore the Dataset

Open:

```text
Overview
```

to understand:

- Dataset size
- Subscription rate
- Account balances
- Previous contacts
- Campaign patterns

---

## Step 2 — Explore Individual Features

Open:

```text
Data Explorer
```

Select a numerical or categorical feature and inspect its relationship with subscription behaviour.

---

## Step 3 — Train a Model

Open:

```text
Model Lab
```

Choose a classifier:

```text
Logistic Regression
Lasso Logistic Regression
KNN
Naive Bayes
Decision Tree
Random Forest
```

Then select:

```text
Train selected model
```

---

## Step 4 — Evaluate Performance

Review:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
```

along with:

```text
Confusion Matrix
ROC Curve
```

---

## Step 5 — Interpret the Model

Inspect the most influential transformed features.

Depending on the algorithm, BankLens uses:

```text
Coefficients
```

or:

```text
Feature Importance
```

---

## Step 6 — Compare Algorithms

Open:

```text
Compare Models
```

BankLens trains the available classifiers and displays side-by-side performance.

---

## Step 7 — Predict a New Customer

Open:

```text
Customer Prediction
```

Enter customer information and estimate the probability that the customer will subscribe.

---

# 🧠 Statistics and Data Science Concepts Covered

BankLens demonstrates practical understanding of several important topics.

## Probability

- Binary outcomes
- Predicted probabilities
- Conditional probability
- Naive Bayes

## Statistical Modeling

- Logistic regression
- Log-odds
- Model coefficients
- Regularization

## Regularization

- L1 regularization
- L2 regularization
- Sparse coefficients
- Overfitting control

## Data Preprocessing

- Numerical standardization
- Categorical encoding
- Mixed-data preprocessing
- Data leakage
- Reproducible pipelines

## Classification

- Logistic regression
- KNN
- Naive Bayes
- Decision trees
- Random forests

## Model Evaluation

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- ROC curves
- ROC-AUC
- Stratified train/test splitting

## Model Interpretation

- Logistic coefficients
- Feature importance
- Feature influence

---

# ⚠️ Class Imbalance

A central challenge in the Bank Marketing dataset is that subscriptions are relatively uncommon.

Imagine:

```text
88 customers → No subscription
12 customers → Subscription
```

A classifier predicting:

```text
No
```

for every customer could achieve approximately:

```text
88% accuracy
```

despite being useless for identifying subscribers.

This is why BankLens evaluates:

```text
Precision
Recall
F1
ROC-AUC
```

in addition to accuracy.

---

# 🔍 Why Multiple Models?

Different models learn different structures.

### Logistic Regression

```text
Linear probability boundary
+
Highly interpretable
```

### KNN

```text
Similarity between observations
```

### Naive Bayes

```text
Probabilistic assumptions
```

### Decision Tree

```text
Rule-based non-linear relationships
```

### Random Forest

```text
Ensemble learning
+
Non-linear patterns
```

Comparing them demonstrates that model selection should depend on:

```text
Predictive performance
+
Interpretability
+
Business objective
+
Computational requirements
```

---

# 🧩 Design Philosophy

BankLens separates the system into modular responsibilities.

Instead of putting everything inside `app.py`:

```text
Loading
Preprocessing
Models
Evaluation
Interpretation
Presentation
```

are separated into dedicated modules.

This makes the codebase:

- Easier to understand
- Easier to test
- Easier to extend
- Easier to debug
- More reusable

---

# 📈 Example Prediction Flow

A user enters:

```text
Age: 42
Job: Management
Balance: 3200
Contact: Cellular
Previous campaign: Success
```

BankLens processes the customer as follows:

```text
Customer Input
      ↓
ColumnTransformer
      ↓
Numerical Scaling
+
Categorical Encoding
      ↓
Trained Classifier
      ↓
Probability
      ↓
Subscription Prediction
```

Example result:

```text
Subscription probability: 71%

Prediction:
Likely to Subscribe
```

---

# 👤 Author

**Hamna Iqbal**

**BankLens — Bank Marketing Model Lab**
