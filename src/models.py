from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def available_models():

    return {

        # Statistical model
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000
            )
        ),

        # Logistic Regression with L1 regularization
        "Lasso Logistic Regression": (
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                max_iter=2000,
            )
        ),

        # Distance-based ML
        "KNN": (
            KNeighborsClassifier(
                n_neighbors=15
            )
        ),

        # Probability-based classifier
        "Naive Bayes": (
            GaussianNB()
        ),

        # Tree-based ML
        "Decision Tree": (
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=20,
                random_state=42,
            )
        ),

        # Ensemble model
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        ),
    }