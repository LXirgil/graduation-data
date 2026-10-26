"""Preprocessing and fixed baseline model definitions."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    """Build train-fitted imputing, encoding, and scaling steps."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                    keep_empty_features=True,
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model(model_name, X_train, random_state=42):
    """Return a preprocessing/model pipeline with the approved fixed settings."""
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, C=1.0, class_weight=None)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            class_weight=None,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_train)),
            ("model", estimator),
        ]
    )
