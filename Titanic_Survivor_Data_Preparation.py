"""
Titanic Survivor Data Preparation Pipeline
Minor Project - Machine Learning

Purpose:
    Prepare the Seaborn Titanic dataset for downstream machine-learning use.
    This project performs data preparation only; it does not train a
    prediction model.

Functions:
    1. load_data()
    2. validate_data()
    3. clean_text()
    4. drop_useless()
    5. handle_missing()
    6. encode_categoricals()
    7. split_data()
    8. scale_features()

Run:
    python Titanic_Survivor_Data_Preparation.py

Required libraries:
    pandas, numpy, seaborn, scikit-learn
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------

class MissingColumnError(Exception):
    """Raised when one or more required dataset columns are missing."""
    pass


# ---------------------------------------------------------------------
# 1. Load the Titanic dataset
# ---------------------------------------------------------------------

def load_data():
    """Load and return the Titanic dataset from Seaborn."""
    data = sns.load_dataset("titanic")
    return data


# ---------------------------------------------------------------------
# 2. Validate required columns
# ---------------------------------------------------------------------

def validate_data(data):
    """Check that all columns required by this pipeline are present."""
    required_columns = [
        "survived",
        "pclass",
        "sex",
        "age",
        "fare",
        "embarked",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise MissingColumnError(
            f"Missing required columns: {missing_columns}"
        )

    print("Data validation successful!")
    return data


# ---------------------------------------------------------------------
# 3. Clean text columns
# ---------------------------------------------------------------------

def clean_text(data):
    """Remove surrounding spaces and convert text values to lowercase."""
    data = data.copy()

    text_columns = data.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in text_columns:
        data[column] = data[column].apply(
            lambda value: value.strip().lower()
            if isinstance(value, str) else value
        )

    return data


# ---------------------------------------------------------------------
# 4. Drop unnecessary columns
# ---------------------------------------------------------------------

def drop_useless(data):
    """Remove target-leaking, duplicate, or mostly empty columns."""
    data = data.copy()

    useless_columns = [
        "alive",        # Directly reveals the target: survived
        "class",        # Duplicate of pclass
        "embark_town",  # Duplicate of embarked
        "who",          # Redundant demographic description
        "adult_male",   # Redundant demographic indicator
        "deck",         # Contains many missing values
    ]

    data = data.drop(
        columns=useless_columns,
        errors="ignore"
    )

    return data


# ---------------------------------------------------------------------
# 5. Handle missing values
# ---------------------------------------------------------------------

def handle_missing(data):
    """Fill numeric columns with their median and categories with mode."""
    data = data.copy()

    numeric_columns = data.select_dtypes(
        include=["number"]
    ).columns

    categorical_columns = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for column in numeric_columns:
        median_value = data[column].median()
        data[column] = data[column].fillna(median_value)

    for column in categorical_columns:
        mode_values = data[column].mode()
        if not mode_values.empty:
            data[column] = data[column].fillna(mode_values[0])

    return data


# ---------------------------------------------------------------------
# 6. Encode categorical columns
# ---------------------------------------------------------------------

def encode_categoricals(data):
    """
    Encode sex as binary values, map embarked categories, and one-hot encode
    the mapped embarked values.
    """
    data = data.copy()

    # Binary encoding: male = 0, female = 1
    data["sex"] = data["sex"].map({
        "male": 0,
        "female": 1,
    })

    # Ordinal mapping for embarkation categories
    data["embarked"] = data["embarked"].map({
        "s": 1,
        "c": 2,
        "q": 3,
    })

    # One-hot encoding
    data = pd.get_dummies(
        data,
        columns=["embarked"],
        dtype=int
    )

    return data


# ---------------------------------------------------------------------
# 7. Split data into training and testing sets
# ---------------------------------------------------------------------

def split_data(data):
    """Separate target from features and create an 80/20 stratified split."""
    X = data.drop(columns=["survived"])
    y = data["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# 8. Scale features
# ---------------------------------------------------------------------

def scale_features(X_train, X_test):
    """
    Fit StandardScaler on training features only, then transform both sets.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------
# Main function: run the complete pipeline
# ---------------------------------------------------------------------

def main():
    """Run all eight data-preparation functions in the correct order."""
    try:
        print("Starting Titanic Data Preparation Pipeline...")

        # 1. Load data
        data = load_data()
        print("Dataset loaded:", data.shape)

        # 2. Validate data
        data = validate_data(data)

        # 3. Clean text
        data = clean_text(data)

        # 4. Drop unnecessary columns
        data = drop_useless(data)
        print("After dropping unnecessary columns:", data.shape)

        # 5. Handle missing values
        data = handle_missing(data)
        print("Remaining missing values:", int(data.isnull().sum().sum()))

        # 6. Encode categorical data
        data = encode_categoricals(data)
        print("After categorical encoding:", data.shape)

        # 7. Split data
        X_train, X_test, y_train, y_test = split_data(data)

        # 8. Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(
            X_train,
            X_test
        )

        print("\nPIPELINE COMPLETED SUCCESSFULLY!")
        print("Training features:", X_train_scaled.shape)
        print("Testing features:", X_test_scaled.shape)
        print("Training target:", y_train.shape)
        print("Testing target:", y_test.shape)

        # Return prepared data for optional downstream use
        return (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            scaler,
        )

    except Exception as error:
        print("Pipeline Error:", error)
        return None


if __name__ == "__main__":
    results = main()
