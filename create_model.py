import json
import pathlib
import pickle
from typing import List
from typing import Tuple

import pandas
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os 
os.chdir('/Users/charlenehack/Desktop/Sam/fastapi_docker_ml')

SALES_PATH = "kc_house_data.csv"  # path to CSV with home sale data

#this was the wrong file in the original model creation script!
DEMOGRAPHICS_PATH = "zipcode_demographics.csv"  # path to CSV with demographics

# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "original_model"  # Directory where output artifacts will be saved

# Directory for saving accuracy scores
SCORES_DIR = "original_model"

def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv("zipcode_demographics.csv",
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=42)

    model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                   neighbors.KNeighborsRegressor()).fit(
                                       x_train, y_train)

    # Calculate the accuracy of the model using mean squared error
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared (R2) Score: {r2}")


    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "original_model.pkl", 'wb'))
    json.dump(list(x_train.columns),
              open(output_dir / "model_features.json", 'w'))

    scores_dir = pathlib.Path(SCORES_DIR)
    scores_dir.mkdir(exist_ok=True)

    # Save accuracy scores to a file
    scores = {
        "Mean Squared Error": mse,
        "Mean Absolute Error": mae,
        "R-squared (R2) Score": r2
    }

    with open(scores_dir / "accuracy_scores.json", 'w') as scores_file:
        json.dump(scores, scores_file)

if __name__ == "__main__":
    main()
