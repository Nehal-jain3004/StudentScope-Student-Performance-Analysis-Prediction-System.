"""
=============================================================================
StudentScope — Machine Learning Model Training & Evaluation Pipeline
=============================================================================
This script:
1. Loads student.csv from the local directory.
2. Extracts target variable G3 and 8 predictor features:
   - Numeric: age, studytime, absences, failures, G1, G2
   - Categorical: schoolsup, higher
3. Performs an 80/20 train/test split with random_state=42.
4. Builds a scikit-learn Pipeline with median imputation + scaling for numeric
   features and most-frequent imputation + one-hot encoding for categorical features.
5. Trains and compares 4 regression algorithms:
   - Linear Regression
   - Lasso Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
6. Evaluates R², MAE, and RMSE on the held-out test set.
7. Automatically selects the top-performing model across all 3 metrics.
8. Exports the fitted pipeline to 'student_model.pkl' via joblib.
9. Exports true evaluation metrics and actual feature importances to 'model_metrics.json'.
10. Prints a clear comparison table and verification test prediction.
=============================================================================
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Compatible RMSE import across scikit-learn versions
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    print("=" * 72)
    print("  StudentScope — Machine Learning Model Training Pipeline")
    print("=" * 72)

    # 1. Locate and Load Dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_candidates = [
        os.path.join(current_dir, "student.csv"),
        os.path.join(os.path.dirname(current_dir), "student.csv")
    ]

    csv_path = None
    for path in csv_candidates:
        if os.path.exists(path):
            csv_path = path
            break

    if not csv_path:
        print("\n[ERROR] student.csv not found in the current or parent directory.")
        print(f"Checked paths:\n  - {csv_candidates[0]}\n  - {csv_candidates[1]}")
        sys.exit(1)

    print(f"\n[1/7] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"      Total records loaded: {len(df)} rows, {len(df.columns)} columns.")

    # 2. Validate Target & Features
    target_col = "G3"
    if target_col not in df.columns:
        print(f"\n[ERROR] Target column '{target_col}' not found in dataset columns.")
        sys.exit(1)

    candidate_num_features = ["age", "studytime", "absences", "failures", "G1", "G2"]
    candidate_cat_features = ["schoolsup", "higher"]

    num_features = [f for f in candidate_num_features if f in df.columns]
    cat_features = [f for f in candidate_cat_features if f in df.columns]
    all_features = num_features + cat_features

    print(f"\n[2/7] Feature verification:")
    print(f"      Target variable (y): {target_col}")
    print(f"      Numeric features   : {num_features}")
    print(f"      Categorical features: {cat_features}")
    print(f"      Total predictor features: {len(all_features)}")

    # Ensure G3 is never in the predictor matrix
    if target_col in all_features:
        all_features.remove(target_col)

    X = df[all_features].copy()
    y = df[target_col].copy()

    # 3. Train-Test Split (80/20, random_state=42)
    print("\n[3/7] Splitting dataset (80% Train, 20% Test, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"      Training set size: {len(X_train)} samples")
    print(f"      Testing set size : {len(X_test)} samples")

    # 4. Construct Scikit-Learn Preprocessing Pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ],
        remainder="drop"
    )

    # 5. Define Candidate Models
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # 6. Train, Evaluate & Compare
    print("\n[4/7] Training and evaluating candidate regression models...")
    evaluation_results = {}
    fitted_pipelines = {}

    for name, model in candidate_models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Fit strictly on X_train, y_train
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Evaluate strictly on held-out X_test, y_test
        y_pred = pipe.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))

        evaluation_results[name] = {
            "r2": round(r2, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4)
        }

    # 7. Print Model Comparison Table
    print("\n" + "=" * 72)
    print(f"{'Model Name':<30} | {'R² Score':<10} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 72)
    for name, metrics in evaluation_results.items():
        print(f"{name:<30} | {metrics['r2']:<10.4f} | {metrics['mae']:<10.4f} | {metrics['rmse']:<10.4f}")
    print("=" * 72)

    # 8. Model Selection Logic
    # Rank models primarily by highest R², verified by lowest RMSE and lowest MAE
    sorted_models = sorted(
        evaluation_results.items(),
        key=lambda item: (item[1]["r2"], -item[1]["rmse"], -item[1]["mae"]),
        reverse=True
    )
    best_model_name, best_metrics = sorted_models[0]
    best_pipeline = fitted_pipelines[best_model_name]

    print(f"\n[5/7] Model Selection Decision:")
    print(f"      Selected Winner: {best_model_name}")
    print(f"      Best Test R²   : {best_metrics['r2']}")
    print(f"      Best Test MAE  : {best_metrics['mae']}")
    print(f"      Best Test RMSE : {best_metrics['rmse']}")

    # 9. Extract Feature Importances (if supported by the selected model)
    underlying_model = best_pipeline.named_steps["model"]
    feature_importances_data = None

    if hasattr(underlying_model, "feature_importances_"):
        print("\n[6/7] Extracting actual feature importances from selected model...")
        prep_fitted = best_pipeline.named_steps["preprocessor"]
        transformed_feature_names = prep_fitted.get_feature_names_out()

        # Clean column prefixes (e.g., 'num__G2' -> 'G2', 'cat__schoolsup_yes' -> 'schoolsup_yes')
        clean_names = [f.split("__")[-1] for f in transformed_feature_names]
        raw_importances = underlying_model.feature_importances_

        feature_importances_data = []
        for feat_name, imp_val in sorted(zip(clean_names, raw_importances), key=lambda x: x[1], reverse=True):
            feature_importances_data.append({
                "feature": feat_name,
                "importance": round(float(imp_val), 4),
                "importance_percentage": round(float(imp_val) * 100, 2)
            })

        print("      Top Predictor Weights:")
        for item in feature_importances_data[:6]:
            print(f"        - {item['feature']:<18}: {item['importance_percentage']}%")
    else:
        print("\n[6/7] Selected model does not expose feature_importances_ (feature importances set to null).")

    # 10. Save Model Artifact (.pkl) and Metrics (.json)
    output_model_path = os.path.join(current_dir, "student_model.pkl")
    output_metrics_path = os.path.join(current_dir, "model_metrics.json")

    joblib.dump(best_pipeline, output_model_path)
    print(f"\n[7/7] Artifacts Exported:")
    print(f"      - Fitted pipeline saved to : {output_model_path}")

    # Prepare structured JSON metrics output
    metrics_payload = {
        "models": evaluation_results,
        "selected_model": best_model_name,
        "selected_metrics": best_metrics,
        "feature_importances": feature_importances_data,
        "features": {
            "all": all_features,
            "numeric": num_features,
            "categorical": cat_features
        },
        "target": target_col,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state": 42
    }

    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"      - Real metrics saved to   : {output_metrics_path}")

    # 11. Verification Test Prediction on Sample Input DataFrame
    print("\n" + "-" * 72)
    print("  Verification Test: Running prediction on sample input DataFrame...")
    sample_student = pd.DataFrame([{
        "age": 17,
        "studytime": 2,
        "absences": 4,
        "failures": 0,
        "G1": 12,
        "G2": 13,
        "schoolsup": "no",
        "higher": "yes"
    }])

    loaded_pipe = joblib.load(output_model_path)
    test_prediction = float(loaded_pipe.predict(sample_student)[0])
    print(f"  Input: Age=17, StudyTime=2 (2-5h), Absences=4, Failures=0, G1=12, G2=13, Supp=no, Higher=yes")
    print(f"  Predicted Final Grade G3: {test_prediction:.2f} / 20 ({test_prediction / 20 * 100:.1f}%)")
    print("-" * 72)
    print("\n[SUCCESS] Model training and evaluation completed successfully.\n")


if __name__ == "__main__":
    main()
