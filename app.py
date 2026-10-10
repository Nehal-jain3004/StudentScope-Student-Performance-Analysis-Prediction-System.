"""
=============================================================================
StudentScope — Flask REST API & Web Server
=============================================================================
Serves the existing frontend UI and provides REST endpoints powered by:
1. Genuine dataset calculations from student.csv
2. Real machine learning pipeline from student_model.pkl
3. Model evaluation benchmarks from model_metrics.json
=============================================================================
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

# Paths to dataset and trained model artifacts
DATASET_PATH = os.path.join(BASE_DIR, "student.csv")
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

# Global resource holders
dataset_df = None
model_pipeline = None
metrics_data = None


def load_resources():
    """Load dataset, model pipeline, and metrics JSON into memory."""
    global dataset_df, model_pipeline, metrics_data

    # 1. Load Dataset
    if os.path.exists(DATASET_PATH):
        try:
            dataset_df = pd.read_csv(DATASET_PATH)
            print(f"[SUCCESS] Loaded student.csv ({len(dataset_df)} rows)")
        except Exception as e:
            print(f"[ERROR] Failed to read student.csv: {e}")
    else:
        print(f"[WARNING] student.csv not found at {DATASET_PATH}")

    # 2. Load Trained Scikit-Learn Pipeline
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            print(f"[SUCCESS] Loaded student_model.pkl")
        except Exception as e:
            print(f"[ERROR] Failed to load student_model.pkl: {e}")
    else:
        print(f"[WARNING] student_model.pkl not found at {MODEL_PATH}")

    # 3. Load Metrics JSON
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            print(f"[SUCCESS] Loaded model_metrics.json")
        except Exception as e:
            print(f"[ERROR] Failed to read model_metrics.json: {e}")
    else:
        print(f"[WARNING] model_metrics.json not found at {METRICS_PATH}")


# Initialize resources upon startup
load_resources()


# =============================================================================
# FRONTEND STATIC ROUTES
# =============================================================================
@app.route("/")
def serve_index():
    """Serve the main index.html file."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve existing static assets (style.css, script.js, etc.)."""
    return send_from_directory(BASE_DIR, filename)


# =============================================================================
# API ROUTE 1: GET /api/dashboard-data
# Computes genuine summary statistics and chart series from student.csv
# =============================================================================
@app.route("/api/dashboard-data", methods=["GET"])
def get_dashboard_data():
    if dataset_df is None:
        return jsonify({
            "status": "error",
            "message": "student.csv is not loaded on the server"
        }), 500

    try:
        df = dataset_df

        # --- 1. Summary Metric Cards ---
        total_students = int(len(df))
        avg_final_grade = round(float(df["G3"].mean()), 2)
        avg_study_hours_tier = round(float(df["studytime"].mean()), 2)
        passing_count = int((df["G3"] >= 10).sum())
        pass_rate = round((passing_count / total_students) * 100, 1) if total_students > 0 else 0.0

        # --- 2. Grade Distribution (G3 Final) ---
        fail_count = int((df["G3"] < 10).sum())
        avg_count = int(((df["G3"] >= 10) & (df["G3"] <= 12)).sum())
        good_count = int(((df["G3"] >= 13) & (df["G3"] <= 15)).sum())
        distinction_count = int((df["G3"] >= 16).sum())

        grade_distribution = {
            "labels": ["Fail (<10)", "Average (10-12)", "Good (13-15)", "Distinction (16-20)"],
            "counts": [fail_count, avg_count, good_count, distinction_count]
        }

        # --- 3. Study Time vs Final Grade ---
        # Study tiers: 1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h)
        study_tiers = [1, 2, 3, 4]
        tier_labels = ["< 2 Hours", "2 - 5 Hours", "5 - 10 Hours", "> 10 Hours"]
        study_g3_means = []
        study_g1_means = []

        for tier in study_tiers:
            subset = df[df["studytime"] == tier]
            if len(subset) > 0:
                study_g3_means.append(round(float(subset["G3"].mean()), 2))
                study_g1_means.append(round(float(subset["G1"].mean()), 2))
            else:
                study_g3_means.append(0.0)
                study_g1_means.append(0.0)

        study_vs_grade = {
            "labels": tier_labels,
            "g3_means": study_g3_means,
            "g1_means": study_g1_means
        }

        # --- 4. Grade Progression (G1 -> G2 -> G3) ---
        top_subset = df[df["G3"] >= 15]
        avg_subset = df[(df["G3"] >= 10) & (df["G3"] < 15)]
        risk_subset = df[df["G3"] < 10]

        progression = {
            "labels": ["Period 1 (G1)", "Period 2 (G2)", "Final Grade (G3)"],
            "top_performers": [
                round(float(top_subset["G1"].mean()), 2) if len(top_subset) else 0.0,
                round(float(top_subset["G2"].mean()), 2) if len(top_subset) else 0.0,
                round(float(top_subset["G3"].mean()), 2) if len(top_subset) else 0.0
            ],
            "avg_performers": [
                round(float(avg_subset["G1"].mean()), 2) if len(avg_subset) else 0.0,
                round(float(avg_subset["G2"].mean()), 2) if len(avg_subset) else 0.0,
                round(float(avg_subset["G3"].mean()), 2) if len(avg_subset) else 0.0
            ],
            "risk_performers": [
                round(float(risk_subset["G1"].mean()), 2) if len(risk_subset) else 0.0,
                round(float(risk_subset["G2"].mean()), 2) if len(risk_subset) else 0.0,
                round(float(risk_subset["G3"].mean()), 2) if len(risk_subset) else 0.0
            ]
        }

        # --- 5. Absences vs Final Grade (Scatter Data) ---
        # Sample points to keep browser rendering responsive while genuine
        scatter_sample = df[["absences", "G3"]].copy()
        scatter_points = [
            {"x": int(row["absences"]), "y": int(row["G3"])}
            for _, row in scatter_sample.iterrows()
        ]

        # --- 6. Study Habits & Support Radar Profile ---
        max_abs = max(1, df["absences"].max())
        high_group = df[df["G3"] >= 15]
        low_group = df[df["G3"] < 10]

        def compute_radar_profile(sub):
            if len(sub) == 0:
                return [0, 0, 0, 0, 0, 0]
            w_study = round((sub["studytime"].mean() / 4.0) * 100, 1)
            g1_pct = round((sub["G1"].mean() / 20.0) * 100, 1)
            g2_pct = round((sub["G2"].mean() / 20.0) * 100, 1)
            att_rate = round(max(0, 100 - (sub["absences"].mean() / max_abs * 100)), 1)
            supp_rate = round((sub["schoolsup"] == "yes").mean() * 100, 1) if "schoolsup" in sub else 50.0
            high_rate = round((sub["higher"] == "yes").mean() * 100, 1) if "higher" in sub else 50.0
            return [w_study, g1_pct, g2_pct, att_rate, supp_rate, high_rate]

        radar_data = {
            "labels": ["Weekly Study Time", "G1 Score", "G2 Score", "Attendance Rate", "Extra Tutoring", "Higher Edu Desire"],
            "high_achievers": compute_radar_profile(high_group),
            "at_risk_group": compute_radar_profile(low_group)
        }

        # --- 7. Failures Impact ---
        failure_labels = ["0 Failures", "1 Failure", "2 Failures", "3+ Failures"]
        failure_means = []
        for f_count in [0, 1, 2, 3]:
            sub_f = df[df["failures"] >= f_count] if f_count == 3 else df[df["failures"] == f_count]
            if len(sub_f) > 0:
                failure_means.append(round(float(sub_f["G3"].mean()), 2))
            else:
                failure_means.append(0.0)

        failures_impact = {
            "labels": failure_labels,
            "means": failure_means
        }

        # --- 8. Student Records Table Data ---
        study_time_map = {1: "< 2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: "> 10 hrs"}
        table_records = []
        for idx, row in df.iterrows():
            g3_val = int(row["G3"])
            status = "pass" if g3_val >= 14 else ("average" if g3_val >= 10 else "risk")
            table_records.append({
                "id": f"STU-{1001 + idx}",
                "age": int(row["age"]),
                "studyTime": study_time_map.get(int(row["studytime"]), "2-5 hrs"),
                "absences": int(row["absences"]),
                "failures": int(row["failures"]),
                "g1": int(row["G1"]),
                "g2": int(row["G2"]),
                "g3": g3_val,
                "schoolsup": str(row.get("schoolsup", "no")),
                "higher": str(row.get("higher", "yes")),
                "status": status
            })

        return jsonify({
            "status": "success",
            "summary": {
                "total_students": total_students,
                "avg_final_grade": avg_final_grade,
                "avg_study_hours_tier": avg_study_hours_tier,
                "pass_rate": pass_rate,
                "passing_count": passing_count
            },
            "charts": {
                "grade_distribution": grade_distribution,
                "study_vs_grade": study_vs_grade,
                "progression": progression,
                "absences_scatter": scatter_points,
                "radar_profile": radar_data,
                "failures_impact": failures_impact
            },
            "students": table_records
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to calculate dashboard statistics: {str(e)}"
        }), 500


# =============================================================================
# API ROUTE 2: GET /api/model-insights
# Returns actual evaluation metrics saved in model_metrics.json
# =============================================================================
@app.route("/api/model-insights", methods=["GET"])
def get_model_insights():
    if metrics_data is None:
        return jsonify({
            "status": "error",
            "message": "model_metrics.json not loaded on the server"
        }), 500

    return jsonify({
        "status": "success",
        "data": metrics_data
    })


# =============================================================================
# API ROUTE 3: POST /api/predict
# Validates input attributes and evaluates fitted pipeline from student_model.pkl
# =============================================================================
@app.route("/api/predict", methods=["POST"])
def predict_performance():
    if model_pipeline is None:
        return jsonify({
            "status": "error",
            "message": "Model pipeline is not loaded on the server."
        }), 500

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({
            "status": "error",
            "message": "Invalid JSON payload or missing body."
        }), 400

    # 1. Validation Rules
    errors = {}

    # Age: 15 to 22
    try:
        age = int(data.get("age", data.get("studentAge", 0)))
        if age < 15 or age > 22:
            errors["age"] = "Age must be between 15 and 22."
    except (ValueError, TypeError):
        errors["age"] = "Age must be a valid integer."

    # Study Time: 1 to 4
    try:
        studytime = int(data.get("studytime", data.get("studyTime", 0)))
        if studytime < 1 or studytime > 4:
            errors["studytime"] = "Study time tier must be between 1 and 4."
    except (ValueError, TypeError):
        errors["studytime"] = "Study time must be a valid integer."

    # Absences: 0 to 30
    try:
        absences = int(data.get("absences", -1))
        if absences < 0 or absences > 30:
            errors["absences"] = "Absences must be between 0 and 30."
    except (ValueError, TypeError):
        errors["absences"] = "Absences must be a valid integer."

    # Failures: 0 to 3
    try:
        failures = int(data.get("failures", -1))
        if failures < 0 or failures > 3:
            errors["failures"] = "Failures must be between 0 and 3."
    except (ValueError, TypeError):
        errors["failures"] = "Failures must be a valid integer."

    # G1: 0 to 20
    try:
        g1 = float(data.get("G1", data.get("g1Grade", -1)))
        if g1 < 0.0 or g1 > 20.0:
            errors["G1"] = "Period 1 (G1) score must be between 0 and 20."
    except (ValueError, TypeError):
        errors["G1"] = "Period 1 score must be a valid number."

    # G2: 0 to 20
    try:
        g2 = float(data.get("G2", data.get("g2Grade", -1)))
        if g2 < 0.0 or g2 > 20.0:
            errors["G2"] = "Period 2 (G2) score must be between 0 and 20."
    except (ValueError, TypeError):
        errors["G2"] = "Period 2 score must be a valid number."

    # Schoolsup: 'yes' or 'no'
    schoolsup = str(data.get("schoolsup", "no")).strip().lower()
    if schoolsup not in ["yes", "no"]:
        errors["schoolsup"] = "schoolsup must be 'yes' or 'no'."

    # Higher: 'yes' or 'no'
    higher = str(data.get("higher", "yes")).strip().lower()
    if higher not in ["yes", "no"]:
        errors["higher"] = "higher must be 'yes' or 'no'."

    if errors:
        return jsonify({
            "status": "error",
            "message": "Validation failed",
            "errors": errors
        }), 400

    # 2. Construct Input DataFrame with exact column names and expected types
    input_df = pd.DataFrame([{
        "age": age,
        "studytime": studytime,
        "absences": absences,
        "failures": failures,
        "G1": g1,
        "G2": g2,
        "schoolsup": schoolsup,
        "higher": higher
    }])

    try:
        # Run inference through fitted scikit-learn pipeline
        raw_prediction = float(model_pipeline.predict(input_df)[0])
        # Clamp score between [0, 20]
        final_g3 = round(max(0.0, min(20.0, raw_prediction)), 2)
        percentage = round((final_g3 / 20.0) * 100, 1)

        # Performance Tier Classification
        if final_g3 >= 14.0:
            tier = "good"
            tier_label = "High Performer (Distinction)"
        elif final_g3 >= 10.0:
            tier = "average"
            tier_label = "Passing (Average)"
        else:
            tier = "risk"
            tier_label = "At Risk (Below Threshold)"

        # Dynamic Contributing Factors Breakdown
        factors = []

        # G2 factor
        if g2 >= 13:
            factors.append({
                "type": "positive",
                "text": f"Strong Period 2 Score: Score of {g2}/20 is the leading positive factor."
            })
        elif g2 < 10:
            factors.append({
                "type": "negative",
                "text": f"Low Period 2 Score: Score of {g2}/20 significantly impacts final prediction."
            })

        # Study time factor
        study_tier_desc = {1: "< 2 hrs/wk", 2: "2-5 hrs/wk", 3: "5-10 hrs/wk", 4: "> 10 hrs/wk"}
        if studytime >= 3:
            factors.append({
                "type": "positive",
                "text": f"Dedicated Study Habit: {study_tier_desc.get(studytime)} supports solid academic retention."
            })
        elif studytime == 1:
            factors.append({
                "type": "negative",
                "text": f"Limited Study Time: {study_tier_desc.get(studytime)} indicates vulnerability for final exams."
            })

        # Absences factor
        if absences > 10:
            factors.append({
                "type": "negative",
                "text": f"High Absences Penalty: {absences} missed school days heavily suppresses score."
            })
        elif absences <= 3:
            factors.append({
                "type": "positive",
                "text": f"High Attendance: Only {absences} absences maintains course continuity."
            })

        # Failures factor
        if failures > 0:
            factors.append({
                "type": "negative",
                "text": f"Historical Failures: {failures} previous failure(s) recorded."
            })

        # Fallback factor if empty
        if not factors:
            factors.append({
                "type": "positive" if final_g3 >= 10 else "negative",
                "text": f"Baseline performance consistent with term indicators."
            })

        # Actionable Recommendation
        if final_g3 >= 14.0:
            recommendation = "Student displays strong mastery across evaluation terms. Recommend maintaining study cadence and preparing for advanced curriculum."
        elif final_g3 >= 10.0:
            recommendation = "Student is in passing standing. Adding 2-3 weekly study hours and limiting further absences will help secure a Distinction tier."
        else:
            recommendation = "Student is at risk of failing the term (G3 < 10). Immediate academic intervention, extra tutoring sessions, and attendance monitoring are recommended."

        selected_model_name = metrics_data.get("selected_model", "Random Forest Regressor") if metrics_data else "Random Forest Regressor"

        return jsonify({
            "status": "success",
            "prediction": {
                "score": final_g3,
                "percentage": percentage,
                "tier": tier,
                "tier_label": tier_label,
                "model_used": selected_model_name,
                "factors": factors,
                "recommendation": recommendation
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Model inference failed: {str(e)}"
        }), 500


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Strictly disable debug mode in production (can be enabled via FLASK_DEBUG=1 for development)
    debug_mode = os.environ.get("FLASK_DEBUG", "0").strip().lower() in ["1", "true"]
    print(f"\n==================================================================")
    print(f"  StudentScope Production Web Server")
    print(f"  Host: 0.0.0.0 | Port: {port} | Debug: {debug_mode}")
    print(f"  Local access: http://127.0.0.1:{port}")
    print(f"==================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
