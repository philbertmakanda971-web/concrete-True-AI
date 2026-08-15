import os
import pickle
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# PROJECT DIRECTORIES
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "concrete_scaler.pkl"
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "model_weights.pkl"
)

PUBLIC_DIR = os.path.join(
    BASE_DIR,
    "public"
)


# ==========================================================
# CHECK FILES
# ==========================================================

print("=" * 60)
print("CONCRETE AI - NUMPY PREDICTION API")
print("=" * 60)

print("Project folder:")
print(BASE_DIR)

print("Scaler:")
print(SCALER_PATH)

print("Weights:")
print(WEIGHTS_PATH)

print("Frontend:")
print(PUBLIC_DIR)


if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"concrete_scaler.pkl was not found:\n{SCALER_PATH}"
    )


if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"model_weights.pkl was not found:\n{WEIGHTS_PATH}"
    )


if not os.path.exists(
    os.path.join(PUBLIC_DIR, "index.html")
):
    raise FileNotFoundError(
        f"index.html was not found:\n"
        f"{os.path.join(PUBLIC_DIR, 'index.html')}"
    )


print("Scaler found!")
print("Model weights found!")
print("Frontend found!")


# ==========================================================
# LOAD SCALER
# ==========================================================

print("Loading scaler...")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("Scaler loaded successfully!")


# ==========================================================
# LOAD MODEL WEIGHTS
# ==========================================================

print("Loading neural-network weights...")

with open(WEIGHTS_PATH, "rb") as f:
    weights = pickle.load(f)

print("Model weights loaded successfully!")

print("Layers:")
for layer_name in weights:
    print("-", layer_name)


# ==========================================================
# ACTIVATION FUNCTIONS
# ==========================================================

def relu(x):
    return np.maximum(0, x)


# ==========================================================
# BATCH NORMALIZATION
# ==========================================================

def batch_normalization(
    x,
    gamma,
    beta,
    moving_mean,
    moving_variance,
    epsilon=0.001
):

    return (
        gamma
        * (
            (x - moving_mean)
            / np.sqrt(moving_variance + epsilon)
        )
        + beta
    )


# ==========================================================
# NUMPY MODEL PREDICTION
# ==========================================================

def predict_numpy(x):

    # ------------------------------------------------------
    # DENSE
    # ------------------------------------------------------

    kernel, bias = weights["dense"]

    x = np.dot(x, np.array(kernel)) + np.array(bias)

    x = relu(x)


    # ------------------------------------------------------
    # BATCH NORMALIZATION
    # ------------------------------------------------------

    gamma, beta, moving_mean, moving_variance = (
        weights["batch_normalization"]
    )

    x = batch_normalization(
        x,
        np.array(gamma),
        np.array(beta),
        np.array(moving_mean),
        np.array(moving_variance)
    )


    # ------------------------------------------------------
    # DENSE 1
    # ------------------------------------------------------

    kernel, bias = weights["dense_1"]

    x = np.dot(x, np.array(kernel)) + np.array(bias)

    x = relu(x)


    # ------------------------------------------------------
    # BATCH NORMALIZATION 1
    # ------------------------------------------------------

    gamma, beta, moving_mean, moving_variance = (
        weights["batch_normalization_1"]
    )

    x = batch_normalization(
        x,
        np.array(gamma),
        np.array(beta),
        np.array(moving_mean),
        np.array(moving_variance)
    )


    # ------------------------------------------------------
    # DENSE 2
    # ------------------------------------------------------

    kernel, bias = weights["dense_2"]

    x = np.dot(x, np.array(kernel)) + np.array(bias)

    x = relu(x)


    # ------------------------------------------------------
    # DENSE 3
    # ------------------------------------------------------

    kernel, bias = weights["dense_3"]

    x = np.dot(x, np.array(kernel)) + np.array(bias)

    x = relu(x)


    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------

    kernel, bias = weights["dense_4"]

    x = np.dot(x, np.array(kernel)) + np.array(bias)


    return float(x[0][0])


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/", methods=["GET"])
def home():

    return send_from_directory(
        PUBLIC_DIR,
        "index.html"
    )


# ==========================================================
# PREDICTION API
# ==========================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        print("=" * 60)
        print("PREDICTION REQUEST RECEIVED")
        print("=" * 60)


        # --------------------------------------------------
        # GET JSON
        # --------------------------------------------------

        data = request.get_json(
            silent=True
        )

        if data is None:

            return jsonify({
                "success": False,
                "error": "No JSON data was received."
            }), 400


        # --------------------------------------------------
        # READ INPUT VALUES
        # --------------------------------------------------

        cement = float(
            data["cement"]
        )

        slag = float(
            data["slag"]
        )

        flyash = float(
            data["flyash"]
        )

        water = float(
            data["water"]
        )

        superplasticizer = float(
            data["superplasticizer"]
        )

        coarse = float(
            data["coarse"]
        )

        fine = float(
            data["fine"]
        )

        age = float(
            data["age"]
        )


        # --------------------------------------------------
        # VALUES IN TRAINING ORDER
        # --------------------------------------------------

        values = [
            cement,
            slag,
            flyash,
            water,
            superplasticizer,
            coarse,
            fine,
            age
        ]


        # --------------------------------------------------
        # CREATE DATAFRAME
        # --------------------------------------------------

        if hasattr(
            scaler,
            "feature_names_in_"
        ):

            expected_columns = list(
                scaler.feature_names_in_
            )

            input_data = pd.DataFrame(
                [values],
                columns=expected_columns
            )

        else:

            input_data = pd.DataFrame(
                [values],
                columns=[
                    "Cement (component 1)(kg in a m^3 mixture)",
                    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
                    "Fly Ash (component 3)(kg in a m^3 mixture)",
                    "Water  (component 4)(kg in a m^3 mixture)",
                    "Superplasticizer (component 5)(kg in a m^3 mixture)",
                    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
                    "Fine Aggregate (component 7)(kg in a m^3 mixture)",
                    "Age (day)"
                ]
            )


        # --------------------------------------------------
        # SCALE INPUT
        # --------------------------------------------------

        input_scaled = scaler.transform(
            input_data
        )

        print("Input scaled successfully!")


        # --------------------------------------------------
        # NUMPY PREDICTION
        # --------------------------------------------------

        strength = predict_numpy(
            input_scaled
        )

        print(
            "Predicted strength:",
            strength
        )


        # --------------------------------------------------
        # GRADE
        # --------------------------------------------------

        if strength < 20:

            grade = "Below C20"

        elif strength < 25:

            grade = "C20"

        elif strength < 30:

            grade = "C25"

        elif strength < 35:

            grade = "C30"

        elif strength < 40:

            grade = "C35"

        elif strength < 50:

            grade = "C40"

        else:

            grade = "High Strength Concrete"


        # --------------------------------------------------
        # RECOMMENDATION
        # --------------------------------------------------

        if strength < 20:

            recommendation = (
                "Concrete strength is low. "
                "Further mix design and laboratory "
                "testing are recommended."
            )

        elif strength < 30:

            recommendation = (
                "The predicted strength may be suitable "
                "for light structural applications, "
                "subject to engineering verification."
            )

        elif strength < 40:

            recommendation = (
                "The predicted strength may be suitable "
                "for reinforced concrete applications, "
                "subject to structural design and "
                "laboratory verification."
            )

        else:

            recommendation = (
                "The predicted strength indicates "
                "relatively high-strength concrete. "
                "Final structural use should be confirmed "
                "through laboratory testing and design."
            )


        # --------------------------------------------------
        # RETURN RESULT
        # --------------------------------------------------

        return jsonify({

            "success": True,

            "strength": round(
                strength,
                2
            ),

            "grade": grade,

            "recommendation": recommendation

        }), 200


    # ======================================================
    # MISSING FIELD
    # ======================================================

    except KeyError as e:

        return jsonify({

            "success": False,

            "error":
            f"Missing input field: {str(e)}"

        }), 400


    # ======================================================
    # INVALID VALUE
    # ======================================================

    except ValueError as e:

        return jsonify({

            "success": False,

            "error":
            f"Invalid input value: {str(e)}"

        }), 400


    # ======================================================
    # OTHER ERROR
    # ======================================================

    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ==========================================================
# LOCAL DEVELOPMENT
# ==========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )