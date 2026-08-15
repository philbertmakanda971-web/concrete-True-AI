import os
import pickle
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# PROJECT DIRECTORY
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "concrete_model.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "concrete_scaler.pkl"
)

PUBLIC_DIR = os.path.join(
    BASE_DIR,
    "public"
)


# ==========================================================
# CHECK REQUIRED FILES
# ==========================================================

print("=" * 60)
print("CONCRETE AI API")
print("=" * 60)

print("Project folder:")
print(BASE_DIR)

print("\nModel:")
print(MODEL_PATH)

print("\nScaler:")
print(SCALER_PATH)

print("\nFrontend:")
print(PUBLIC_DIR)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\nconcrete_model.keras was not found.\n"
        f"Expected location:\n{MODEL_PATH}"
    )


if not os.path.exists(SCALER_PATH):

    raise FileNotFoundError(
        "\nconcrete_scaler.pkl was not found.\n"
        f"Expected location:\n{SCALER_PATH}"
    )


if not os.path.exists(
    os.path.join(PUBLIC_DIR, "index.html")
):

    raise FileNotFoundError(
        "\nindex.html was not found.\n"
        f"Expected location:\n"
        f"{os.path.join(PUBLIC_DIR, 'index.html')}"
    )


print("\nModel file found!")
print("Scaler file found!")
print("Frontend file found!")


# ==========================================================
# LOAD MODEL
# ==========================================================

model = load_model(MODEL_PATH)

print("Neural network loaded successfully!")


# ==========================================================
# LOAD SCALER
# ==========================================================

with open(SCALER_PATH, "rb") as f:

    scaler = pickle.load(f)


print("Scaler loaded successfully!")


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
# PREDICTION ROUTE
# ==========================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # --------------------------------------------------
        # CHECK REQUEST DATA
        # --------------------------------------------------

        data = request.get_json(silent=True)

        if data is None:

            return jsonify({
                "success": False,
                "error": "No JSON data was received."
            }), 400


        # --------------------------------------------------
        # GET INPUT VALUES
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
        # CREATE DATAFRAME
        # --------------------------------------------------

        input_data = pd.DataFrame(
            [[
                cement,
                slag,
                flyash,
                water,
                superplasticizer,
                coarse,
                fine,
                age
            ]],
            columns=[
                "Cement (component 1)(kg in a m^3 mixture)",

                "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",

                "Fly Ash (component 3)(kg in a m^3 mixture)",

                "Water (component 4)(kg in a m^3 mixture)",

                "Superplasticizer (component 5)(kg in a m^3 mixture)",

                "Coarse Aggregate (component 6)(kg in a m^3 mixture)",

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


        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        prediction = model.predict(
            input_scaled,
            verbose=0
        )

        strength = float(
            prediction[0][0]
        )


        # --------------------------------------------------
        # CONCRETE GRADE
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
        # ENGINEERING RECOMMENDATION
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
    # HANDLE MISSING INPUT
    # ======================================================

    except KeyError as e:

        return jsonify({

            "success": False,

            "error": (
                f"Missing input field: {str(e)}"
            )

        }), 400


    # ======================================================
    # HANDLE INVALID INPUT
    # ======================================================

    except ValueError as e:

        return jsonify({

            "success": False,

            "error": (
                f"Invalid input value: {str(e)}"
            )

        }), 400


    # ======================================================
    # HANDLE OTHER ERRORS
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
# RUN LOCALLY
# ==========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )