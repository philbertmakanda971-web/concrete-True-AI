import os
import pickle
import pandas as pd

# ==========================================================
# LIMIT TENSORFLOW RESOURCES
# ==========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, request, jsonify, send_from_directory

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
# GLOBAL MODEL VARIABLES
# ==========================================================

model = None
scaler = None


# ==========================================================
# LOAD SCALER
# ==========================================================

def load_scaler():

    global scaler

    if scaler is not None:
        return scaler

    print("=" * 60)
    print("Loading scaler...")
    print("=" * 60)

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"concrete_scaler.pkl was not found: {SCALER_PATH}"
        )

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    print("Scaler loaded successfully!")

    if hasattr(scaler, "feature_names_in_"):

        print("Scaler feature names:")

        for feature in scaler.feature_names_in_:
            print("-", feature)

    return scaler


# ==========================================================
# LOAD MODEL ONLY WHEN NEEDED
# ==========================================================

def load_tensorflow_model():

    global model

    if model is not None:
        return model

    print("=" * 60)
    print("Loading TensorFlow model...")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"concrete_model.keras was not found: {MODEL_PATH}"
        )

    # Import TensorFlow only when prediction is requested
    import tensorflow as tf

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Neural network loaded successfully!")

    return model


# ==========================================================
# STARTUP CHECK
# ==========================================================

print("=" * 60)
print("CONCRETE AI API")
print("=" * 60)

print("Project folder:")
print(BASE_DIR)

print("Model:")
print(MODEL_PATH)

print("Scaler:")
print(SCALER_PATH)

print("Frontend:")
print(PUBLIC_DIR)

# Check files WITHOUT loading TensorFlow
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"concrete_model.keras was not found: {MODEL_PATH}"
    )

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"concrete_scaler.pkl was not found: {SCALER_PATH}"
    )

if not os.path.exists(
    os.path.join(PUBLIC_DIR, "index.html")
):
    raise FileNotFoundError(
        f"index.html was not found: "
        f"{os.path.join(PUBLIC_DIR, 'index.html')}"
    )

print("Model file found!")
print("Scaler file found!")
print("Frontend file found!")
print("TensorFlow model will load when prediction is requested.")


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
# HEALTH CHECK
# ==========================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "service": "Concrete AI API"
    }), 200


# ==========================================================
# PREDICTION API
# ==========================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        print("=" * 60)
        print("PREDICTION REQUEST RECEIVED")
        print("=" * 60)

        # --------------------------------------------------
        # GET JSON
        # --------------------------------------------------

        data = request.get_json(silent=True)

        if data is None:

            return jsonify({
                "success": False,
                "error": "No JSON data was received."
            }), 400

        # --------------------------------------------------
        # READ VALUES
        # --------------------------------------------------

        cement = float(data["cement"])
        slag = float(data["slag"])
        flyash = float(data["flyash"])
        water = float(data["water"])
        superplasticizer = float(
            data["superplasticizer"]
        )
        coarse = float(data["coarse"])
        fine = float(data["fine"])
        age = float(data["age"])

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
        # LOAD SCALER
        # --------------------------------------------------

        current_scaler = load_scaler()

        # --------------------------------------------------
        # CREATE DATAFRAME
        # --------------------------------------------------

        if hasattr(
            current_scaler,
            "feature_names_in_"
        ):

            expected_columns = list(
                current_scaler.feature_names_in_
            )

            if len(expected_columns) != len(values):

                return jsonify({
                    "success": False,
                    "error":
                    "Scaler feature count does not "
                    "match the model input."
                }), 500

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
        # SCALE
        # --------------------------------------------------

        input_scaled = current_scaler.transform(
            input_data
        )

        print("Input scaled successfully!")

        # --------------------------------------------------
        # LOAD TENSORFLOW MODEL
        # --------------------------------------------------

        current_model = load_tensorflow_model()

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        prediction = current_model.predict(
            input_scaled,
            verbose=0
        )

        strength = float(
            prediction[0][0]
        )

        print("Predicted strength:", strength)

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