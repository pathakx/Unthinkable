"""
backend.py
-----------------
Flask backend to serve LLM-powered recommendations
"""

import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

# Add scripts folder to Python path
BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
sys.path.append(str(SCRIPTS_DIR))

# Import main recommender
from scripts.recommend_master import recommend_for_user

# ================================
# Flask setup
# ================================
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ================================
# Routes
# ================================
@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    """
    Expects JSON:
    { "user_id": "C00023" }
    """
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        logging.info(f"Generating recommendations for user {user_id}")
        recs = recommend_for_user(user_id)

        # Sanitize output for frontend
        for rec in recs:
            rec.pop("_cached", None)
        return jsonify({"user_id": user_id, "recommendations": recs})

    except Exception as e:
        logging.exception("Error generating recommendations")
        return jsonify({"error": str(e)}), 500


# ================================
# Entry Point
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
