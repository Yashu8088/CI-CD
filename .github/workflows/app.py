from flask import Flask, request, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Wine Quality Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]

    prediction = model.predict([features])

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
