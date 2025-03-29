from flask import Flask, request, jsonify
from src.predict import predict_sentiment

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    tweet = data["tweet"]
    result = predict_sentiment(tweet)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
