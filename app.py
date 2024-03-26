from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load the trained model
model = joblib.load("models/Logistic_Regression.joblib")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")  # Load the vectorizer used for training

# Create Flask app
app = Flask(__name__)

# Render the home page with input form
@app.route("/")
def home():
    return render_template("index.html")

# Define a route for making predictions
# @app.route("/predict", methods=["POST"])
# Modify the predict route to accept both GET and POST requests
# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         data = request.json
#         # Perform prediction and return response
#         prediction = model.predict([data['tweet_text']])[0]
#         return jsonify({"prediction": prediction})
#     else:
#         # Handle GET request (if needed)
#         return "GET request received"

# new code 
@app.route("/predict", methods=["GET"])
def predict():
    # Get tweet text from form
    tweet_text = request.form.get("tweet_text")
    
    # Vectorize the tweet text using the loaded TF-IDF vectorizer
    tweet_vectorized = vectorizer.transform([tweet_text])
    
    # Make predictions using the loaded model
    prediction = model.predict(tweet_vectorized)[0]
    
    # Provide clear instructions and feedback on the classification result
    if prediction == 1:
        result = "This tweet indicates a disaster."
    else:
        result = "This tweet does not indicate a disaster."
    
    # Return prediction result
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app in debug mode