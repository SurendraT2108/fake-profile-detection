from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    followers = int(request.form["followers"])
    following = int(request.form["following"])
    has_pic = int(request.form["has_pic"])
    has_bio = int(request.form["has_bio"])

    features = np.array([[followers, following, has_pic, has_bio]])
    prediction = model.predict(features)[0]

    result = "FAKE PROFILE" if prediction == 1 else "REAL PROFILE"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
