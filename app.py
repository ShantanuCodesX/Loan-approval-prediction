from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained Random Forest model
model = joblib.load("loan_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    Gender = int(request.form["Gender"])
    Married = int(request.form["Married"])
    Dependents = int(request.form["Dependents"])
    Education = int(request.form["Education"])
    Self_Employed = int(request.form["Self_Employed"])

    ApplicantIncome = float(request.form["ApplicantIncome"])
    CoapplicantIncome = float(request.form["CoapplicantIncome"])
    LoanAmount = float(request.form["LoanAmount"])
    Loan_Amount_Term = float(request.form["Loan_Amount_Term"])

    Credit_History = int(request.form["Credit_History"])

    Property_Area = request.form["Property_Area"]


    # One-Hot Encoding for Property Area
    if Property_Area == "Rural":

        Property_Area_Rural = 1
        Property_Area_Semiurban = 0
        Property_Area_Urban = 0

    elif Property_Area == "Semiurban":

        Property_Area_Rural = 0
        Property_Area_Semiurban = 1
        Property_Area_Urban = 0

    else:

        Property_Area_Rural = 0
        Property_Area_Semiurban = 0
        Property_Area_Urban = 1


    # Create DataFrame
    new_applicant = pd.DataFrame({

        "Gender": [Gender],

        "Married": [Married],

        "Dependents": [Dependents],

        "Education": [Education],

        "Self_Employed": [Self_Employed],

        "ApplicantIncome": [ApplicantIncome],

        "CoapplicantIncome": [CoapplicantIncome],

        "LoanAmount": [LoanAmount],

        "Loan_Amount_Term": [Loan_Amount_Term],

        "Credit_History": [Credit_History],

        "Property_Area_Rural": [Property_Area_Rural],

        "Property_Area_Semiurban": [Property_Area_Semiurban],

        "Property_Area_Urban": [Property_Area_Urban]
    })


    # Make prediction
    prediction = model.predict(new_applicant)

    # Get approval probability
    probability = model.predict_proba(new_applicant)[0][1]


    # Convert prediction into readable result
    if prediction[0] == 1:

        result = "Loan Approved"

    else:

        result = "Loan Rejected"


    # Send result back to HTML
    return render_template(
        "index.html",
        prediction=result,
        probability=f"{probability * 100:.2f}%"
    )


if __name__ == "__main__":
    app.run(debug=True)