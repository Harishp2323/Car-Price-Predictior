from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model pipeline
with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all input data from the form
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = request.form['fuel_type']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = int(request.form['owner'])

        # Create a dataframe with exact column names used in training
        input_data = pd.DataFrame([{
            'Year': year,
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Fuel_Type': fuel_type,
            'Seller_Type': seller_type,
            'Transmission': transmission,
            'Owner': owner
        }])

        # Predict using pipeline
        prediction = pipeline.predict(input_data)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
