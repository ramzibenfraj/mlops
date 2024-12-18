from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = {
            'Gender': request.form['Gender'],
            'Age': int(request.form['Age']),
            'HasDrivingLicense': int(request.form['HasDrivingLicense']),
            'RegionID': float(request.form['RegionID']),
            'Switch': int(request.form['Switch']),
            'PastAccident': request.form['PastAccident'],
            'AnnualPremium': float(request.form['AnnualPremium'])
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)[0]

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
