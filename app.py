from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/model.pkl')

@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('index'))  
        
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

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(405)
def method_not_allowed_error(error):
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
