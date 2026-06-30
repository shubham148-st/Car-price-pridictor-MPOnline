from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print("Warning: model.pkl not found. Please run train_model.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not found. Please train the model first.", 500
        
    try:
        # Get parameters from the form
        year = int(request.form['year'])
        mileage = int(request.form['mileage'])
        engine_size = float(request.form['engine_size'])
        horsepower = int(request.form['horsepower'])
        
        # Create input array
        input_data = np.array([[year, mileage, engine_size, horsepower]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Format the price
        predicted_price = f"${prediction:,.2f}"
        
        # Return to form with prediction
        return render_template('index.html', 
                             predicted_price=predicted_price,
                             year=year,
                             mileage=mileage,
                             engine_size=engine_size,
                             horsepower=horsepower)
                             
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
