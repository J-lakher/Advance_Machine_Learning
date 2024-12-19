from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the models and scaler
print("Loading models and scaler...")
regression_model = joblib.load('model_regression.pkl')
classification_model = joblib.load('model_classification.pkl')
scaler = joblib.load('scaler.pkl')
print("Models and scaler loaded successfully.")

# Category Labels for Classification
CATEGORY_LABELS = {
    0: "No Significant Sea Level Rise",
    1: "Significant Sea Level Rise"
}

@app.route('/')
def index():
    """Render the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the form submission and make predictions."""
    try:
        # Get inputs from the form
        year = float(request.form['year'])
        co2_emissions = float(request.form['co2_emissions'])
        wind_speed = float(request.form['wind_speed'])
        humidity = float(request.form['humidity'])

        # Combine inputs into a feature array
        input_features = np.array([[year, co2_emissions, wind_speed, humidity]])
        input_features_scaled = scaler.transform(input_features)

        # Make predictions
        temp_prediction = regression_model.predict(input_features_scaled)[0]
        sea_level_class = classification_model.predict(input_features_scaled)[0]

        # Get the class label
        class_label = CATEGORY_LABELS[sea_level_class]

        # Render the result
        return render_template('result.html',
                               temp_prediction=round(temp_prediction, 2),
                               class_label=class_label)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
