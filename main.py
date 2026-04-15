from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index_1.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from the form
    input_values = [
        float(request.form['mean_radius']),
        float(request.form['mean_texture']),
        float(request.form['mean_perimeter']),
        float(request.form['mean_area']),
        float(request.form['area_error']),
        float(request.form['worst_texture']),
        float(request.form['worst_perimeter']),
        float(request.form['worst_area'])
    ]
    input_array = np.array(input_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    # Return the result to the user
    if prediction == 1:
        result = "The tumor is **Benign** (Non-cancerous)."
    else:
        result = "The tumor is **Malignant** (Cancerous)."

    return render_template('index_1.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
