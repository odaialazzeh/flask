from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Function to load a model based on parameters
def load_model(bedroom, property_type):
    model_path = f'forecast_model_{bedroom}_{property_type}.pkl'
    saved_data = joblib.load(model_path)
    return saved_data

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        if data is None:
            raise ValueError("No JSON data received")

        # Print the received data for debugging
        print("Received data:", data)
        
        # Extract parameters from the received data
        bedroom = data.get('bedroom', None)
        property_type = data.get('propertyType', None)
        
        # Load the model and differences dynamically
        saved_data = load_model(bedroom, property_type)
        model = saved_data['model']
        forecast_diff = saved_data['forecast_diff']
        
        if model is None:
            raise ValueError("No suitable model found for the provided parameters")

        # Generate the forecast using the loaded model
        forecast = model.forecast(steps=3)  # Adjust based on your model's forecast method
        
        # Create a DataFrame for the forecasted values
        forecast_index = pd.date_range(start=pd.Timestamp.now() + pd.DateOffset(months=0), periods=3, freq='MS')
        forecast_df = pd.DataFrame({'Date': forecast_index, 'Value': forecast, 'Difference': forecast_diff})
        
        # Convert the DataFrame to a dictionary and return as JSON
        return jsonify(forecast_df.to_dict(orient='records'))
    except Exception as e:
        # Return a JSON error message with status code 400
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
