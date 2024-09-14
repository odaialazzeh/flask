import logging
import joblib
from flask_cors import CORS
from flask import Flask, jsonify, request, abort
import pandas as pd
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to load a model based on parameters


def load_model(bedroom, property_type, region):
    model_path = f'{region}_{bedroom}_{property_type}.pkl'
    saved_data = joblib.load(model_path)
    return saved_data


def plot_forecast(original_dates, original_values, forecast_dates, forecast_values, bedroom, property_type, region):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    bar_width = 5

    # Plot original data
    if len(original_dates) == len(original_values):
        ax.bar(original_dates, original_values, color='skyblue',
               width=bar_width, label='Original Data (Bar)', alpha=0.7)
        ax.plot(original_dates, original_values, marker='o',
                linestyle='-', color='#fc6100', label='Original Data (Line)')
    else:
        logging.warning(
            f"Length mismatch for original_dates ({len(original_dates)}) and original_values ({len(original_values)})")

    # Plot forecast data
    if len(forecast_dates) == len(forecast_values):
        ax.plot(forecast_dates, forecast_values, marker='o',
                linestyle='--', color='red', label='Forecast (Line)')
        ax.bar(forecast_dates, forecast_values, color='lightcoral',
               width=bar_width, label='Forecast (Bar)', alpha=0.7)
    else:
        logging.warning(
            f"Length mismatch for forecast_dates ({len(forecast_dates)}) and forecast_values ({len(forecast_values)})")

    # Set x-ticks to include all dates
    all_dates = pd.concat(
        [pd.Series(original_dates), pd.Series(forecast_dates)])
    ax.set_xticks(all_dates)
    ax.set_xticklabels(all_dates.dt.strftime('%b-%Y'), rotation=45, ha='right')

    # Add text annotations
    for date, value in zip(original_dates, original_values):
        ax.annotate(f'{value:,.0f}', (date, value), textcoords="offset points", xytext=(
            0, 10), rotation=55, ha='center', color='#fc6100')
    for date, value in zip(forecast_dates, forecast_values):
        ax.annotate(f'{value:,.0f}', (date, value), textcoords="offset points", xytext=(
            0, 10), rotation=55, ha='center', color='red')

    # Customizing the plot
    ax.set_title(
        f'Original Data and Forecast for {bedroom} Bedroom(s) {property_type} In {region} ', pad=25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (AED/Sqft)')
    ax.legend(loc='lower left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Return the figure object
    return fig


def save_plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    img_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        if not data or not all(k in data for k in ['bedroom', 'propertyType', 'area', 'price', 'region']):
            abort(400, description="Missing required fields")

        # Extract parameters from the received data
        bedroom = data.get('bedroom')
        property_type = data.get('propertyType')
        area = data.get('area')
        price = data.get('price')
        region = data.get('region')

        # Convert price and area to numeric values
        try:
            area = float(area) if area else None
            price = float(price) if price else None
            price_sqft = price / area if area else None
        except ValueError:
            abort(400, description="Price and area must be valid numbers")

        # Load the model and differences dynamically
        saved_data = load_model(bedroom, property_type, region)
        model = saved_data['model']
        forecast_diff = saved_data['forecast_diff']
        original_values = saved_data['original_values']

        if model is None:
            abort(400, description="No suitable model found for the provided parameters")

        # Generate the forecast using the loaded model
        forecast = model.forecast(steps=3)
        forecast_index = pd.date_range(start=pd.Timestamp.now(
        ) + pd.DateOffset(months=0), periods=3, freq='MS')
        forecast_df = pd.DataFrame(
            {'Date': forecast_index, 'Value': forecast, 'Difference': forecast_diff})

        # Extract the original data for plotting
        original_dates = pd.date_range(start=pd.Timestamp.now(
        ) - pd.DateOffset(months=len(original_values)), periods=len(original_values), freq='MS')
        original_df = pd.DataFrame(
            {'Date': original_dates, 'Value': original_values})

        # Calculate the current price (absolute difference between price_sqft and the last value)
        last_value = original_values[-1]
        current_price = abs(
            price_sqft - last_value) if price_sqft is not None else None

        # Calculate pre_price and forecast_price
        if len(original_df) > 1:
            pre_price = abs(
                current_price + np.array(original_df['Value'])) * area
            pre_dates = original_dates
        else:
            pre_price = np.array([])
            pre_dates = np.array([])

        forecast_price = abs(
            current_price + np.array(forecast_df['Value'])) * area

        # Plotting
        fig = plot_forecast(pre_dates, pre_price,
                            forecast_df['Date'], forecast_price, bedroom, property_type, region)
        img_base64 = save_plot_to_base64(fig)

        # Convert the DataFrame to a dictionary and return the forecast and image as JSON
        return jsonify({
            'forecast': forecast_df.to_dict(orient='records'),
            'image': img_base64,
            'forecast_dates': forecast_df['Date'].dt.strftime('%b-%Y').tolist(),
            'original_dates': original_df['Date'].dt.strftime('%b-%Y').tolist(),
            'original_values': original_values,
            'current_price_diff': current_price,
            'pre_price': pre_price.tolist() if pre_price.size > 0 else None,
            'forecast_price': forecast_price.tolist() if forecast_price.size > 0 else None
        })

    except Exception as e:
        logging.error(f"Error in /forecast: {str(e)}")
        return jsonify({"error": str(e)}), 400


@app.route('/update-image', methods=['POST'])
def update_image():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        if not data or not all(k in data for k in ['prePrices', 'forecastPrices', 'preDate', 'forecastDate']):
            abort(400, description="Missing required fields")

        # Extract parameters from the received data
        prePrices = data.get('prePrices')
        forecastPrices = data.get('forecastPrices')
        preDate = data.get('preDate')
        forecastDate = data.get('forecastDate')
        bedroom = data.get('bedroom')
        property_type = data.get('propertyType')
        region = data.get('region')

        # Validate input data
        if not isinstance(prePrices, list) or not isinstance(forecastPrices, list):
            abort(400, description="prePrices and forecastPrices must be lists")
        if not isinstance(preDate, list) or not isinstance(forecastDate, list):
            abort(400, description="preDate and forecastDate must be lists")

        prePrices = [float(p) for p in prePrices]
        forecastPrices = [float(p) for p in forecastPrices]

        # Update date format as necessary
        date_format = '%b-%Y'  # Replace with your date format
        preDate = pd.to_datetime(preDate, format=date_format).tolist()
        forecastDate = pd.to_datetime(
            forecastDate, format=date_format).tolist()

        if len(prePrices) != len(preDate) or len(forecastPrices) != len(forecastDate):
            abort(400, description="Mismatch between dates and prices length")

        # Plot updated forecast
        fig = plot_forecast(preDate, prePrices, forecastDate,
                            forecastPrices, bedroom, property_type, region)
        img_base64 = save_plot_to_base64(fig)

        # Return the updated image
        return jsonify({'image': img_base64})

    except Exception as e:
        logging.error(f"Error in /update-image: {str(e)}")
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
