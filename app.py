from PIL import Image
import logging
import joblib
from flask_cors import CORS
from flask import Flask, jsonify, request, abort
import pandas as pd
import numpy as np
import base64
import io
import os
import matplotlib.pyplot as plt
from PIL import Image  # Import PIL for image handling
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to load a model based on parameters


def load_model(bedroom, property_type, region, mainRegion):
    # Construct paths based on mainRegion and region
    mainRegion_model_path = f'model/{mainRegion}_model/{region}_{bedroom}_{property_type}.pkl' if mainRegion else None
    region_model_path = f'model/{region}_model/{region}_{bedroom}_{property_type}.pkl'

    # Check if the folder for mainRegion exists, otherwise fallback to region
    if mainRegion and os.path.exists(os.path.dirname(mainRegion_model_path)):
        model_path = mainRegion_model_path
    elif os.path.exists(os.path.dirname(region_model_path)):
        model_path = region_model_path
    else:
        raise FileNotFoundError(
            f"Neither model folder for {mainRegion} nor {region} exists."
        )

    # Load the model from the selected path
    saved_data = joblib.load(model_path)
    return saved_data


def plot_forecast(original_dates, original_values, forecast_dates, forecast_values, bedroom, property_type):
    def generate_plot(fig, ax, bar_width, logo_image_path, logo_position="bottom"):
        # Plot original data (Quarterly)
        ax.bar(original_quarterly.index, original_quarterly['Value'], color='gray', width=bar_width,
               label='Original Data (Bar)', alpha=0.7)
        ax.plot(original_quarterly.index, original_quarterly['Value'], marker='o', linestyle='--',
                color='#005a8c', label='Original Data (Line)')

        # Plot forecast data (Quarterly)
        ax.bar(forecast_quarterly.index, forecast_quarterly['Value'], color='skyblue', width=bar_width,
               label='Forecast (Bar)', alpha=0.7)
        ax.plot(forecast_quarterly.index, forecast_quarterly['Value'], marker='o', linestyle='--',
                color='#6b6b6b', label='Forecast (Line)')

        # Ensure the prices are visible on the plot
        for date, value in zip(original_quarterly.index, original_quarterly['Value']):
            if pd.notna(value):
                ax.annotate(f'{value:,.0f}', (date, value), textcoords="offset points", xytext=(0, 10),
                            rotation=55, ha='center', color='#005a8c')

        for date, value in zip(forecast_quarterly.index, forecast_quarterly['Value']):
            if pd.notna(value):
                ax.annotate(f'{value:,.0f}', (date, value), textcoords="offset points", xytext=(0, 10),
                            rotation=55, ha='center', color='#6b6b6b')

        # Customizing the plot
        ax.set_title("")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (AED)')
        ax.legend(loc='lower right',
                  title=f'{bedroom} Bedroom {property_type}')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Remove the top and right spines, keep only the bottom and left spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # Set x-ticks to show quarters (Q1, Q2, etc.), starting from 2023-04-01
        quarters = pd.date_range(
            start='2023-04-01', end=forecast_quarterly.index[-1], freq='QE')
        ax.set_xticks(quarters)
        ax.set_xticklabels([f'Q{(i.quarter)} {i.year}' for i in quarters])

        plt.xticks(rotation=45)  # Rotate x-axis labels for readability
        plt.tight_layout()

        # Determine the minimum value of the original data
        min_value_original = original_quarterly['Value'].min()

        # Add the logo as a watermark
        if logo_image_path:
            try:
                logo_image = Image.open(logo_image_path)
                logo_resized = logo_image.resize((2200, 200))

                if logo_position == "bottom":
                    # Center bottom for fig1
                    fig_center_x = (fig.bbox.xmin + fig.bbox.xmax) / \
                        2 - (logo_resized.width / 2)
                    logo_position_y = ax.transData.transform(
                        (0, min_value_original - (min_value_original * 0.15)))[1]
                else:
                    # Center top for fig2
                    fig_center_x = (fig.bbox.xmin + fig.bbox.xmax) / \
                        2 - (logo_resized.width / 8)
                    logo_position_y = fig.bbox.ymax

                # Set the logo as a watermark
                fig.figimage(logo_resized, xo=fig_center_x,
                             yo=logo_position_y, alpha=0.8, zorder=5)
            except Exception as e:
                logging.error(f"Error loading logo image: {str(e)}")

    # Convert 'Na' in the input data to np.nan
    original_values = [np.nan if v == 'Na' else v for v in original_values]
    forecast_values = [np.nan if v == 'Na' else v for v in forecast_values]

    # Convert dates and values into DataFrame
    original_df = pd.DataFrame({'Date': pd.to_datetime(
        original_dates), 'Value': original_values}).set_index('Date')
    forecast_df = pd.DataFrame({'Date': pd.to_datetime(
        forecast_dates), 'Value': forecast_values}).set_index('Date')

    # Filter data to include only dates starting from 2023-04-01
    original_df = original_df[original_df.index >= '2023-04-01']
    forecast_df = forecast_df[forecast_df.index >= '2023-04-01']

    # Resample to quarterly, taking the mean of each quarter
    original_quarterly = original_df.resample('QE').mean()
    forecast_quarterly = forecast_df.resample('QE').mean()

    # Define logo path using url_for
    logo_image_path = os.path.join('static', 'images', 'logo.png')

    # 1. Original Plot (14x6 inches)
    fig1, ax1 = plt.subplots(figsize=(14, 6), dpi=300)
    generate_plot(fig1, ax1, bar_width=30,
                  logo_image_path=logo_image_path, logo_position="bottom")

    # 2. Plot with 1080x1920 dimensions (9x16 inches at 120 dpi), logo at top
    fig2, ax2 = plt.subplots(figsize=(9, 16), dpi=120)
    generate_plot(fig2, ax2, bar_width=20,
                  logo_image_path=logo_image_path, logo_position="top")

    # Return both figures
    return fig1, fig2


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
        region = data.get('location')
        mainRegion = data.get('region')

        # Convert price and area to numeric values
        try:
            area = float(area) if area else None
            price = float(price) if price else None
            price_sqft = price / area if area else None
        except ValueError:
            abort(400, description="Price and area must be valid numbers")

        # Load the model and differences dynamically
        saved_data = load_model(bedroom, property_type, region, mainRegion)
        model = saved_data['model']
        forecast_diff = saved_data['forecast_diff']
        original_values = saved_data['original_values']

        if model is None:
            abort(400, description="No suitable model found for the provided parameters")

        # Define the cutoff for Q3 2024
        cutoff_date = pd.Timestamp('2024-09-30')

        # Generate the forecast using the loaded model
        forecast = model.forecast(steps=6)
        forecast_index = pd.date_range(start=cutoff_date, periods=6, freq='MS')
        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            'Value': forecast,
            'Difference': forecast_diff
        })

        # Extract the original data for plotting, but limit it to Q3 2024
        original_dates = pd.date_range(start=cutoff_date - pd.DateOffset(months=len(original_values)),
                                       periods=len(original_values), freq='MS')
        original_df = pd.DataFrame(
            {'Date': original_dates, 'Value': original_values})

        last_value = original_values[-1]
        current_price = price_sqft - last_value if price_sqft is not None else None

        # Calculate pre_price and forecast_price
        if len(original_df) > 1:
            pre_price = (current_price + np.array(original_df['Value'])) * area
            pre_dates = original_dates
        else:
            pre_price = np.array([])
            pre_dates = np.array([])

        forecast_price = (
            current_price + np.array(forecast_df['Value'])) * area

        # Resample original and forecast data to quarterly (optional if needed for other parts of the project)
        original_quarterly = original_df.set_index(
            'Date').resample('QE').mean()
        forecast_quarterly = forecast_df.set_index(
            'Date').resample('QE').mean()

        # Filter the data to include the forecasted dates
        forecast_dates_filtered = [date.strftime(
            '%Y-%m-%d %H:%M:%S') for date in forecast_df['Date']]

        # Filter forecast prices
        forecast_price_filtered = [price for price in forecast_price]

        # Convert original dates to the correct format
        filtered_original_dates = [date.strftime(
            '%Y-%m-%d %H:%M:%S') for date in original_df['Date']]

        # Filter pre_price values if needed
        filtered_pre_price = pre_price.tolist()

        # Save both figures to base64 with the logo as watermark (if provided)
        fig_original, fig_story = plot_forecast(
            pre_dates, pre_price, forecast_df['Date'], forecast_price,
            bedroom, property_type
        )

        img_base64_original = save_plot_to_base64(fig_original)
        img_base64_story = save_plot_to_base64(fig_story)

        # Return the forecast and images as JSON
        return jsonify({
            'forecast': forecast_df[['Date', 'Difference', 'Value']].to_dict(orient='records'),
            'image_standard': img_base64_original,
            'image_story': img_base64_story,
            'forecast_dates': forecast_dates_filtered,
            'original_dates': filtered_original_dates,
            'original_values': original_values,
            'current_price_diff': current_price,
            'pre_price': filtered_pre_price,
            'forecast_price': forecast_price_filtered
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
        region = data.get('location')
        email = data.get('email')

        # Validate input data
        if not isinstance(prePrices, list) or not isinstance(forecastPrices, list):
            abort(400, description="prePrices and forecastPrices must be lists")
        if not isinstance(preDate, list) or not isinstance(forecastDate, list):
            abort(400, description="preDate and forecastDate must be lists")

        prePrices = [float(p) for p in prePrices]
        forecastPrices = [float(p) for p in forecastPrices]

        # Let pandas infer the date formats
        preDate = pd.to_datetime(preDate).tolist()
        forecastDate = pd.to_datetime(forecastDate).tolist()

        if len(prePrices) != len(preDate) or len(forecastPrices) != len(forecastDate):
            abort(400, description="Mismatch between dates and prices length")

        # Plot updated forecast (generate both standard and story format images)
        fig_original, fig_story = plot_forecast(preDate, prePrices, forecastDate,
                                                forecastPrices, bedroom, property_type, region, email)

        # Save both standard and story images to base64
        img_base64_original = save_plot_to_base64(fig_original)
        img_base64_story = save_plot_to_base64(fig_story)

        # Return the updated images in the response
        return jsonify({
            'image_standard': img_base64_original,  # Standard format image
            'image_story': img_base64_story  # Story format image (1080x1920)
        })

    except Exception as e:
        logging.error(f"Error in /update-image: {str(e)}")
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
