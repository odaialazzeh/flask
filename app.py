import warnings
from PIL import Image
import logging
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
import joblib
matplotlib.use('Agg')  # Use non-interactive backend

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to load a saved model


def plot_forecast(original_dates, original_values, forecast_dates, forecast_values, bedroom, property_type):
    def generate_plot(fig, ax, bar_width, logo_image_path, logo_position="bottom"):
        # Plot original data (Quarterly)
        ax.bar(original_quarterly.index, original_quarterly['Value'], color='gray', width=bar_width,
               label='Original Data (Bar)', alpha=0.7)
        ax.plot(original_quarterly.index, original_quarterly['Value'], marker='o', linestyle='--',
                color='#005a8c', label='Original Data (Line)')

        # Plot forecast data (Monthly)
        ax.bar(forecast_monthly.index, forecast_monthly['Value'], color='skyblue', width=20,
               label='Forecast (Bar)', alpha=0.7)
        ax.plot(forecast_monthly.index, forecast_monthly['Value'], marker='o', linestyle='--',
                color='#6b6b6b', label='Forecast (Line)')

        # Ensure the prices are visible on the plot
        for date, value in zip(original_quarterly.index, original_quarterly['Value']):
            if pd.isna(value):
                ax.annotate('Na', (date, 0), textcoords="offset points", xytext=(0, 10),
                            rotation=55, ha='center', color='#005a8c')
            elif pd.notna(value):
                ax.annotate(f'{value:,.0f}', (date, value), textcoords="offset points", xytext=(0, 10),
                            rotation=55, ha='center', color='#005a8c')

        for date, value in zip(forecast_monthly.index, forecast_monthly['Value']):
            if pd.isna(value):
                ax.annotate('Na', (date, 0), textcoords="offset points", xytext=(0, 10),
                            rotation=55, ha='center', color='#6b6b6b')
            elif pd.notna(value):
                ax.annotate(f'{value:,.0f}', (date, value), textcoords="offset points", xytext=(0, 10),
                            rotation=55, ha='center', color='#6b6b6b')

        # Customizing the plot
        ax.set_title("")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (AED)')
        ax.legend(
            loc='lower right',
            title=f"{'Studio' if bedroom == 0 else f'{bedroom} Bedroom'} {property_type}"
        )
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Remove the top and right spines, keep only the bottom and left spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # Set x-ticks for original data (quarterly)
        quarters = original_quarterly.index
        ax.set_xticks(quarters)
        ax.set_xticklabels([f'Q{(i.quarter)} {i.year}' for i in quarters])

        # Add the forecast x-ticks as months
        months = forecast_monthly.index
        ax.set_xticks(list(quarters) + list(months))
        ax.set_xticklabels(
            [f'Q{(i.quarter)} {i.year}' for i in quarters] +
            [f'{i.strftime("%b %Y")}' for i in months]
        )

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
    original_values = [np.nan if v == 0 else v for v in original_values]
    forecast_values = [np.nan if v == 0 else v for v in forecast_values]

    # Convert dates and values into DataFrame
    original_df = pd.DataFrame({'Date': pd.to_datetime(
        original_dates), 'Value': original_values}).set_index('Date')
    forecast_df = pd.DataFrame({'Date': pd.to_datetime(
        forecast_dates), 'Value': forecast_values}).set_index('Date')

    # Filter data to include only dates starting from 2023-04-01
    original_df = original_df[(
        original_df.index >= '2023-04-01') & (original_df.index <= '2024-09-30')]

    # Start forecast from Jan 2025
    forecast_df = forecast_df[forecast_df.index >= '2025-02-01']

    # Resample original data to quarterly, forecast data to monthly
    original_quarterly = original_df.resample('QE').mean()

    # Or any date further in the future
    forecast_cutoff_date = pd.Timestamp('2025-03-01')

    forecast_monthly = forecast_df[forecast_df.index <=
                                   forecast_cutoff_date].resample('ME').mean()

    # Define logo path using url_for
    logo_image_path = os.path.join('static', 'images', 'logo.png')

    # 1. Original Plot (14x6 inches)
    fig1, ax1 = plt.subplots(figsize=(14, 6), dpi=300)
    generate_plot(fig1, ax1, bar_width=27,
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
        data = request.get_json()
        if not data or not all(k in data for k in ['bedroom', 'propertyType', 'area', 'price', 'location']):
            abort(400, description="Missing required fields")

        bedroom = data.get('bedroom')
        property_type = data.get('propertyType')
        area = data.get('area')
        price = data.get('price')
        region = data.get('location')

        if str(bedroom).lower() == 'studio':
            bedroom = 0

        try:
            area = float(area) if area else None
            price = float(price) if price else None
            price_sqft = price / area if area else None
        except ValueError:
            abort(400, description="Price and area must be valid numbers")

        model_filename = f"model/{region}_{bedroom}_{property_type}.pkl"
        try:
            saved_data = joblib.load(model_filename)
            forecast_model = saved_data['model']
            original_values = saved_data['original_values']
        except FileNotFoundError:
            abort(
                404, description=f"Forecast model '{model_filename}' not found.")
        except Exception as e:
            abort(500, description=f"Error loading forecast model: {str(e)}")

        # Generate forecast values
        forecast_values = forecast_model.forecast(steps=6)

        forecast_index = pd.date_range(start=pd.Timestamp(
            '2024-11-01'), periods=len(forecast_values), freq='MS')
        forecast_df = pd.DataFrame(
            {'Date': forecast_index, 'Value': forecast_values})

        original_df = pd.DataFrame.from_dict(
            original_values, orient='index', columns=['Value'])
        original_df.index = pd.to_datetime(original_df.index)
        original_df = original_df.sort_index()

        last_value = original_df['Value'].iloc[-1]

        current_price = price_sqft - last_value if price_sqft is not None else None

        if current_price is not None and area:
            # Ensure area is a float for multiplication
            area = float(area)

            # Calculate pre_price and forecast_price
            pre_price = [(current_price + val) * area if val !=
                         0 else 0 for val in original_df['Value']]
            forecast_price = (
                current_price + forecast_df['Value']).mul(area).values.tolist()

        # Generate plots using plot_forecast
        fig_original, fig_story = plot_forecast(
            original_df.index, pre_price, forecast_df['Date'], forecast_price, bedroom, property_type)

        # Convert plots to Base64
        img_base64_original = save_plot_to_base64(fig_original)
        img_base64_story = save_plot_to_base64(fig_story)

        return jsonify({
            'forecast': forecast_df.reset_index().rename(columns={'index': 'Date'}).to_dict(orient='records'),
            'original_values': original_df.reset_index().rename(columns={'index': 'Date'}).to_dict(orient='records'),
            'current_price_diff': current_price,
            'pre_price': pre_price,
            'forecast_price': forecast_price,
            'image_standard': img_base64_original,
            'image_story': img_base64_story
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
                                                forecastPrices, bedroom, property_type)

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
