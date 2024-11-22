from urllib.parse import urlparse, parse_qs
import json
import requests
import warnings
from PIL import Image
import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to load a model based on parameters


# Function to extract parameters from the URL

def extract_params_from_url(url):
    parsed_url = urlparse(url)
    return {k: v[0] for k, v in parse_qs(parsed_url.query).items()}

# Function to fetch data from the API, save to a JSON file, and retrieve towerPrice values for '2Y'


def fetch_and_extract_prices(api_url):
    # Extract parameters from the URL
    params = extract_params_from_url(api_url)

    # Set up headers (e.g., user-agent)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Make the GET request with headers and parameters
    response = requests.get(api_url.split(
        '?')[0], params=params, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON data
        data = response.json()

        # Initialize an array to store the towerPrice values for the '2Y' time frame
        tower_prices_2y = []

        # Access the 'graph' key and the '2Y' array within it for towerPrice
        if "graph" in data and "2Y" in data["graph"]:
            for entry in data["graph"]["2Y"]:
                tower_prices_2y.append(entry.get("towerPrice"))

        # Return the array of towerPrice values
        return tower_prices_2y
    else:
        return None

# Forecast function with external values parameter


def generate_forecast_model(values):
    logging.info(f"Original input values: {values}")

    # Convert to numeric and filter NaN values
    values = pd.to_numeric(values, errors='coerce')
    logging.info(f"Numeric values after conversion: {values}")

    if isinstance(values, (pd.Series, np.ndarray)):
        values = [v for v in values if pd.notna(v)]
    else:
        values = [values] if pd.notna(values) else []

    logging.info(f"Filtered values (non-NaN): {values}")

    # Ensure 'values' is not empty
    if not values:
        raise ValueError(
            "Input values contain no valid data after conversion.")

    # Data setup
    data = {
        'Date': [
            'Nov 2022', 'Dec 2022', 'Jan 2023', 'Feb 2023', 'Mar 2023', 'Apr 2023',
            'May 2023', 'Jun 2023', 'Jul 2023', 'Aug 2023', 'Sep 2023', 'Oct 2023',
            'Nov 2023', 'Dec 2023', 'Jan 2024', 'Feb 2024', 'Mar 2024', 'Apr 2024',
            'May 2024', 'Jun 2024', 'Jul 2024', 'Aug 2024', 'Sep 2024', 'Oct 2024'
        ],
        'Value': values
    }
    df = pd.DataFrame(data)

    # Convert 'Date' to datetime format and set as index
    df['Date'] = pd.to_datetime(df['Date'], format='%b %Y')
    df.set_index('Date', inplace=True)

    # Explicitly set the frequency of the time series to monthly
    df = df.asfreq('MS')

    # Check for any remaining NaN or non-numeric issues
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    if df['Value'].isna().any():
        raise ValueError(
            "Data contains NaN or non-numeric values after conversion.")

    # Define model configuration
    trend = 'add'
    seasonal = None
    seasonal_periods = 12

    # Fit the model with the specified configuration
    model = ExponentialSmoothing(
        df['Value'], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fit = model.fit()

    # Forecast the next 6 months
    forecast = fit.forecast(steps=6)

    # Create a DataFrame for the forecasted values
    forecast_index = pd.date_range(
        start=df.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')
    forecast_df = pd.DataFrame({'Value': forecast}, index=forecast_index)

    # Calculate differences using iloc
    last_value = df['Value'].iloc[-1]  # Last value in the original data
    diff_holt = [forecast.iloc[0] - last_value]
    for i in range(1, len(forecast)):
        diff_holt.append(forecast.iloc[i] -
                         forecast.iloc[i-1] + diff_holt[i-1])

    return forecast_df, diff_holt, df['Value'].tolist()


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

    # Start forecast from November 2024
    forecast_df = forecast_df[forecast_df.index >= '2024-12-01']

    # Resample original data to quarterly, forecast data to monthly
    original_quarterly = original_df.resample('QE').mean()

    # Or any date further in the future
    forecast_cutoff_date = pd.Timestamp('2025-02-01')

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


# Load the region-location mappings once at the start
with open('region_location_mapping.json', 'r') as file:
    region_location_mapping = json.load(file)


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

        if str(bedroom).lower() == 'studio':
            bedroom = 0

        # Convert price and area to numeric values
        try:
            area = float(area) if area else None
            price = float(price) if price else None
            price_sqft = price / area if area else None
        except ValueError:
            abort(400, description="Price and area must be valid numbers")

        # Define the mapping of property types to typeId
        property_type_to_typeId = {
            "Villa": 35,
            "Apartment": 1,
            "Townhouse": 22,
            "Duplex": 24
        }

        # Get the typeId based on the property type, with a fallback if the type is unknown
        typeId = property_type_to_typeId.get(property_type)
        if typeId is None:
            abort(400, description="Invalid property type")

         # Get the location ID based on region
        location = region_location_mapping.get(region)
        if location is None:
            abort(400, description="Invalid region")

        # Construct the URL with the dynamic id
        api_url = f"https://www.propertyfinder.ae/api/pwa/tower-insights/price-trends?id={location}&categoryId=1&bedrooms={bedroom}&propertyTypeId={typeId}&locale=en"

        # Fetch and process data
        tower_prices_2y = fetch_and_extract_prices(api_url)

        # Generate the forecast using the generate_forecast_model function
        forecast_df, forecast_diff, original_values = generate_forecast_model(
            tower_prices_2y)

        # Define the cutoff for Q3 2024
        cutoff_date = pd.Timestamp('2024-10-30')

        # Prepare the forecasted data for response
        forecast_index = pd.date_range(
            start=pd.Timestamp('2024-11-01'), periods=6, freq='MS')
        forecast_df['Date'] = forecast_index
        forecast_df['Difference'] = forecast_diff

        # Extract the original data for plotting, but limit it to Q3 2024
        original_dates = pd.date_range(start=cutoff_date - pd.DateOffset(months=len(original_values)),
                                       periods=len(original_values), freq='MS')
        original_df = pd.DataFrame(
            {'Date': original_dates, 'Value': original_values})

        last_value = original_values[-1]
        current_price = price_sqft - last_value if price_sqft is not None else None

        # Calculate pre_price and forecast_price, ensuring 0 values are handled
        if len(original_df) > 1:
            pre_price = np.where(original_df['Value'] == 0, 0,
                                 (current_price + np.array(original_df['Value'])) * area)
            pre_dates = original_dates
        else:
            pre_price = np.array([])
            pre_dates = np.array([])

        forecast_price = np.where(forecast_df['Value'] == 0, 0,
                                  (current_price + np.array(forecast_df['Value'])) * area)

        # Convert forecast dates and prices for JSON response
        forecast_dates_filtered = [date.strftime(
            '%Y-%m-%d %H:%M:%S') for date in forecast_df['Date']]
        forecast_price_filtered = forecast_price.tolist()
        filtered_original_dates = [date.strftime(
            '%Y-%m-%d %H:%M:%S') for date in original_df['Date']]
        filtered_pre_price = pre_price.tolist()

        # Generate plot images in base64
        fig_original, fig_story = plot_forecast(
            pre_dates, pre_price, forecast_df['Date'], forecast_price, bedroom, property_type)
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
