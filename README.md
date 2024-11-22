# Stock Price Prediction App

This application predicts the next day's stock price using historical data and a Random Forest model. It allows users to input stock ticker symbols, current closing prices, and volumes to predict the next day's price.

![Stock Price Predictor](stock-prediction/screenshot.png)

## Features
- **Fetch Historical Data**: The app fetches historical stock data using the Yahoo Finance API.
- **Model Training**: A Random Forest Regressor model is used to predict the next day's stock price based on the input data.
- **Prediction**: Predicts the next day's stock price using the current closing price and volume.
- **GUI**: User-friendly graphical interface built with Tkinter.

## Requirements
- Python 3.x
- Libraries:
  - `tkinter`
  - `yfinance`
  - `pandas`
  - `scikit-learn`
  - `threading`

You can install the required libraries with the following command:

```bash
pip install yfinance pandas scikit-learn
```
## Usage

1. **Enter Stock Ticker**: Enter the stock symbol (e.g., AAPL for Apple).
2. **Enter Current Closing Price**: Provide the most recent closing price.
3. **Enter Current Volume**: Input the most recent volume.
4. **Predict**: Click the "Predict Next Day's Price" button to get the prediction.

The application will show:

- The predicted next day's stock price.
- The model's mean absolute error (MAE).

## Running the Application

To run the application, install all the dependencies and simply execute the Python script:

```bash
python main.py
```
## License

This project is open-source and available under the MIT License.