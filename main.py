import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import threading

# Function to fetch stock data
def get_stock_data(ticker):
    """Fetches historical stock data for a given ticker symbol."""
    try:
        data = yf.download(ticker, start="2010-01-01", end="2023-01-01")
        return data
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while fetching data: {e}")
        return None

# Function to preprocess data
def preprocess_data(data):
    """Prepares the data by creating the target variable and selecting relevant features."""
    # Flatten the MultiIndex columns
    data.columns = [' '.join(col).strip() for col in data.columns.values]

    # Identify the columns for 'Close' and 'Volume'
    close_col = [col for col in data.columns if 'Close' in col][0]
    volume_col = [col for col in data.columns if 'Volume' in col][0]

    # Ensure the required columns exist
    if close_col not in data.columns or volume_col not in data.columns:
        raise KeyError("The required columns ('Close' and 'Volume') are missing from the data.")

    # Create the target variable
    data['Target'] = data[close_col].shift(-1)

    # Drop rows with missing values in the 'Target' column
    data = data.dropna(subset=['Target'])

    # Select the relevant columns
    data = data[[close_col, volume_col, 'Target']]
    data.columns = ['Close', 'Volume', 'Target']  # Standardize column names

    return data

# Function to train the model and predict stock prices
def train_model(data):
    """Trains the RandomForestRegressor model and makes predictions."""
    # Features and target variable
    X = data[['Close', 'Volume']]  # Use standardized column names from preprocess_data
    y = data['Target']  # Target

    # Split the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features (important for many models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test_scaled)

    # Calculate the Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)

    return model, mae

# Function to predict the next day's stock price
def predict_price(model, current_price, current_volume):
    """Uses the trained model to predict the next day's stock price."""
    prediction = model.predict([[current_price, current_volume]])
    return prediction[0]

# GUI for the application
def create_gui():
    """Creates and runs the GUI application."""
    
    def on_predict_button_click():
        """Handles the prediction process when the 'Predict' button is clicked."""
        ticker = ticker_entry.get().strip().upper()
        
        if not ticker:
            messagebox.showwarning("Input Error", "Please enter a stock ticker symbol.")
            return
        
        # Show loading message while processing
        loading_label.pack_forget()  # Hide previous loading message
        loading_label.pack(pady=10)  # Show loading message
        
        # Fetch and process data in a separate thread to keep UI responsive
        def fetch_and_predict():
            data = get_stock_data(ticker)
            if data is None:
                return  # Exit if the data couldn't be fetched
            
            prepared_data = preprocess_data(data)

            # Train the model and get MAE
            model, mae = train_model(prepared_data)

            try:
                current_price = float(current_price_entry.get())
                current_volume = int(current_volume_entry.get())
            except ValueError:
                messagebox.showwarning("Input Error", "Please enter valid numerical values for the current price and volume.")
                loading_label.pack_forget()
                return

            # Predict the next day's stock price
            predicted_price = predict_price(model, current_price, current_volume)

            # Hide the loading message
            loading_label.pack_forget()

            # Show the prediction and MAE
            result_label.config(text=f"Predicted Price: ${predicted_price:.2f}")
            mae_label.config(text=f"Model MAE: ${mae:.2f}")

        # Run the prediction function in a separate thread to avoid freezing the GUI
        threading.Thread(target=fetch_and_predict).start()

    # Create the main window
    window = tk.Tk()
    window.title("Stock Price Prediction")
    window.geometry("500x600")  # Increased window size
    window.config(bg="#f0f0f0")

    # Add a stylish title label
    title_label = tk.Label(window, text="Stock Price Predictor", font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#2c3e50")
    title_label.pack(pady=20)

    # Ticker entry
    ticker_label = tk.Label(window, text="Enter Stock Ticker (e.g., AAPL):", font=("Arial", 12), bg="#f0f0f0")
    ticker_label.pack(padx=20, pady=5)
    ticker_entry = tk.Entry(window, width=25, font=("Arial", 14), bd=2, relief="solid")
    ticker_entry.pack(padx=20, pady=5)

    # Current price entry
    current_price_label = tk.Label(window, text="Enter Current Closing Price:", font=("Arial", 12), bg="#f0f0f0")
    current_price_label.pack(padx=20, pady=5)
    current_price_entry = tk.Entry(window, width=25, font=("Arial", 14), bd=2, relief="solid")
    current_price_entry.pack(padx=20, pady=5)

    # Current volume entry
    current_volume_label = tk.Label(window, text="Enter Current Volume:", font=("Arial", 12), bg="#f0f0f0")
    current_volume_label.pack(padx=20, pady=5)
    current_volume_entry = tk.Entry(window, width=25, font=("Arial", 14), bd=2, relief="solid")
    current_volume_entry.pack(padx=20, pady=5)

    # Loading label
    loading_label = tk.Label(window, text="Loading... Please wait.", font=("Arial", 12), bg="#f0f0f0", fg="#f39c12")

    # Predict button
    predict_button = tk.Button(window, text="Predict Next Day's Price", font=("Arial", 14), bg="#3498db", fg="white", command=on_predict_button_click, relief="raised", bd=3)
    predict_button.pack(padx=20, pady=20)

    # Result labels
    result_label = tk.Label(window, text="Predicted Price: $0.00", font=("Arial", 14), bg="#f0f0f0", fg="#2ecc71")
    result_label.pack(padx=20, pady=5)

    mae_label = tk.Label(window, text="Model MAE: $0.00", font=("Arial", 14), bg="#f0f0f0", fg="#e74c3c")
    mae_label.pack(padx=20, pady=5)

    # Run the application
    window.mainloop()

# Run the GUI
if __name__ == "__main__":
    create_gui()