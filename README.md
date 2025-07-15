# Stock Market Price Prediction using Machine Learning

This project implements a machine learning model to predict stock market prices using historical data and technical indicators.

## Prerequisites

- Python 3.8 or higher
- Git

## Installation Guide

1. Clone the repository
```bash
git clone https://github.com/BiswarupBM/STOCK-Market-Price-prediction-using-ML-.git
cd STOCK-Market-Price-prediction-using-ML-
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/MacOS
source venv/bin/activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run src/app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the web interface to:
   - Select a stock symbol
   - View historical data and technical indicators
   - Get price predictions

## Project Structure

- `src/app.py`: Main Streamlit application
- `src/data_collection.py`: Functions for fetching and processing stock data
- `src/model.py`: Machine learning model implementation
- `requirements.txt`: List of Python dependencies

## Required Python Packages

- numpy (>= 1.21.0)
- pandas (>= 1.3.0)
- yfinance (>= 0.2.3)
- scikit-learn (>= 1.0.0)
- tensorflow (>= 2.13.0)
- streamlit (>= 1.24.0)
- matplotlib (>= 3.7.1)
- plotly (>= 5.15.0)
- ta (>= 0.10.2)
- pandas-ta (>= 0.3.14b)
- requests (>= 2.31.0)

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are correctly installed
2. Check if your Python version is compatible
3. Make sure you have an active internet connection for stock data fetching
4. Verify that the virtual environment is activated before running the application

## License

This project is open source and available under the MIT License.