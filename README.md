# Stock Market Price Prediction using Machine Learning

This project implements a machine learning model to predict stock market prices using historical data and technical indicators.


<img width="1920" height="871" alt="{81189AAF-34BF-408B-8A13-8EAEA3C7C351}" src="https://github.com/user-attachments/assets/a387ae94-631f-40b3-b8fd-b39549652d72" />

<img width="1920" height="870" alt="{263B3457-F85A-4131-9219-A5427CCA4012}" src="https://github.com/user-attachments/assets/75504467-d962-4464-80b2-bfa1aa70c25b" />

<img width="1919" height="872" alt="{1887DE0F-A9BA-489D-A00F-E90495C51ACE}" src="https://github.com/user-attachments/assets/35418b5e-d2e3-4c21-ba8d-eba7bbda2773" />

<img width="1920" height="1080" alt="{F393A7AB-DD6C-4B94-9CE7-EF93E2A8BF5C}" src="https://github.com/user-attachments/assets/ab10d9d8-52b9-4e6d-9646-b4a639c0461f" />

<img width="1920" height="1080" alt="{408B1BBA-6054-45CB-904E-C0925747A644}" src="https://github.com/user-attachments/assets/dfe9b017-a0e3-4e0a-a409-35c28e2ea38a" />


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
