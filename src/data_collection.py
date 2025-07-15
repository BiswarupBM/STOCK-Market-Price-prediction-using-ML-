import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarketDataCollector:
    def __init__(self, symbol, market_type='stocks'):
        self.symbol = symbol.upper()
        self.market_type = market_type.lower()
        
        # Format symbol based on market type
        if self.market_type == 'forex':
            self.formatted_symbol = f"{self.symbol}=X"
        elif self.market_type == 'crypto':
            self.formatted_symbol = f"{self.symbol}-USD"
        else:
            self.formatted_symbol = self.symbol
            
        self.ticker = yf.Ticker(self.formatted_symbol)
        
        # Enhanced feature columns for better prediction
        self.feature_columns = [
            'Close', 'Volume', 'High', 'Low', 'Open',
            'RSI', 'RSI_3', 'RSI_14', 'RSI_21',
            'MACD', 'MACD_signal', 'MACD_diff',
            'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200',
            'Volume_EMA', 'Volume_SMA', 'Volume_ratio',
            'MFI', 'ADX', 'CCI', 'Williams_R', 'ROC',
            'ATR', 'VWAP', 'Price_change', 'Price_volatility',
            'Stoch_k', 'Stoch_d', 'OBV', 'TRIX',
            'Upper_shadow', 'Lower_shadow', 'Body_size',
            'HL_ratio', 'OC_ratio', 'Price_momentum_1',
            'Price_momentum_3', 'Price_momentum_5', 'Volume_momentum'
        ]

    def get_historical_data(self, period='2y', interval='15m'):
        """
        Fetches historical data with enhanced error handling and validation.
        """
        try:
            # Try multiple periods to ensure enough data
            periods_to_try = [period, '2y', '1y', '6mo', 'max']
            
            for p in periods_to_try:
                try:
                    df = self.ticker.history(period=p, interval=interval, auto_adjust=True)
                    if len(df) >= 200:  # Minimum required for technical indicators
                        break
                except:
                    continue
            
            if df.empty:
                print(f"No data available for {self.symbol}")
                return pd.DataFrame()
            
            # Data validation and cleaning
            df = df[df['Close'] > 0]
            df = df[df['Volume'] >= 0]
            df = df.dropna()
            
            if len(df) < 200:
                print(f"Insufficient data for {self.symbol}. Got {len(df)} points, need at least 200.")
                return pd.DataFrame()
            
            print(f"Successfully fetched {len(df)} data points for {self.symbol}")
            return df
            
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return pd.DataFrame()

    def add_technical_indicators(self, df):
        """
        Adds comprehensive technical indicators for enhanced prediction accuracy.
        """
        try:
            if df.empty or len(df) < 50:
                return df
            
            # Basic price features
            df['Price_change'] = df['Close'].pct_change()
            df['Price_volatility'] = df['Close'].rolling(window=20).std()
            df['HL_ratio'] = (df['High'] - df['Low']) / df['Close']
            df['OC_ratio'] = (df['Open'] - df['Close']) / df['Close']
            
            # Candlestick patterns
            df['Upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['Lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            df['Body_size'] = abs(df['Close'] - df['Open'])
            
            # Multiple RSI timeframes
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            df['RSI_3'] = ta.momentum.rsi(df['Close'], window=3)
            df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
            df['RSI_21'] = ta.momentum.rsi(df['Close'], window=21)
            
            # MACD indicators
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']
            df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            
            # Multiple Moving Averages
            for window in [5, 10, 20, 50, 200]:
                df[f'SMA_{window}'] = ta.trend.sma_indicator(df['Close'], window=window)
                df[f'EMA_{window}'] = ta.trend.ema_indicator(df['Close'], window=window)
            
            # Volume indicators
            df['Volume_EMA'] = ta.trend.ema_indicator(df['Volume'], window=20)
            df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
            df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
            df['Volume_momentum'] = df['Volume'].pct_change()
            
            # Advanced indicators
            df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
            df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
            df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            df['ROC'] = ta.momentum.roc(df['Close'])
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['TRIX'] = ta.trend.trix(df['Close'])
            
            # Stochastic oscillator
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_k'] = stoch.stoch()
            df['Stoch_d'] = stoch.stoch_signal()
            
            # VWAP
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # Price momentum features
            df['Price_momentum_1'] = df['Close'].pct_change(periods=1)
            df['Price_momentum_3'] = df['Close'].pct_change(periods=3)
            df['Price_momentum_5'] = df['Close'].pct_change(periods=5)
            
            # Handle NaN values
            df = df.ffill().bfill()
            
            # Remove any remaining NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"Error adding technical indicators: {e}")
            return df

    def get_live_data(self):
        """
        Get the most recent data for real-time predictions.
        """
        try:
            # Get recent data for live prediction
            df = self.get_historical_data(period='5d', interval='15m')
            if not df.empty:
                df = self.add_technical_indicators(df)
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error getting live data: {e}")
            return pd.DataFrame()

    def validate_data_quality(self, df):
        """
        Validates data quality and removes anomalies.
        """
        if df.empty:
            return df
            
        # Remove extreme outliers (more than 5 standard deviations)
        for col in ['Close', 'Volume', 'High', 'Low', 'Open']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df = df[abs(df[col] - mean) <= 5 * std]
        
        # Ensure logical price relationships
        df = df[df['High'] >= df['Low']]
        df = df[df['High'] >= df['Close']]
        df = df[df['High'] >= df['Open']]
        df = df[df['Low'] <= df['Close']]
        df = df[df['Low'] <= df['Open']]
        
        return df