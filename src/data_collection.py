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
        
        if self.market_type == 'forex':
            self.formatted_symbol = f"{self.symbol}=X"
        elif self.market_type == 'crypto':
            self.formatted_symbol = f"{self.symbol}-USD"
        else:
            self.formatted_symbol = self.symbol
            
        self.ticker = yf.Ticker(self.formatted_symbol)
        
        self.feature_columns = [
            'Close', 'Volume', 'High', 'Low', 'Open',
            'RSI', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff',
            'BB_upper', 'BB_lower', 'BB_position',
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
            'Volume_ratio', 'MFI', 'ADX', 'Price_change', 'Price_volatility',
            'Stoch_k', 'Stoch_d', 'ATR'
        ]

    def get_historical_data(self, period='60d', interval='15m'):
        """
        Fetches historical data with enhanced error handling, respecting the 60-day limit for 15m data.
        Falls back to daily data if 15m fails.
        """
        try:
            # Try 15-minute data within 60 days
            df = self.ticker.history(period=period, interval=interval, auto_adjust=True)
            if df.empty or len(df) < 200:
                print(f"Insufficient 15m data for {self.symbol}. Trying daily data...")
                df = self.ticker.history(period='max', interval='1d', auto_adjust=True)
            
            if df.empty:
                print(f"No data available for {self.symbol}")
                return pd.DataFrame()
            
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
            
            df['Price_change'] = df['Close'].pct_change()
            df['Price_volatility'] = df['Close'].rolling(window=20).std()
            
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            df['RSI_14'] = df['RSI']
            
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
            
            bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
            
            df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            
            df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
            df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_k'] = stoch.stoch()
            df['Stoch_d'] = stoch.stoch_signal()
            
            df = df.ffill().bfill()
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
            df = self.get_historical_data(period='60d', interval='15m')
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
            
        for col in ['Close', 'Volume', 'High', 'Low', 'Open']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df = df[abs(df[col] - mean) <= 3 * std]
        
        df = df[df['High'] >= df['Low']]
        df = df[df['High'] >= df['Close']]
        df = df[df['High'] >= df['Open']]
        df = df[df['Low'] <= df['Close']]
        df = df[df['Low'] <= df['Open']]
        
        return df