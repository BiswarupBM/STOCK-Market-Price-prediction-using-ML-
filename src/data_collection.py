import yfinance as yf
import pandas as pd
import numpy as np
import ta

class MarketDataCollector:
    def __init__(self, symbol, market_type='stock'):
        self.symbol = symbol.upper()
        self.market_type = market_type.lower()
        
        if self.market_type == 'forex':
            self.formatted_symbol = f"{self.symbol}=X"
        elif self.market_type == 'crypto':
            self.formatted_symbol = f"{self.symbol}-USD"
        else:
            self.formatted_symbol = self.symbol
            
        self.ticker = yf.Ticker(self.formatted_symbol)

    def get_historical_data(self, period='5y', interval='1h'):
        """
        Fetches historical data from Yahoo Finance.
        
        Args:
            period (str): Time period to download ('1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max')
            interval (str): Data interval ('1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo')
        """
        try:
            # First try with the specified period
            df = self.ticker.history(period=period, interval=interval, auto_adjust=True)
            
            # If we don't get enough data, try with a longer period
            if len(df) < 60:
                print(f"Insufficient data points ({len(df)}) with period={period}. Attempting with longer period...")
                df = self.ticker.history(period='max', interval=interval, auto_adjust=True)
            
            if df.empty:
                print(f"No data available for {self.symbol} with symbol {self.formatted_symbol}")
                return pd.DataFrame()
            
            if len(df) < 60:
                print(f"Warning: Still insufficient data points ({len(df)}) for {self.symbol}. Minimum required: 60")
                return pd.DataFrame()
            
            # Filter invalid data
            df = df[df['Close'] > 0]
            df = df[df['Volume'] >= 0]
            
            print(f"Successfully fetched {len(df)} data points for {self.symbol}")
            return df
            
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return pd.DataFrame()

    def add_technical_indicators(self, df):
        """Adds a suite of technical indicators to the DataFrame."""
        try:
            if df.empty or len(df) < 20:
                return df
            
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd_diff()
            bb = ta.volatility.BollingerBands(df['Close'])
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['Volume_EMA'] = ta.trend.ema_indicator(df['Volume'].astype(float), window=20)
            
            # Use ffill() and bfill() to handle NaNs from indicator calculations
            df = df.ffill().bfill()
            return df
        except Exception as e:
            print(f"Error adding technical indicators: {e}")
            return df