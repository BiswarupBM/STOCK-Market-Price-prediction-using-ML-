import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import time

from data_collection import MarketDataCollector
from model import StockPredictor

# Configure Streamlit page
st.set_page_config(
    page_title="Market Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for a better look
st.markdown("""
    <style>
    .stApp {
        background_color: #0e1117;
        color: white;
    }
    .stButton>button {
        background_color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
    }
    .stMetric {
        background_color: #1e2127;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_tradingview_widget(symbol, market_type, height=500):
    """Generates the HTML for the TradingView Advanced Real-Time Chart Widget."""
    market_type = market_type.lower()
    # Adapt the yfinance symbol for TradingView's format
    if market_type == 'crypto':
        tv_symbol = f"COINBASE:{symbol.replace('-','')}USD"
    elif market_type == 'forex':
        tv_symbol = f"FX:{symbol}"
    else: # stocks
        tv_symbol = symbol

    html_code = f"""
        <div class="tradingview-widget-container" style="height: {height}px; width: 100%;">
          <div id="tradingview_chart_widget" style="height: calc(100% - 32px); width: 100%;"></div>
          <div class="tradingview-widget-copyright">
              <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
                  <span class="blue-text">Track all markets on TradingView</span>
              </a>
          </div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
              "autosize": true,
              "symbol": "{tv_symbol}",
              "interval": "15",
              "timezone": "Etc/UTC",
              "theme": "dark",
              "style": "1",
              "locale": "en",
              "enable_publishing": false,
              "allow_symbol_change": true,
              "container_id": "tradingview_chart_widget"
          }}
          );
          </script>
        </div>
        """
    return html_code

def create_trading_chart(data, predictions, signals=None):
    """Create an interactive trading chart with indicators and signals"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                       subplot_titles=('Price & Predictions', 'Volume'), row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)

    if predictions is not None and len(predictions) > 0:
        fig.add_trace(go.Scatter(x=data.index[-len(predictions):], y=predictions,
                                 line=dict(color='yellow', width=2), name='Predicted Price'), row=1, col=1)

    if signals is not None and not signals.empty:
        for _, signal in signals.iterrows():
            color = 'green' if signal['signal'] == 'BUY' else 'red'
            symbol = 'triangle-up' if signal['signal'] == 'BUY' else 'triangle-down'
            fig.add_trace(go.Scatter(x=[data.index[-1]], y=[signal['entry_price']], mode='markers',
                                     marker=dict(symbol=symbol, size=15, color=color), name=f"{signal['signal']} Signal"), row=1, col=1)
            fig.add_shape(type="line", x0=data.index[-1], y0=signal['target_price'], x1=data.index[-1] + pd.Timedelta(hours=24),
                          y1=signal['target_price'], line=dict(color="green", width=2, dash="dash"), row=1, col=1)
            fig.add_shape(type="line", x0=data.index[-1], y0=signal['stop_loss'], x1=data.index[-1] + pd.Timedelta(hours=24),
                          y1=signal['stop_loss'], line=dict(color="red", width=2, dash="dash"), row=1, col=1)

    colors = ['green' if close >= open else 'red' for open, close in zip(data['Open'], data['Close'])]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=600, title_text="Price Prediction & Trading Signals", showlegend=True,
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"),
                      margin=dict(t=30, l=0, r=0, b=0))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    return fig

def get_popular_symbols(market_type):
    """Get list of popular symbols for each market type"""
    popular_symbols = {
        "STOCKS": {"Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"], "Electric Vehicles": ["TSLA", "NIO", "RIVN"], "Finance": ["JPM", "BAC", "GS", "V", "MA"], "Retail": ["AMZN", "WMT", "COST", "TGT"], "Healthcare": ["JNJ", "PFE", "MRNA", "UNH"]},
        "CRYPTO": {"Large Cap": ["BTC", "ETH", "BNB"], "DeFi": ["SOL", "ADA", "DOT"], "Metaverse": ["SAND", "MANA", "AXS"], "Layer 2": ["MATIC", "OP", "ARB"]},
        "FOREX": {"Major Pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"], "Commodity Pairs": ["AUDUSD", "NZDUSD", "USDCAD"], "Cross Rates": ["EURGBP", "EURJPY", "GBPJPY"]}
    }
    return popular_symbols.get(market_type, {})

def main():
    st.title("📈 Market Prediction System")

    st.sidebar.header("Market Selection")
    market_type = st.sidebar.selectbox("Select Market Type", ["STOCKS", "CRYPTO", "FOREX"])
    
    st.sidebar.subheader("Popular Symbols")
    popular_symbols = get_popular_symbols(market_type)
    selected_symbol = None
    for category, symbols in popular_symbols.items():
        with st.sidebar.expander(f"📁 {category}"):
            for symbol_in_list in symbols:
                if st.button(symbol_in_list, key=f"btn_{symbol_in_list}", use_container_width=True):
                    selected_symbol = symbol_in_list

    symbol_input = st.sidebar.text_input("Or Enter Symbol", value=selected_symbol if selected_symbol else "AAPL").upper()
    signal_confidence = st.sidebar.slider("Signal Confidence Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    symbol = selected_symbol if selected_symbol else symbol_input

    try:
        data_collector = MarketDataCollector(symbol, market_type)
        test_data = data_collector.get_historical_data(period='1d', interval='15m')
        if test_data.empty:
            st.error(f"No data available for {symbol}. Please check the symbol and try again.")
            st.stop()
            
        col1, col2, col3 = st.columns(3)
        col1.metric("Market", market_type)
        col2.metric("Symbol", symbol)
        col3.metric("24h Volume", f"${test_data['Volume'].sum():,.2f}")
        
        st.divider()
        st.subheader("Live Market Chart (TradingView)")
        components.html(create_tradingview_widget(symbol, market_type), height=500)
        st.divider()

        st.subheader("Model Training and Predictions")
        if st.button("Train Model & Predict", key="train_button"):
            try:
                model = StockPredictor(sequence_length=60)
                with st.spinner("Fetching 3Y of historical data for training..."):
                    df = data_collector.get_historical_data(period='3y', interval='15m')
                    if df.empty or len(df) < model.sequence_length:
                        st.error(f"Not enough historical data to train for {symbol}. Need > {model.sequence_length} data points.")
                        st.stop()
                    df = data_collector.add_technical_indicators(df)

                with st.spinner("Training model... This may take a moment."):
                    X, y = model.prepare_data_and_fit_scaler(df)
                    if X.size == 0:
                        st.error("Failed to prepare data for training.")
                        st.stop()
                    _, avg_scores = model.train(X, y)
                    val_mae = avg_scores.get('mae', 0)
                    
                    # Calculate accuracy
                    predictions_for_accuracy = model.predict(X)
                    estimated_accuracy = model.calculate_accuracy(y, predictions_for_accuracy)
                    
                col_metric_1, col_metric_2 = st.columns(2)
                with col_metric_1:
                    st.metric("Model Validation MAE", f"{val_mae:.4f}")
                with col_metric_2:
                    st.metric("Estimated Accuracy", f"{estimated_accuracy:.2f}%")
                
                with st.spinner("Generating predictions..."):
                    live_data = data_collector.get_historical_data(period='3mo', interval='15m')

                    if live_data.empty or len(live_data) < model.sequence_length:
                        st.error(f"Could not fetch enough recent data for {symbol} to make a prediction.")
                        st.stop()
                    live_data_processed = data_collector.add_technical_indicators(live_data.copy())
                    X_live = model.prepare_prediction_data(live_data_processed)
                    
                    if X_live.size > 0:
                        predictions = model.predict(X_live)
                        if predictions is not None and predictions.size > 0:
                            current_price = live_data['Close'].iloc[-1]
                            predicted_price = predictions[-1]
                            
                            ti_data = {'RSI': live_data_processed['RSI'].iloc[-1], 'MACD': live_data_processed['MACD'].iloc[-1],
                                       'BB_upper': live_data_processed['BB_upper'].iloc[-1], 'BB_lower': live_data_processed['BB_lower'].iloc[-1]}
                            
                            signal = model.generate_signals(current_price, predicted_price, confidence=signal_confidence, technical_indicators=ti_data)
                            
                            sig_col, chart_col = st.columns([1, 3])
                            with sig_col:
                                st.markdown("#### Trading Signal")
                                if signal and signal['signal']:
                                    if signal['signal'] == 'BUY':
                                        st.success(f"**BUY at ${signal['entry_price']:.2f}**")
                                    else:
                                        st.error(f"**SELL at ${signal['entry_price']:.2f}**")
                                    st.write(f"**Target:** ${signal['target_price']:.2f}")
                                    st.write(f"**Stop Loss:** ${signal['stop_loss']:.2f}")
                                    st.write(f"**Confidence:** {signal['confidence']:.2f}%")
                                else:
                                    st.info("No strong signal generated.")
                            
                            with chart_col:
                                fig = create_trading_chart(live_data, predictions, pd.DataFrame([signal]) if signal and signal['signal'] else None)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Model failed to generate predictions.")
                    else:
                        st.warning("Not enough recent data to form a sequence for prediction.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    except Exception as e:
        st.error(f"An error occurred during data initialization: {e}")

if __name__ == "__main__":
    main()