import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import time
import warnings
warnings.filterwarnings('ignore')

from data_collection import MarketDataCollector
from model import EnhancedStockPredictor

st.set_page_config(
    page_title="🚀 Advanced Market Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0c1426, #1a2332);
        color: white;
    }
    .stButton>button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #45a049, #4CAF50);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    .signal-box {
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    .buy-signal {
        background: linear-gradient(45deg, #00C851, #007E33);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .sell-signal {
        background: linear-gradient(45deg, #ff4444, #CC0000);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .no-signal {
        background: linear-gradient(45deg, #ffbb33, #ff8800);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .accuracy-high {
        background: linear-gradient(45deg, #00C851, #007E33);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .accuracy-medium {
        background: linear-gradient(45deg, #ffbb33, #ff8800);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .accuracy-low {
        background: linear-gradient(45deg, #ff4444, #CC0000);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #1a2332, #2a3441);
    }
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    .stTextInput > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    .stSlider > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_tradingview_widget(symbol, market_type, height=500):
    market_type = market_type.lower()
    
    if market_type == 'crypto':
        tv_symbol = f"COINBASE:{symbol.replace('-','')}USD"
    elif market_type == 'forex':
        tv_symbol = f"FX:{symbol}"
    else:
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
              "container_id": "tradingview_chart_widget",
              "studies": [
                  "RSI@tv-basicstudies",
                  "MACD@tv-basicstudies",
                  "BB@tv-basicstudies"
              ],
              "show_popup_button": true,
              "popup_width": "1000",
              "popup_height": "650"
          }}
          );
          </script>
        </div>
        """
    return html_code

def create_advanced_trading_chart(data, predictions, signals=None, technical_indicators=None):
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        subplot_titles=('Price & Predictions', 'RSI', 'MACD', 'Volume'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )

    if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_upper'],
                line=dict(color='rgba(173, 204, 255, 0.8)', width=1),
                name='BB Upper',
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_lower'],
                line=dict(color='rgba(173, 204, 255, 0.8)', width=1),
                name='BB Lower',
                fill='tonexty',
                fillcolor='rgba(173, 204, 255, 0.1)',
                showlegend=False
            ),
            row=1, col=1
        )

    if 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                line=dict(color='orange', width=2),
                name='SMA 20'
            ),
            row=1, col=1
        )

    if 'EMA_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['EMA_50'],
                line=dict(color='purple', width=2),
                name='EMA 50'
            ),
            row=1, col=1
        )

    if predictions is not None and len(predictions) > 0:
        last_timestamp = data.index[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(minutes=15),
            periods=len(predictions),
            freq='15T'
        )
        
        # Predicted price line
        fig.add_trace(
            go.Scatter(
                x=future_timestamps,
                y=predictions,
                line=dict(color='yellow', width=3, dash='dash'),
                name='Predicted Price',
                mode='lines+markers',
                marker=dict(size=6, color='yellow')
            ),
            row=1, col=1
        )
        
        # Confidence band (±5%)
        upper_bound = predictions * 1.05
        lower_bound = predictions * 0.95
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([future_timestamps, future_timestamps[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 255, 0, 0.2)',
                line=dict(color='rgba(255, 255, 0, 0)'),
                name='Confidence Band (±5%)',
                showlegend=True
            ),
            row=1, col=1
        )

    if signals is not None and signals.get('signal'):
        signal_data = signals
        color = '#00ff88' if signal_data['signal'] == 'BUY' else '#ff4444'
        symbol = 'triangle-up' if signal_data['signal'] == 'BUY' else 'triangle-down'
        
        fig.add_trace(
            go.Scatter(
                x=[data.index[-1]],
                y=[signal_data['entry_price']],
                mode='markers',
                marker=dict(symbol=symbol, size=20, color=color),
                name=f"{signal_data['signal']} Signal"
            ),
            row=1, col=1
        )
        
        fig.add_hline(
            y=signal_data['target_price'],
            line_dash="dash",
            line_color="green",
            annotation_text="Target",
            row=1, col=1
        )
        fig.add_hline(
            y=signal_data['stop_loss'],
            line_dash="dash",
            line_color="red",
            annotation_text="Stop Loss",
            row=1, col=1
        )

    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                line=dict(color='#ff6b6b', width=2),
                name='RSI',
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="solid", line_color="gray", row=2, col=1)

    if 'MACD' in data.columns and 'MACD_signal' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                line=dict(color='#4ecdc4', width=2),
                name='MACD',
                showlegend=False
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD_signal'],
                line=dict(color='#ff9f43', width=2),
                name='MACD Signal',
                showlegend=False
            ),
            row=3, col=1
        )
        if 'MACD_diff' in data.columns:
            colors = ['green' if val >= 0 else 'red' for val in data['MACD_diff']]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['MACD_diff'],
                    marker_color=colors,
                    name='MACD Histogram',
                    showlegend=False
                ),
                row=3, col=1
            )

    colors = ['#00ff88' if close >= open else '#ff4444' 
              for open, close in zip(data['Open'], data['Close'])]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=colors,
            name='Volume',
            showlegend=False
        ),
        row=4, col=1
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=800,
        title_text="Advanced Market Analysis & Predictions",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.8)"
        ),
        margin=dict(t=60, l=0, r=0, b=0),
        xaxis=dict(rangeslider=dict(visible=False))
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

    return fig

def get_popular_symbols(market_type):
    popular_symbols = {
        "STOCKS": {
            "🔥 Trending": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "💻 Technology": ["NVDA", "META", "NFLX", "ADBE", "CRM"],
            "🏦 Finance": ["JPM", "BAC", "WFC", "GS", "V", "MA"],
            "🏥 Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "⚡ Energy": ["XOM", "CVX", "COP", "EOG", "SLB"],
            "🏭 Industrial": ["GE", "CAT", "BA", "MMM", "HON"]
        },
        "CRYPTO": {
            "🚀 Large Cap": ["BTC", "ETH", "BNB", "XRP", "ADA"],
            "⚡ DeFi": ["UNI", "AAVE", "COMP", "MKR", "SNX"],
            "🎮 Gaming": ["AXS", "SAND", "MANA", "ENJ", "GALA"],
            "🔗 Layer 1": ["SOL", "AVAX", "DOT", "ALGO", "ATOM"],
            "🌉 Layer 2": ["MATIC", "OP", "ARB", "LRC", "IMX"]
        },
        "FOREX": {
            "💱 Major Pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
            "🏦 Minor Pairs": ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY"],
            "📊 Exotic Pairs": ["USDTRY", "USDZAR", "USDMXN", "USDRUB"],
            "💰 Commodity": ["AUDUSD", "NZDUSD", "USDCAD", "USDNOK"]
        }
    }
    return popular_symbols.get(market_type, {})

def display_accuracy_badge(accuracy):
    if accuracy >= 80:
        return f'<span class="accuracy-high">🎯 {accuracy:.1f}% Accuracy</span>'
    elif accuracy >= 70:
        return f'<span class="accuracy-medium">⚠️ {accuracy:.1f}% Accuracy</span>'
    else:
        return f'<span class="accuracy-low">🔴 {accuracy:.1f}% Accuracy</span>'

def display_signal_box(signal):
    if not signal or not signal.get('signal'):
        return '<div class="no-signal">📊 No Strong Signal - Market Analysis Needed</div>'
    
    signal_type = signal['signal']
    if signal_type == 'BUY':
        signal_html = f'''
        <div class="buy-signal">
            🚀 <strong>BUY SIGNAL</strong><br>
            💰 Entry: ${signal['entry_price']:.4f}<br>
            🎯 Target: ${signal['target_price']:.4f}<br>
            🛑 Stop Loss: ${signal['stop_loss']:.4f}<br>
            📊 Confidence: {signal['confidence']:.1f}%<br>
            ⚖️ Risk/Reward: {signal['risk_reward']:.2f}
        </div>
        '''
    else:
        signal_html = f'''
        <div class="sell-signal">
            📉 <strong>SELL SIGNAL</strong><br>
            💰 Entry: ${signal['entry_price']:.4f}<br>
            🎯 Target: ${signal['target_price']:.4f}<br>
            🛑 Stop Loss: ${signal['stop_loss']:.4f}<br>
            📊 Confidence: {signal['confidence']:.1f}%<br>
            ⚖️ Risk/Reward: {signal['risk_reward']:.2f}
        </div>
        '''
    
    return signal_html

def main():
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'last_symbol' not in st.session_state:
        st.session_state.last_symbol = None
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = "AAPL"
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = None

    st.markdown("# 🚀 Advanced Market Prediction System")
    st.markdown("### *Powered by Enhanced LSTM-GRU Hybrid Model*")
    
    st.sidebar.header("🎯 Market Selection")
    market_type = st.sidebar.selectbox(
        "Select Market Type",
        ["STOCKS", "CRYPTO", "FOREX"],
        help="Choose the market you want to analyze"
    )
    
    st.sidebar.subheader("📈 Popular Symbols")
    popular_symbols = get_popular_symbols(market_type)
    
    for category, symbols in popular_symbols.items():
        with st.sidebar.expander(f"{category}"):
            cols = st.columns(2)
            for i, symbol in enumerate(symbols):
                with cols[i % 2]:
                    if st.button(
                        symbol,
                        key=f"btn_{symbol}",
                        use_container_width=True,
                        help=f"Select {symbol} for analysis"
                    ):
                        st.session_state.selected_symbol = symbol

    symbol_input = st.sidebar.text_input(
        "🔍 Or Enter Custom Symbol",
        value=st.session_state.selected_symbol,
        help="Enter any valid symbol for analysis"
    ).upper()

    if symbol_input != st.session_state.selected_symbol:
        st.session_state.selected_symbol = symbol_input
    
    symbol = st.session_state.selected_symbol

    st.sidebar.subheader("⚙️ Advanced Settings")
    signal_confidence = st.sidebar.slider(
        "Signal Confidence Threshold", 0.0, 100.0, 80.0, 5.0,
        help="Minimum confidence level for trading signals"
    )
    prediction_horizon = st.sidebar.selectbox(
        "Prediction Horizon", [1, 2, 3, 4, 6, 8], index=3,
        help="Number of time periods to predict ahead"
    )
    training_epochs = st.sidebar.slider(
        "Training Epochs", 10, 50, 20, 5,
        help="Number of training epochs (more epochs = better accuracy but slower)"
    )
    
    try:
        data_collector = MarketDataCollector(symbol, market_type)
        test_data = data_collector.get_historical_data(period='60d', interval='15m')
        
        if test_data.empty:
            st.error(f"❌ No data available for {symbol}. Please check the symbol and try again.")
            st.stop()
        
        col1, col2, col3, col4 = st.columns(4)
        current_price = test_data['Close'].iloc[-1]
        price_change = test_data['Close'].pct_change().iloc[-1] * 100
        volume = test_data['Volume'].sum()
        col1.metric("📊 Market", market_type)
        col2.metric("🔤 Symbol", symbol)
        col3.metric("💰 Current Price", f"${current_price:,.4f}", f"{price_change:+.2f}%")
        col4.metric("📊 24h Volume", f"{volume:,.0f}")
        
        st.markdown("---")
        st.subheader("📈 Live Market Chart")
        components.html(create_tradingview_widget(symbol, market_type, height=500), height=500)
        
        st.markdown("---")
        st.subheader("🤖 ML Model Training & Prediction")
        
        if st.button("🚀 Train Model & Generate Predictions", use_container_width=True):
            try:
                if (st.session_state.trained_model is not None and 
                    st.session_state.last_symbol == symbol and
                    st.session_state.trained_model.prediction_horizon == prediction_horizon):
                    st.info(f"✅ Using cached model for {symbol}. To retrain with new settings, change a parameter.")
                    model = st.session_state.trained_model
                    df = st.session_state.historical_data
                    training_metrics = st.session_state.training_metrics
                else:
                    model = EnhancedStockPredictor(
                        sequence_length=60,
                        prediction_horizon=prediction_horizon
                    )
                    
                    with st.spinner("📊 Fetching historical data for training..."):
                        df = data_collector.get_historical_data(period='60d', interval='15m')
                        if df.empty or len(df) < model.sequence_length:
                            st.error(f"❌ Insufficient data for {symbol}. Need at least {model.sequence_length} data points.")
                            st.stop()
                        df = data_collector.add_technical_indicators(df)
                        st.success(f"✅ Collected {len(df)} data points.")
                    
                    with st.spinner("🧠 Training model... This should take less than 2 minutes."):
                        X, y = model.prepare_data_and_fit_scaler(df)
                        if X.size == 0:
                            st.error("❌ Failed to prepare data for training.")
                            st.stop()
                        
                        ensemble_results, training_metrics = model.train_ensemble(
                            X, y, epochs=training_epochs, batch_size=64
                        )
                        
                        if not ensemble_results:
                            st.error("❌ Model training failed. Please check the console for errors.")
                            st.stop()
                    
                    st.success("🎉 Model Training Completed!")
                    
                    st.session_state.trained_model = model
                    st.session_state.historical_data = df
                    st.session_state.last_symbol = symbol
                    st.session_state.training_metrics = training_metrics

                metric_cols = st.columns(3)
                accuracy = training_metrics.get('accuracy', 0)
                mae = training_metrics.get('mae', 0)
                loss = training_metrics.get('loss', 0)
                metric_cols[0].markdown(display_accuracy_badge(accuracy), unsafe_allow_html=True)
                metric_cols[1].metric("📊 MAE", f"{mae:.6f}")
                metric_cols[2].metric("🎯 Loss", f"{loss:.6f}")

                with st.spinner("🔮 Generating predictions..."):
                    live_data = data_collector.get_live_data()
                    if live_data.empty or len(live_data) < model.sequence_length:
                        st.error(f"❌ Insufficient recent data for {symbol} prediction.")
                        st.stop()
                    
                    live_data_processed = data_collector.add_technical_indicators(live_data.copy())
                    X_live = model.prepare_prediction_data(live_data_processed)
                    if X_live.size == 0:
                        st.error("❌ Failed to prepare prediction data.")
                        st.stop()

                    predictions = model.predict_ensemble(X_live)
                    if predictions is None or len(predictions) == 0:
                        st.error("❌ Failed to generate predictions.")
                        st.stop()

                current_price = live_data['Close'].iloc[-1]
                next_prediction = predictions[-1]
                
                technical_indicators = {
                    'RSI': live_data_processed['RSI'].iloc[-1],
                    'MACD': live_data_processed['MACD'].iloc[-1],
                    'MACD_signal': live_data_processed['MACD_signal'].iloc[-1],
                    'BB_position': live_data_processed['BB_position'].iloc[-1],
                    'ADX': live_data_processed['ADX'].iloc[-1]
                }
                
                signal = model.generate_advanced_signals(
                    current_price, next_prediction, confidence=signal_confidence,
                    technical_indicators=technical_indicators
                )
                
                st.markdown("---")
                st.subheader("📊 Trading Analysis Results")
                
                signal_col, chart_col = st.columns([1, 2])
                
                with signal_col:
                    st.markdown("### 🎯 Trading Signal")
                    st.markdown(display_signal_box(signal), unsafe_allow_html=True)
                    
                with chart_col:
                    fig = create_advanced_trading_chart(
                        live_data_processed.tail(200), predictions, signal
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

    except Exception as e:
        st.error(f"❌ Error setting up the app: {str(e)}")
        st.exception(e)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; margin-top: 2rem;">
        <p><strong>Disclaimer:</strong> This is an educational tool. Not financial advice.
           Trading involves significant risk.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()