import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import time
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from data_collection import MarketDataCollector
from model import EnhancedStockPredictor

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 Advanced Market Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
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
    """Generates the HTML for the TradingView Advanced Real-Time Chart Widget."""
    market_type = market_type.lower()
    
    # Adapt the symbol for TradingView's format
    if market_type == 'crypto':
        tv_symbol = f"COINBASE:{symbol.replace('-','')}USD"
    elif market_type == 'forex':
        tv_symbol = f"FX:{symbol}"
    else:  # stocks
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
    """Create an advanced interactive trading chart with multiple indicators and signals"""
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        subplot_titles=('Price & Predictions', 'RSI', 'MACD', 'Volume'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )

    # Main price chart with candlesticks
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

    # Add Bollinger Bands if available
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

    # Add moving averages
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

    # Add predictions
    if predictions is not None and len(predictions) > 0:
        # Create future timestamps for predictions
        last_timestamp = data.index[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(minutes=15),
            periods=len(predictions),
            freq='15T'
        )
        
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

    # Add trading signals
    if signals is not None and signals.get('signal'):
        signal_data = signals
        color = '#00ff88' if signal_data['signal'] == 'BUY' else '#ff4444'
        symbol = 'triangle-up' if signal_data['signal'] == 'BUY' else 'triangle-down'
        
        # Entry signal
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
        
        # Target and Stop Loss lines
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

    # RSI Chart
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

    # MACD Chart
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

    # Volume Chart
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

    # Update layout
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

    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

    return fig

def get_popular_symbols(market_type):
    """Get list of popular symbols for each market type"""
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
    """Display accuracy with appropriate styling"""
    if accuracy >= 85:
        return f'<span class="accuracy-high">🎯 {accuracy:.1f}% Accuracy</span>'
    elif accuracy >= 70:
        return f'<span class="accuracy-medium">⚠️ {accuracy:.1f}% Accuracy</span>'
    else:
        return f'<span class="accuracy-low">🔴 {accuracy:.1f}% Accuracy</span>'

def display_signal_box(signal):
    """Display trading signal with enhanced styling"""
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
        

    # Initialize session state for symbol if not already set
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = "AAPL"

    st.markdown("# 🚀 Advanced Market Prediction System")
    st.markdown("### *Powered by Enhanced LSTM-GRU Hybrid Model*")
    
    # Sidebar configuration
    st.sidebar.header("🎯 Market Selection")
    market_type = st.sidebar.selectbox(
        "Select Market Type",
        ["STOCKS", "CRYPTO", "FOREX"],
        help="Choose the market you want to analyze"
    )
    
    # Popular symbols section
    st.sidebar.subheader("📈 Popular Symbols")
    popular_symbols = get_popular_symbols(market_type)
    selected_symbol = None
    
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

    # Manual symbol input
    symbol_input = st.sidebar.text_input(
        "🔍 Or Enter Custom Symbol",
        value=st.session_state.selected_symbol,  # Use session state value
        help="Enter any valid symbol for analysis"
    ).upper()

    # Update session state if manual input is provided
    if symbol_input != st.session_state.selected_symbol:
        st.session_state.selected_symbol = symbol_input
    
    # Advanced settings
    st.sidebar.subheader("⚙️ Advanced Settings")
    signal_confidence = st.sidebar.slider(
        "Signal Confidence Threshold",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=5.0,
        help="Minimum confidence level for trading signals"
    )
    
    prediction_horizon = st.sidebar.selectbox(
        "Prediction Horizon",
        [1, 2, 3, 4, 6, 8],
        index=3,
        help="Number of time periods to predict ahead"
    )
    
    training_epochs = st.sidebar.slider(
        "Training Epochs",
        min_value=50,
        max_value=200,
        value=100,
        step=25,
        help="Number of training epochs (more epochs = better accuracy but slower)"
    )
    
    # Use selected symbol or manual input
    #symbol = selected_symbol if selected_symbol else symbol_input
    symbol = st.session_state.selected_symbol

    # Test symbol validity
    try:
        data_collector = MarketDataCollector(symbol, market_type)
        test_data = data_collector.get_historical_data(period='1d', interval='15m')
        
        if test_data.empty:
            st.error(f"❌ No data available for {symbol}. Please check the symbol and try again.")
            st.stop()
        
        # Display market info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Market", market_type, help="Current market type")
        with col2:
            st.metric("🔤 Symbol", symbol, help="Selected trading symbol")
        with col3:
            current_price = test_data['Close'].iloc[-1]
            price_change = test_data['Close'].pct_change().iloc[-1] * 100
            st.metric(
                "💰 Current Price",
                f"${current_price:.4f}",
                f"{price_change:+.2f}%",
                help="Latest price and 24h change"
            )
        with col4:
            volume = test_data['Volume'].sum()
            st.metric(
                "📊 24h Volume",
                f"{volume:,.0f}",
                help="Total 24-hour trading volume"
            )
        
        # TradingView Chart
        st.markdown("---")
        st.subheader("📈 Live Market Chart")
        st.markdown("*Real-time data from TradingView*")
        
        chart_col1, chart_col2 = st.columns([3, 1])
        with chart_col1:
            components.html(
                create_tradingview_widget(symbol, market_type, height=500),
                height=500
            )
        with chart_col2:
            st.markdown("### 🎯 Quick Stats")
            if not test_data.empty:
                st.metric("📈 High 24h", f"${test_data['High'].max():.4f}")
                st.metric("📉 Low 24h", f"${test_data['Low'].min():.4f}")
                st.metric("📊 Avg Volume", f"{test_data['Volume'].mean():.0f}")
                
                # Technical indicators preview
                test_with_indicators = data_collector.add_technical_indicators(test_data.copy())
                if 'RSI' in test_with_indicators.columns:
                    rsi_value = test_with_indicators['RSI'].iloc[-1]
                    st.metric("📊 RSI", f"{rsi_value:.1f}")
        
        # Model Training Section
        st.markdown("---")
        st.subheader("🤖 AI Model Training & Prediction")
        st.markdown("*Advanced LSTM-GRU Hybrid Model with Ensemble Learning*")
        
        # Training button
        if st.button(
            "🚀 Train Model & Generate Predictions",
            key="train_button",
            use_container_width=True,
            help="Train the AI model and generate predictions"
        ):
            try:
                if (st.session_state.trained_model is not None and 
                    st.session_state.last_symbol == symbol):
                    model = st.session_state.trained_model
                    df = st.session_state.historical_data
                else:
                    # Initialize and train model as before
                    model = EnhancedStockPredictor(
                        sequence_length=120,
                        prediction_horizon=prediction_horizon
                    )
                    df = data_collector.get_historical_data(period='2y', interval='15m')
                # Initialize model
                # model = EnhancedStockPredictor(
                #     sequence_length=120,
                #     prediction_horizon=prediction_horizon
                # )
                
                # Data collection phase
                with st.spinner("📊 Fetching historical data for training..."):
                    df = data_collector.get_historical_data(period='5y', interval='15m')
                    
                    if df.empty or len(df) < model.sequence_length:
                        st.error(f"❌ Insufficient data for {symbol}. Need at least {model.sequence_length} data points.")
                        st.stop()
                    
                    df = data_collector.add_technical_indicators(df)
                    st.success(f"✅ Collected {len(df)} data points with {len(df.columns)} features")
                    
                    st.session_state.last_symbol = symbol
                    st.session_state.trained_model = model
                    st.session_state.historical_data = df
                # Model training phase
                training_progress = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Preparing data and fitting scalers...")
                training_progress.progress(20)
                
                X, y = model.prepare_data_and_fit_scaler(df)
                if X.size == 0:
                    st.error("❌ Failed to prepare data for training.")
                    st.stop()
                
                status_text.text("🧠 Training ensemble models...")
                training_progress.progress(40)
                
                ensemble_results, training_metrics = model.train_ensemble(
                    X, y, 
                    epochs=training_epochs,
                    batch_size=32
                )
                
                training_progress.progress(80)
                status_text.text("📊 Evaluating model performance...")
                
                if not ensemble_results or not training_metrics:
                    st.error("❌ Model training failed. Please try again.")
                    st.stop()
                
                training_progress.progress(100)
                status_text.text("✅ Training completed successfully!")
                
                # Display training metrics
                st.success("🎉 Model Training Completed!")
                
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    accuracy = training_metrics.get('accuracy', 0)
                    st.markdown(display_accuracy_badge(accuracy), unsafe_allow_html=True)
                with metric_cols[1]:
                    mae = training_metrics.get('mae', 0)
                    st.metric("📊 MAE", f"{mae:.6f}")
                with metric_cols[2]:
                    loss = training_metrics.get('loss', 0)
                    st.metric("🎯 Loss", f"{loss:.6f}")
                
                # Prediction phase
                st.markdown("### 🔮 Generating Predictions")
                
                with st.spinner("🔍 Fetching latest market data..."):
                    live_data = data_collector.get_live_data()
                    
                    if live_data.empty or len(live_data) < model.sequence_length:
                        st.error(f"❌ Insufficient recent data for {symbol} prediction.")
                        st.stop()
                    
                    live_data_processed = data_collector.add_technical_indicators(live_data.copy())
                    X_live = model.prepare_prediction_data(live_data_processed)
                    
                    if X_live.size == 0:
                        st.error("❌ Failed to prepare prediction data.")
                        st.stop()
                
                with st.spinner("🔮 Generating ensemble predictions..."):
                    predictions = model.predict_ensemble(X_live)
                    
                    if predictions is None or len(predictions) == 0:
                        st.error("❌ Failed to generate predictions.")
                        st.stop()
                
                # Generate trading signals
                current_price = live_data['Close'].iloc[-1]
                next_prediction = predictions[-1] if len(predictions) > 0 else current_price
                
                # Get technical indicators for signal generation
                technical_indicators = {
                    'RSI': live_data_processed['RSI'].iloc[-1],
                    'MACD': live_data_processed['MACD'].iloc[-1],
                    'BB_position': live_data_processed['BB_position'].iloc[-1],
                    'Volume_ratio': live_data_processed['Volume_ratio'].iloc[-1]
                }
                
                signal = model.generate_advanced_signals(
                    current_price,
                    next_prediction,
                    confidence=signal_confidence,
                    technical_indicators=technical_indicators
                )
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Trading Analysis Results")
                
                # Create two columns for signal and chart
                signal_col, chart_col = st.columns([1, 2])
                
                with signal_col:
                    st.markdown("### 🎯 Trading Signal")
                    st.markdown(display_signal_box(signal), unsafe_allow_html=True)
                    
                    # Additional analysis
                    st.markdown("### 📈 Technical Analysis")
                    rsi = technical_indicators['RSI']
                    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.write(f"**RSI:** {rsi:.1f} ({rsi_status})")
                    
                    macd = technical_indicators['MACD']
                    macd_status = "Bullish" if macd > 0 else "Bearish"
                    st.write(f"**MACD:** {macd:.4f} ({macd_status})")
                    
                    bb_pos = technical_indicators['BB_position']
                    bb_status = "Upper Band" if bb_pos > 0.8 else "Lower Band" if bb_pos < 0.2 else "Middle"
                    st.write(f"**BB Position:** {bb_pos:.2f} ({bb_status})")
                    
                    # Price prediction
                    price_change = ((next_prediction - current_price) / current_price) * 100
                    st.markdown(f"### 🔮 Next Hour Prediction")
                    st.write(f"**Current:** ${current_price:.4f}")
                    st.write(f"**Predicted:** ${next_prediction:.4f}")
                    st.write(f"**Change:** {price_change:+.2f}%")
                
                with chart_col:
                    # Create advanced chart
                    fig = create_advanced_trading_chart(
                        live_data_processed.tail(200),  # Last 200 data points
                        predictions,
                        signal,
                        technical_indicators
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                st.markdown("---")
                st.subheader("📊 Model Performance & Risk Analysis")
                
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                
                with perf_col1:
                    st.markdown("### 🎯 Model Accuracy")
                    st.markdown(display_accuracy_badge(accuracy), unsafe_allow_html=True)
                    st.write(f"**Training Loss:** {loss:.6f}")
                    st.write(f"**MAE:** {mae:.6f}")
                    
                    # Model confidence
                    confidence_score = signal.get('confidence', 0) if signal else 0
                    st.write(f"**Signal Confidence:** {confidence_score:.1f}%")
                
                with perf_col2:
                    st.markdown("### 📈 Market Volatility")
                    volatility = live_data['Close'].pct_change().std() * 100
                    st.write(f"**Price Volatility:** {volatility:.2f}%")
                    
                    # Volume analysis
                    volume_change = live_data['Volume'].pct_change().iloc[-1] * 100
                    st.write(f"**Volume Change:** {volume_change:+.1f}%")
                    
                    # Trend strength
                    trend_strength = abs(price_change)
                    trend_status = "Strong" if trend_strength > 2 else "Moderate" if trend_strength > 1 else "Weak"
                    st.write(f"**Trend Strength:** {trend_status}")
                
                with perf_col3:
                    st.markdown("### ⚠️ Risk Assessment")
                    
                    # Calculate risk metrics
                    if signal and signal.get('risk_reward'):
                        risk_reward = signal['risk_reward']
                        risk_level = "Low" if risk_reward > 2 else "Medium" if risk_reward > 1.5 else "High"
                        st.write(f"**Risk/Reward:** {risk_reward:.2f}")
                        st.write(f"**Risk Level:** {risk_level}")
                    
                    # Market conditions
                    market_condition = "Bullish" if price_change > 0 else "Bearish"
                    st.write(f"**Market Condition:** {market_condition}")
                    
                    # Recommendation
                    if signal and signal.get('confidence', 0) > signal_confidence:
                        recommendation = f"✅ Follow {signal['signal']} signal"
                    else:
                        recommendation = "⚠️ Wait for better setup"
                    st.write(f"**Recommendation:** {recommendation}")
                
                # Prediction table
                st.markdown("---")
                st.subheader("📊 Detailed Predictions")
                
                # Create prediction dataframe
                current_time = pd.Timestamp.now()
                prediction_times = pd.date_range(
                    start=current_time + pd.Timedelta(minutes=15),
                    periods=len(predictions),
                    freq='15T'
                )
                
                prediction_df = pd.DataFrame({
                    'Time': prediction_times,
                    'Predicted Price': predictions,
                    'Price Change %': [(p - current_price) / current_price * 100 for p in predictions],
                    'Confidence': [max(0, accuracy - i*2) for i in range(len(predictions))]
                })
                
                st.dataframe(
                    prediction_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Export options
                st.markdown("---")
                st.subheader("📥 Export & Save")
                
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    # Export predictions to CSV
                    csv = prediction_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Predictions CSV",
                        data=csv,
                        file_name=f"{symbol}_predictions_{current_time.strftime('%Y%m%d_%H%M')}.csv",
                        mime='text/csv'
                    )
                
                with export_col2:
                    # Export signal data
                    if signal:
                        signal_df = pd.DataFrame([signal])
                        signal_csv = signal_df.to_csv(index=False)
                        st.download_button(
                            label="🎯 Download Signal Data",
                            data=signal_csv,
                            file_name=f"{symbol}_signal_{current_time.strftime('%Y%m%d_%H%M')}.csv",
                            mime='text/csv'
                        )
                
                with export_col3:
                    # Export chart data
                    chart_data = live_data_processed.tail(50)
                    chart_csv = chart_data.to_csv()
                    st.download_button(
                        label="📈 Download Chart Data",
                        data=chart_csv,
                        file_name=f"{symbol}_chart_{current_time.strftime('%Y%m%d_%H%M')}.csv",
                        mime='text/csv'
                    )
                
                # Real-time updates
                st.markdown("---")
                st.subheader("🔄 Real-time Updates")
                
                auto_refresh = st.checkbox(
                    "Enable Auto-refresh (30 seconds)",
                    value=False,
                    help="Automatically refresh predictions every 30 seconds"
                )
                
                if auto_refresh:
                    time.sleep(30)
                    st.rerun()
                
                # Manual refresh button
                if st.button("🔄 Refresh Data & Predictions", use_container_width=True):
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during model training: {str(e)}")
                st.write("Please try again or contact support if the issue persists.")
                
    except Exception as e:
        st.error(f"❌ Error accessing symbol {symbol}: {str(e)}")
        st.write("Please check the symbol name and try again.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(45deg, #1e3c72, #2a5298); 
                border-radius: 15px; margin-top: 2rem;">
        <h3>🚀 Advanced Market Prediction System</h3>
        <p>Powered by Enhanced LSTM-GRU Hybrid Model with Ensemble Learning</p>
        <p><strong>Disclaimer:</strong> This system is for educational purposes only. 
           Trading involves risk and past performance does not guarantee future results.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()