import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd
import numpy as np

from data_collection import StockDataCollector
from model import StockPredictor

def create_candlestick_chart(df, predictions=None, signals=None):
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Market Data'
    ))
    
    # Add technical indicators with error handling
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='blue', width=1)
        ))
    
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_20'],
            mode='lines',
            name='EMA 20',
            line=dict(color='orange', width=1)
        ))
    
    # Add Bollinger Bands if available
    if all(indicator in df.columns for indicator in ['BB_upper', 'BB_lower']):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty'
        ))
    
    # Add predictions if available
    if predictions is not None:
        fig.add_trace(go.Scatter(
            x=df.index[-len(predictions):],
            y=predictions,
            mode='lines',
            name='Predictions',
            line=dict(color='red', width=2)
        ))
    
    # Add trading signals if available
    if signals is not None:
        # Add buy signals
        buy_signals = signals[signals['signal'] == 'BUY']
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['entry_price'],
            mode='markers',
            name='Buy Signal',
            marker=dict(
                symbol='triangle-up',
                size=15,
                color='green',
            )
        ))
        
        # Add sell signals
        sell_signals = signals[signals['signal'] == 'SELL']
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['entry_price'],
            mode='markers',
            name='Sell Signal',
            marker=dict(
                symbol='triangle-down',
                size=15,
                color='red',
            )
        ))
    
    fig.update_layout(
        title='Stock Price Analysis with Technical Indicators',
        yaxis_title='Stock Price',
        xaxis_title='Date',
        template='plotly_dark',
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_technical_chart(df):
    # Determine which indicators are available
    has_rsi = 'RSI' in df.columns
    has_macd = all(col in df.columns for col in ['MACD', 'MACD_signal'])
    
    # Calculate number of subplots needed
    num_subplots = 1  # Volume is always shown
    if has_rsi:
        num_subplots += 1
    if has_macd:
        num_subplots += 1
    
    # Create subplots
    fig = make_subplots(
        rows=num_subplots, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[1/num_subplots] * num_subplots
    )
    
    current_row = 1
    
    # Volume (always shown)
    colors = ['green' if row['Close'] - row['Open'] >= 0 else 'red' for idx, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.3),
        row=current_row, col=1, secondary_y=True
    )
    
    # RSI if available
    if has_rsi:
        current_row += 1
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI'),
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
    
    # MACD if available
    if has_macd:
        current_row += 1
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD'),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal'),
            row=current_row, col=1
        )
    
    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        title_text="Technical Analysis Chart"
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    
    return fig

def main():
    try:
        st.set_page_config(layout="wide")
        st.title('Advanced Stock Market Prediction System')
        
        # Sidebar for user input
        st.sidebar.header('User Input Parameters')
        
        # Stock symbol input with validation
        symbol = st.sidebar.text_input('Stock Symbol', 'AAPL').upper()
        
        # Timeframe selection
        timeframe = st.sidebar.selectbox(
            'Select Timeframe',
            ['15m', '30m', '1h', '4h', '1d'],
            index=0
        )
        
        # Add warning for 15-minute timeframe
        if timeframe == '15m':
            st.sidebar.warning('15-minute data might be limited for some stocks. If you encounter issues, try a longer timeframe.')
            
        # Validate stock symbol
        if not symbol.strip():
            st.error('Please enter a valid stock symbol.')
            return
        elif not symbol.isalnum():
            st.error('Stock symbol should only contain letters and numbers.')
            return
            
        # Initialize data collector with error handling
        try:
            data_collector = StockDataCollector(symbol)
            df = data_collector.get_historical_data(interval=timeframe)
            
            if df.empty:
                st.error(f"No data available for {symbol}. Please check the symbol and try again.")
                return
                
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            st.info("Please check the stock symbol and try again. If the problem persists, try a different timeframe.")
            return
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return
    
    # Training parameters
    st.sidebar.subheader('Training Parameters')
    epochs = st.sidebar.slider('Epochs', 50, 200, 100)
    sequence_length = st.sidebar.slider('Sequence Length', 10, 100, 60)
    
    # Initialize data collector
    data_collector = StockDataCollector(symbol)
    
    try:
        # Get historical data
        df = data_collector.get_historical_data(interval=timeframe)
        
        # Create predictor
        predictor = StockPredictor(sequence_length=sequence_length)
        
        # Prepare data
        X, y = predictor.prepare_data(df)
        
        # Training section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button('Train Model', key='train_button'):
                st.info('Training model... This may take a few minutes.')
                histories = predictor.train(X, y, epochs=epochs)
                st.success('Model training completed!')
                
                # Plot training history
                st.subheader('Training History')
                for i, history in enumerate(histories):
                    hist_df = pd.DataFrame(history.history)
                    st.line_chart(hist_df[['loss', 'val_loss']])
                    st.caption(f'Fold {i+1} Training History')
        
        # Make predictions
        if predictor.model is not None:
            predictions = predictor.predict(X[-sequence_length:])
            
            # Generate trading signals for recent data points
            signals_df = pd.DataFrame(index=df.index[-len(predictions):])
            signals_df['signal'] = 'HOLD'
            signals_df['entry_price'] = df['Close'][-len(predictions):]
            
            for i in range(len(predictions)):
                current_price = df['Close'].iloc[-(len(predictions)-i)]
                predicted_price = predictions[i]
                confidence = 1 - abs(predictions[i-1] - df['Close'].iloc[-(len(predictions)-i+1)]) / df['Close'].iloc[-(len(predictions)-i+1)] if i > 0 else 0.5
                
                signal = predictor.generate_signals(current_price, predicted_price, confidence)
                signals_df.iloc[i]['signal'] = signal['signal']
                signals_df.iloc[i]['entry_price'] = signal['entry_price']
                signals_df.iloc[i]['target_price'] = signal['target_price']
                signals_df.iloc[i]['stop_loss'] = signal['stop_loss']
                signals_df.iloc[i]['confidence'] = signal['confidence']
            
            # Display latest prediction
            with col2:
                st.subheader('Trading Signal Analysis')
                
                # Get latest technical indicators for signal generation
                latest_indicators = {
                    'RSI': df['RSI'].iloc[-1] if 'RSI' in df.columns else None,
                    'MACD': df['MACD'].iloc[-1] if 'MACD' in df.columns else None,
                    'MACD_signal': df['MACD_signal'].iloc[-1] if 'MACD_signal' in df.columns else None,
                    'SMA_20': df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else None,
                    'EMA_20': df['EMA_20'].iloc[-1] if 'EMA_20' in df.columns else None,
                    'ATR': df['ATR'].iloc[-1] if 'ATR' in df.columns else None
                }
                
                latest_signal = predictor.generate_signals(
                    df['Close'].iloc[-1],
                    predictions[-1],
                    confidence=max(min(1 - abs(predictions[-2] - df['Close'].iloc[-1]) / df['Close'].iloc[-1], 1), 0),
                    technical_indicators=latest_indicators
                )
                
                signal_color = {
                    'BUY': 'green',
                    'SELL': 'red',
                    'HOLD': 'gray',
                    'ERROR': 'yellow'
                }
                
                # Display Signal Box
                st.markdown(
                    f"""
                    <div style='padding: 20px; border-radius: 10px; background-color: rgba({
                        '0,255,0,0.1' if latest_signal['signal'] == 'BUY' else
                        '255,0,0,0.1' if latest_signal['signal'] == 'SELL' else
                        '128,128,128,0.1'
                    })'>
                        <h1 style='text-align: center; color: {signal_color[latest_signal["signal"]]};'>{latest_signal['signal']}</h1>
                        <p style='text-align: center; font-size: 1.2em;'>Confidence: {latest_signal['confidence']}%</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                st.markdown("### Price Analysis")
                
                # Display current price and prediction
                price_col1, price_col2 = st.columns(2)
                with price_col1:
                    st.metric(
                        "Current Price",
                        f"${df['Close'].iloc[-1]:.2f}",
                        f"{latest_signal['price_change_percent']:.1f}%"
                    )
                    st.metric(
                        "Predicted Price",
                        f"${latest_signal['predicted_price']:.2f}",
                        f"{((latest_signal['predicted_price'] - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100):.1f}%"
                    )
                
                # Display target and stop loss
                with price_col2:
                    st.metric(
                        "Target Price",
                        f"${latest_signal['target_price']:.2f}",
                        f"{((latest_signal['target_price'] - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100):.1f}%"
                    )
                    st.metric(
                        "Stop Loss",
                        f"${latest_signal['stop_loss']:.2f}",
                        f"{((latest_signal['stop_loss'] - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100):.1f}%"
                    )
                
                # Display risk metrics
                st.markdown("### Risk Analysis")
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                with risk_col1:
                    st.metric("Risk/Reward Ratio", f"{latest_signal['risk_reward_ratio']:.1f}")
                with risk_col2:
                    potential_profit = abs(latest_signal['target_price'] - latest_signal['entry_price'])
                    st.metric("Potential Profit", f"${potential_profit:.2f}")
                with risk_col3:
                    max_loss = abs(latest_signal['stop_loss'] - latest_signal['entry_price'])
                    st.metric("Maximum Loss", f"${max_loss:.2f}")
                
                # Display technical indicators
                if any(v is not None for v in latest_indicators.values()):
                    st.markdown("### Technical Indicators")
                    tech_col1, tech_col2 = st.columns(2)
                    
                    with tech_col1:
                        if latest_indicators['RSI'] is not None:
                            rsi_color = (
                                'red' if latest_indicators['RSI'] > 70 else
                                'green' if latest_indicators['RSI'] < 30 else
                                'white'
                            )
                            st.markdown(f"RSI: <span style='color:{rsi_color}'>{latest_indicators['RSI']:.1f}</span>", unsafe_allow_html=True)
                        
                        if latest_indicators['MACD'] is not None:
                            st.write(f"MACD: {latest_indicators['MACD']:.3f}")
                            
                    with tech_col2:
                        if latest_indicators['ATR'] is not None:
                            st.write(f"ATR: {latest_indicators['ATR']:.3f}")
                        
                        if latest_indicators['MACD_signal'] is not None:
                            st.write(f"MACD Signal: {latest_indicators['MACD_signal']:.3f}")
                            
                signal_color = {
                    'BUY': 'green',
                    'SELL': 'red',
                    'HOLD': 'gray'
                }
                
                st.markdown(f"<h1 style='text-align: center; color: {signal_color[latest_signal['signal']]};'>{latest_signal['signal']}</h1>", unsafe_allow_html=True)
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric('Current Price', f"${df['Close'].iloc[-1]:.2f}")
                    st.metric('Target Price', f"${latest_signal['target_price']:.2f}")
                with metrics_col2:
                    st.metric('Stop Loss', f"${latest_signal['stop_loss']:.2f}")
                    st.metric('Confidence', f"{latest_signal['confidence']:.1f}%")
            
            # Display charts
            st.subheader('Technical Analysis')
            tab1, tab2 = st.tabs(['Price Chart', 'Technical Indicators'])
            
            with tab1:
                fig = create_candlestick_chart(df, predictions, signals_df)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = create_technical_chart(df)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display performance metrics
            st.subheader('Model Performance Metrics')
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                accuracy = np.mean(np.sign(df['Returns'].iloc[-len(predictions):]) == 
                                np.sign(predictions - df['Close'].iloc[-len(predictions):]))
                st.metric('Direction Accuracy', f"{accuracy*100:.1f}%")
            
            with metrics_col2:
                rmse = np.sqrt(np.mean((predictions - df['Close'].iloc[-len(predictions):])**2))
                st.metric('RMSE', f"${rmse:.2f}")
            
            with metrics_col3:
                mape = np.mean(np.abs((df['Close'].iloc[-len(predictions):] - predictions) / 
                                    df['Close'].iloc[-len(predictions):])) * 100
                st.metric('MAPE', f"{mape:.1f}%")
        
        # Display raw data
        with st.expander('View Raw Data'):
            st.dataframe(df.tail(100))
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check the stock symbol and try again.")
        
    except Exception as e:
        st.error(f'Error: {str(e)}')

if __name__ == '__main__':
    main()
