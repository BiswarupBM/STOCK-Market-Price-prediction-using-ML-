import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

class StockPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def prepare_data(self, df):
        """
        Prepare data for LSTM/GRU model
        """
        # Select features for prediction
        features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 
                   'SMA_20', 'EMA_20', 'Volume_EMA']
        data = df[features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length, 0])  # Predicting Close price
            
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """
        Build hybrid LSTM-GRU model with attention mechanism
        """
        from tensorflow.keras.layers import Input, Concatenate, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping
        
        inputs = Input(shape=input_shape)
        
        # LSTM branch
        x1 = LSTM(128, return_sequences=True)(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = LSTM(64, return_sequences=True)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.3)(x1)
        
        # GRU branch
        x2 = GRU(128, return_sequences=True)(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        x2 = GRU(64, return_sequences=True)(x2)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        
        # Combine branches
        combined = Concatenate()([x1, x2])
        
        # Final GRU layer
        x = GRU(32)(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Use a learning rate scheduler
        initial_learning_rate = 0.001
        decay_steps = 1000
        decay_rate = 0.9
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps, decay_rate
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Less sensitive to outliers than MSE
            metrics=['mae', 'mse']
        )
        
        return model

    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the model with early stopping and k-fold cross validation
        """
        from sklearn.model_selection import TimeSeriesSplit
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        import tempfile
        
        if self.model is None:
            self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Create callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            mode='min'
        )
        
        # Use TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        histories = []
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\nTraining Fold {fold + 1}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create a temporary file for model checkpoints
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
                checkpoint = ModelCheckpoint(
                    tmp.name,
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=0
                )
                
                # Train the model
                history = self.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, checkpoint],
                    verbose=1
                )
                
                # Load the best model
                self.model.load_weights(tmp.name)
            
            # Evaluate the model
            scores = self.model.evaluate(X_val, y_val, verbose=0)
            fold_scores.append(scores)
            histories.append(history)
            
        # Calculate average scores across folds
        avg_scores = np.mean(fold_scores, axis=0)
        print("\nAverage scores across folds:")
        for metric, score in zip(self.model.metrics_names, avg_scores):
            print(f"{metric}: {score:.4f}")
        
        return histories

    def predict(self, X):
        """
        Make predictions with error handling and validation
        """
        try:
            if self.model is None:
                raise Exception("Model needs to be trained first!")
            
            if not isinstance(X, np.ndarray):
                raise ValueError("Input X must be a numpy array")
                
            if len(X.shape) != 3:
                raise ValueError(f"Expected 3D input array, got shape {X.shape}")
                
            if X.shape[1:] != self.model.input_shape[1:]:
                raise ValueError(f"Input shape {X.shape} does not match model input shape {self.model.input_shape}")
            
            predictions = self.model.predict(X, verbose=0)
            
            # Inverse transform to get actual prices
            dummy_array = np.zeros((len(predictions), X.shape[2]))
            dummy_array[:, 0] = predictions.flatten()
            predictions = self.scaler.inverse_transform(dummy_array)[:, 0]
            
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None

    def generate_signals(self, current_price, predicted_price, confidence, technical_indicators=None):
        """
        Generate trading signals based on predictions and technical indicators
        """
        try:
            # Calculate price change percentage
            price_change_percent = ((predicted_price - current_price) / current_price) * 100
            
            # Dynamic thresholds based on volatility
            volatility = technical_indicators.get('ATR', 0) / current_price * 100 if technical_indicators else 2.0
            
            BUY_THRESHOLD = max(0.5, volatility * 0.5)
            SELL_THRESHOLD = -max(0.5, volatility * 0.5)
            
            # Risk parameters
            RISK_REWARD_RATIO = 2.0  # Target should be 2x the risk
            MAX_RISK_PERCENT = 2.0   # Maximum risk per trade
            
            signal = "HOLD"
            entry_price = current_price
            target_price = current_price
            stop_loss = current_price
            signal_confidence = 0.0
            
            # Technical confirmation (if indicators available)
            technical_confirmation = 0
            if technical_indicators:
                # RSI confirmation
                if 'RSI' in technical_indicators:
                    rsi = technical_indicators['RSI']
                    if rsi < 30:  # Oversold
                        technical_confirmation += 1
                    elif rsi > 70:  # Overbought
                        technical_confirmation -= 1
                
                # MACD confirmation
                if 'MACD' in technical_indicators and 'MACD_signal' in technical_indicators:
                    if technical_indicators['MACD'] > technical_indicators['MACD_signal']:
                        technical_confirmation += 1
                    else:
                        technical_confirmation -= 1
                
                # Trend confirmation with moving averages
                if 'SMA_20' in technical_indicators and 'EMA_20' in technical_indicators:
                    if current_price > technical_indicators['SMA_20'] and current_price > technical_indicators['EMA_20']:
                        technical_confirmation += 1
                    elif current_price < technical_indicators['SMA_20'] and current_price < technical_indicators['EMA_20']:
                        technical_confirmation -= 1
            
            # Generate signals
            if price_change_percent > BUY_THRESHOLD and (technical_confirmation >= 0 or technical_indicators is None):
                signal = "BUY"
                entry_price = current_price
                
                # Calculate stop loss (maximum risk)
                max_risk_amount = current_price * (MAX_RISK_PERCENT / 100)
                stop_loss = current_price - max_risk_amount
                
                # Calculate target using risk/reward ratio
                target_price = current_price + (max_risk_amount * RISK_REWARD_RATIO)
                
                # Calculate signal confidence
                base_confidence = min(abs(price_change_percent) / (BUY_THRESHOLD * 2), 1.0)
                technical_factor = (technical_confirmation + 2) / 4 if technical_indicators else 0.5
                signal_confidence = (base_confidence * 0.7 + technical_factor * 0.3) * confidence
                
            elif price_change_percent < SELL_THRESHOLD and (technical_confirmation <= 0 or technical_indicators is None):
                signal = "SELL"
                entry_price = current_price
                
                # Calculate stop loss (maximum risk)
                max_risk_amount = current_price * (MAX_RISK_PERCENT / 100)
                stop_loss = current_price + max_risk_amount
                
                # Calculate target using risk/reward ratio
                target_price = current_price - (max_risk_amount * RISK_REWARD_RATIO)
                
                # Calculate signal confidence
                base_confidence = min(abs(price_change_percent) / (abs(SELL_THRESHOLD) * 2), 1.0)
                technical_factor = (abs(technical_confirmation) + 2) / 4 if technical_indicators else 0.5
                signal_confidence = (base_confidence * 0.7 + technical_factor * 0.3) * confidence
            
            return {
                "signal": signal,
                "entry_price": round(entry_price, 2),
                "target_price": round(target_price, 2),
                "stop_loss": round(stop_loss, 2),
                "confidence": round(signal_confidence * 100, 1),
                "predicted_price": round(predicted_price, 2),
                "price_change_percent": round(price_change_percent, 2),
                "risk_reward_ratio": RISK_REWARD_RATIO,
                "max_risk_percent": MAX_RISK_PERCENT
            }
            
        except Exception as e:
            print(f"Error generating signals: {e}")
            return {
                "signal": "ERROR",
                "entry_price": current_price,
                "target_price": current_price,
                "stop_loss": current_price,
                "confidence": 0.0,
                "predicted_price": predicted_price,
                "price_change_percent": 0.0,
                "risk_reward_ratio": 0.0,
                "max_risk_percent": 0.0
            }
        
        signal = "HOLD"
        entry_price = current_price
        target_price = current_price
        stop_loss = current_price
        
        # Risk/Reward ratio
        RISK_REWARD_RATIO = 2.0
        
        if price_change_percent > BUY_THRESHOLD and confidence > 0.7:
            signal = "BUY"
            entry_price = current_price
            # Set target using risk/reward ratio
            price_move = abs(predicted_price - current_price)
            target_price = current_price + (price_move * RISK_REWARD_RATIO)
            stop_loss = current_price - price_move
            
        elif price_change_percent < SELL_THRESHOLD and confidence > 0.7:
            signal = "SELL"
            entry_price = current_price
            # Set target using risk/reward ratio
            price_move = abs(predicted_price - current_price)
            target_price = current_price - (price_move * RISK_REWARD_RATIO)
            stop_loss = current_price + price_move
            
        return {
            "signal": signal,
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "confidence": confidence * 100,
            "predicted_price": predicted_price
        }
        
        if price_change_percent > BUY_THRESHOLD and confidence > 0.7:
            signal = "BUY"
            target_price = current_price * (1 + price_change_percent/100 * 2)  # 2x the predicted move
            stop_loss = current_price * (1 - price_change_percent/100 * 0.5)   # 0.5x the predicted move
            
        elif price_change_percent < SELL_THRESHOLD and confidence > 0.7:
            signal = "SELL"
            target_price = current_price * (1 + price_change_percent/100 * 2)
            stop_loss = current_price * (1 - price_change_percent/100 * 0.5)
            
        return {
            'signal': signal,
            'entry_price': entry_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'confidence': confidence * 100,  # Convert to percentage
            'predicted_change': price_change_percent
        }
