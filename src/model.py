import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow components
keras = tf.keras
layers = keras.layers
callbacks = keras.callbacks
optimizers = keras.optimizers

class EnhancedStockPredictor:
    def __init__(self, sequence_length=120, prediction_horizon=4):  # Increased for better patterns
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = RobustScaler()  # More robust to outliers
        self.target_scaler = MinMaxScaler()
        self._scaler_fitted = False
        self.model = None
        self.ensemble_models = []
        
        # Enhanced feature set
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

    def create_sequences(self, data, target_data=None):
        """
        Creates sequences for time series prediction with multiple targets.
        """
        X, y = [], []
        for i in range(self.sequence_length, len(data) - self.prediction_horizon + 1):
            X.append(data[i-self.sequence_length:i])
            if target_data is not None:
                # Predict next few time steps
                y.append(target_data[i:i+self.prediction_horizon])
            else:
                y.append(data[i, 0])  # Just next close price
        return np.array(X), np.array(y)

    def prepare_data_and_fit_scaler(self, df):
        """
        Prepares data with enhanced feature engineering and scaling.
        """
        try:
            if df.empty or len(df) < self.sequence_length + self.prediction_horizon:
                return np.array([]), np.array([])
            
            # Select and validate features
            available_features = [col for col in self.feature_columns if col in df.columns]
            if not available_features:
                print("No valid features found in the dataframe")
                return np.array([]), np.array([])
            
            # Use available features
            data = df[available_features].values
            
            # Fit scalers
            self.scaler.fit(data)
            self.target_scaler.fit(df[['Close']].values)
            self._scaler_fitted = True
            
            # Scale the data
            scaled_data = self.scaler.transform(data)
            scaled_target = self.target_scaler.transform(df[['Close']].values).flatten()
            
            # Create sequences
            X, y = self.create_sequences(scaled_data, scaled_target)
            
            print(f"Prepared {len(X)} sequences with {len(available_features)} features")
            return X, y
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            return np.array([]), np.array([])

    def prepare_prediction_data(self, df):
        """
        Prepares data for prediction using fitted scalers.
        """
        if not self._scaler_fitted:
            raise RuntimeError("Scalers not fitted. Call prepare_data_and_fit_scaler first.")
        
        try:
            if df.empty or len(df) < self.sequence_length:
                return np.array([])
            
            # Select available features
            available_features = [col for col in self.feature_columns if col in df.columns]
            data = df[available_features].values
            
            # Scale the data
            scaled_data = self.scaler.transform(data)
            
            # Create sequences for prediction
            X = []
            for i in range(self.sequence_length, len(scaled_data) + 1):
                X.append(scaled_data[i-self.sequence_length:i])
                
            return np.array(X)
            
        except Exception as e:
            print(f"Error preparing prediction data: {e}")
            return np.array([])

    def build_enhanced_model(self, input_shape):
        """
        Builds an enhanced hybrid LSTM-GRU model with attention mechanism.
        """
        try:
            inputs = layers.Input(shape=input_shape)
            
            # First LSTM branch
            lstm1 = layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
            lstm1 = layers.BatchNormalization()(lstm1)
            lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm1)
            lstm1 = layers.BatchNormalization()(lstm1)
            
            # Second GRU branch
            gru1 = layers.GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
            gru1 = layers.BatchNormalization()(gru1)
            gru1 = layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(gru1)
            gru1 = layers.BatchNormalization()(gru1)
            
            # Attention mechanism
            attention_lstm = layers.MultiHeadAttention(num_heads=8, key_dim=16)(lstm1, lstm1)
            attention_gru = layers.MultiHeadAttention(num_heads=8, key_dim=16)(gru1, gru1)
            
            # Combine branches
            combined = layers.Concatenate()([attention_lstm, attention_gru])
            combined = layers.Dropout(0.3)(combined)
            
            # Final processing layers
            x = layers.LSTM(64, return_sequences=False)(combined)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            # Dense layers with residual connections
            dense1 = layers.Dense(128, activation='relu')(x)
            dense1 = layers.BatchNormalization()(dense1)
            dense1 = layers.Dropout(0.2)(dense1)
            
            dense2 = layers.Dense(64, activation='relu')(dense1)
            dense2 = layers.BatchNormalization()(dense2)
            dense2 = layers.Dropout(0.2)(dense2)
            
            # Residual connection
            residual = layers.Dense(64, activation='relu')(x)
            combined_dense = layers.Add()([dense2, residual])
            
            # Final output layers
            final_dense = layers.Dense(32, activation='relu')(combined_dense)
            final_dense = layers.BatchNormalization()(final_dense)
            final_dense = layers.Dropout(0.1)(final_dense)
            
            # Multiple outputs for ensemble
            output1 = layers.Dense(1, activation='linear', name='main_output')(final_dense)
            output2 = layers.Dense(1, activation='linear', name='aux_output')(dense1)
            
            # Create model
            model = keras.Model(inputs=inputs, outputs=[output1, output2])
            
            # Custom optimizer with learning rate scheduling
            optimizer = optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                clipnorm=1.0
            )
            
            # Compile with multiple losses
            model.compile(
                optimizer=optimizer,
                loss={'main_output': 'huber', 'aux_output': 'mse'},
                loss_weights={'main_output': 0.8, 'aux_output': 0.2},
                metrics=['mae', 'mse']
            )
            
            return model
            
        except Exception as e:
            print(f"Error building model: {e}")
            return None

    def train_ensemble(self, X, y, epochs=100, batch_size=32):
        """
        Trains an ensemble of models for better accuracy.
        """
        try:
            if len(X) == 0 or len(y) == 0:
                print("No data to train on")
                return [], {}
            
            # Create multiple models with different configurations
            ensemble_results = []
            
            for i in range(3):  # Train 3 models
                print(f"\nTraining ensemble model {i+1}/3...")
                
                # Create model
                model = self.build_enhanced_model((X.shape[1], X.shape[2]))
                if model is None:
                    continue
                
                # Callbacks
                callbacks_list = [
                    callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=35,
                        restore_best_weights=True,
                        min_delta=0.0001
                    ),
                    callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=8,
                        min_lr=1e-6,
                        min_delta=0.0001
                    ),
                    callbacks.LearningRateScheduler(
                        lambda epoch: 0.001 * 0.98 ** epoch
                    )
                ]
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.2))
                fold_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Prepare targets for multiple outputs
                    y_train_dict = {'main_output': y_train, 'aux_output': y_train}
                    y_val_dict = {'main_output': y_val, 'aux_output': y_val}
                    
                    # Train model
                    history = model.fit(
                        X_train, y_train_dict,
                        validation_data=(X_val, y_val_dict),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks_list,
                        verbose=1 if fold == 0 else 0
                    )
                    
                    # Evaluate
                    val_loss = model.evaluate(X_val, y_val_dict, verbose=0)
                    fold_scores.append(val_loss)
                
                # Store model and results
                self.ensemble_models.append(model)
                ensemble_results.append({
                    'model': model,
                    'scores': np.mean(fold_scores, axis=0)
                })
            
            # Select best model as primary
            if ensemble_results:
                best_model = min(ensemble_results, key=lambda x: x['scores'][0])
                self.model = best_model['model']
                avg_scores = best_model['scores']
                
                # Calculate accuracy metrics
                accuracy = max(0, min(100, (1 - avg_scores[0]) * 100))
                
                return ensemble_results, {
                    'loss': avg_scores[0],
                    'mae': avg_scores[1] if len(avg_scores) > 1 else avg_scores[0],
                    'accuracy': accuracy
                }
            else:
                return [], {}
                
        except Exception as e:
            print(f"Error training ensemble: {e}")
            return [], {}

    def predict_ensemble(self, X):
        """
        Makes predictions using ensemble of models.
        """
        try:
            if not self.ensemble_models or X.size == 0:
                return None
            
            predictions = []
            for model in self.ensemble_models:
                try:
                    pred = model.predict(X, verbose=0)
                    if isinstance(pred, list):
                        pred = pred[0]  # Take main output
                    predictions.append(pred)
                except:
                    continue
            
            if not predictions:
                return None
            
            # Average predictions
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Inverse transform to get actual prices
            dummy_array = np.zeros((len(ensemble_pred), 1))
            dummy_array[:, 0] = ensemble_pred.flatten()
            actual_predictions = self.target_scaler.inverse_transform(dummy_array)
            
            return actual_predictions.flatten()
            
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return None

    def predict(self, X):
        """
        Makes predictions using the best model.
        """
        try:
            if self.model is None or X.size == 0:
                return None
            
            predictions = self.model.predict(X, verbose=0)
            if isinstance(predictions, list):
                predictions = predictions[0]  # Take main output
            
            # Inverse transform
            dummy_array = np.zeros((len(predictions), 1))
            dummy_array[:, 0] = predictions.flatten()
            actual_predictions = self.target_scaler.inverse_transform(dummy_array)
            
            return actual_predictions.flatten()
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    def calculate_advanced_accuracy(self, y_true, y_pred):
        """
        Calculates multiple accuracy metrics.
        """
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return 0
            
            # Directional accuracy
            y_true_direction = np.sign(np.diff(y_true))
            y_pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
            
            # Price accuracy (within 5% tolerance)
            price_accuracy = np.mean(np.abs(y_true - y_pred) / y_true < 0.05) * 100
            
            # R-squared
            r2 = r2_score(y_true, y_pred) * 100
            
            # Combined accuracy
            combined_accuracy = (directional_accuracy + price_accuracy + max(0, r2)) / 3
            
            return max(0, min(100, combined_accuracy))
            
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            return 0

    def generate_advanced_signals(self, current_price, predicted_price, confidence=0.0, technical_indicators=None):
        """
        Generates advanced trading signals with multiple confirmations.
        """
        try:
            if predicted_price is None:
                return None
            
            price_change = (predicted_price - current_price) / current_price
            signal = {
                'signal': None,
                'entry_price': current_price,
                'target_price': None,
                'stop_loss': None,
                'confidence': 0.0,
                'risk_reward': 0.0
            }
            
            # Technical analysis confirmation
            tech_score = 0
            if technical_indicators:
                rsi = technical_indicators.get('RSI', 50)
                macd = technical_indicators.get('MACD', 0)
                bb_position = technical_indicators.get('BB_position', 0.5)
                
                # Buy conditions
                if price_change > 0:
                    if rsi < 70: tech_score += 1
                    if macd > 0: tech_score += 1
                    if bb_position < 0.8: tech_score += 1
                
                # Sell conditions
                elif price_change < 0:
                    if rsi > 30: tech_score += 1
                    if macd < 0: tech_score += 1
                    if bb_position > 0.2: tech_score += 1
            
            # Calculate signal confidence
            price_confidence = abs(price_change) * 100
            tech_confidence = (tech_score / 3) * 100 if technical_indicators else 50
            signal_confidence = (price_confidence + tech_confidence) / 2
            
            # Generate signal if confidence is high enough
            if signal_confidence >= confidence and abs(price_change) > 0.01:  # At least 1% movement
                if price_change > 0:
                    signal['signal'] = 'BUY'
                    signal['target_price'] = current_price * (1 + abs(price_change) * 1.5)
                    signal['stop_loss'] = current_price * (1 - abs(price_change) * 0.5)
                else:
                    signal['signal'] = 'SELL'
                    signal['target_price'] = current_price * (1 + price_change * 1.5)
                    signal['stop_loss'] = current_price * (1 - price_change * 0.5)
                
                signal['confidence'] = signal_confidence
                
                # Calculate risk-reward ratio
                if signal['signal'] == 'BUY':
                    potential_profit = signal['target_price'] - current_price
                    potential_loss = current_price - signal['stop_loss']
                else:
                    potential_profit = current_price - signal['target_price']
                    potential_loss = signal['stop_loss'] - current_price
                
                signal['risk_reward'] = potential_profit / potential_loss if potential_loss > 0 else 0
            
            return signal
            
        except Exception as e:
            print(f"Error generating signals: {e}")
            return None