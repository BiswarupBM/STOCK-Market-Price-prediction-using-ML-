import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

keras = tf.keras
layers = keras.layers
callbacks = keras.callbacks
optimizers = keras.optimizers

class EnhancedStockPredictor:
    def __init__(self, sequence_length=60, prediction_horizon=4):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = RobustScaler()
        self.target_scaler = MinMaxScaler()
        self._scaler_fitted = False
        self.model = None
        self.ensemble_models = []
        
        self.feature_columns = [
            'Close', 'Volume', 'High', 'Low', 'Open',
            'RSI', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff',
            'BB_upper', 'BB_lower', 'BB_position',
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
            'Volume_ratio', 'MFI', 'ADX', 'Price_change', 'Price_volatility',
            'Stoch_k', 'Stoch_d', 'ATR'
        ]

    def create_sequences(self, data, target_data=None):
        X, y = [], []
        for i in range(self.sequence_length, len(data) - self.prediction_horizon + 1):
            X.append(data[i-self.sequence_length:i])
            if target_data is not None:
                y.append(target_data[i:i+self.prediction_horizon])
            else:
                y.append(data[i, 0])
        return np.array(X), np.array(y)

    def prepare_data_and_fit_scaler(self, df):
        try:
            if df.empty or len(df) < self.sequence_length + self.prediction_horizon:
                return np.array([]), np.array([])
            
            available_features = [col for col in self.feature_columns if col in df.columns]
            if not available_features:
                print("No valid features found in the dataframe")
                return np.array([]), np.array([])
            
            data = df[available_features].values
            
            for col_idx in range(data.shape[1]):
                mean = np.mean(data[:, col_idx])
                std = np.std(data[:, col_idx])
                data = data[abs(data[:, col_idx] - mean) <= 3 * std]
            
            self.scaler.fit(data)
            self.target_scaler.fit(df[['Close']].values)
            self._scaler_fitted = True
            
            scaled_data = self.scaler.transform(data)
            scaled_target = self.target_scaler.transform(df[['Close']].values).flatten()
            
            X, y = self.create_sequences(scaled_data, scaled_target)
            
            if len(X) < 50:
                print("Insufficient sequences for training")
                return np.array([]), np.array([])
            
            print(f"Prepared {len(X)} sequences with {len(available_features)} features")
            return X, y
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            return np.array([]), np.array([])

    def prepare_prediction_data(self, df):
        if not self._scaler_fitted:
            raise RuntimeError("Scalers not fitted. Call prepare_data_and_fit_scaler first.")
        
        try:
            if df.empty or len(df) < self.sequence_length:
                return np.array([])
            
            available_features = [col for col in self.feature_columns if col in df.columns]
            data = df[available_features].values
            
            scaled_data = self.scaler.transform(data)
            
            X = []
            for i in range(self.sequence_length, len(scaled_data) + 1):
                X.append(scaled_data[i-self.sequence_length:i])
                
            return np.array(X)
            
        except Exception as e:
            print(f"Error preparing prediction data: {e}")
            return np.array([])

    def build_enhanced_model(self, input_shape):
        try:
            inputs = layers.Input(shape=input_shape)
            
            x = layers.LSTM(128, return_sequences=False, dropout=0.1)(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)
            
            dense1 = layers.Dense(64, activation='relu')(x)
            dense1 = layers.BatchNormalization()(dense1)
            dense1 = layers.Dropout(0.1)(dense1)
            
            outputs = layers.Dense(self.prediction_horizon, activation='linear')(dense1)
            
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            optimizer = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
            
            model.compile(
                optimizer=optimizer,
                loss='huber',
                metrics=['mae', 'mse']
            )
            
            return model
            
        except Exception as e:
            print(f"Error building model: {e}")
            return None

    def train_ensemble(self, X, y, epochs=20, batch_size=64):
        try:
            if len(X) == 0 or len(y) == 0:
                print("No data to train on")
                return [], {}
            
            ensemble_results = []
            
            print("\nTraining optimized model...")
            
            model = self.build_enhanced_model((X.shape[1], X.shape[2]))
            if model is None:
                return [], {}
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    min_delta=0.0001
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    min_delta=0.0001
                )
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=1
            )
            
            val_loss, val_mae, val_mse = model.evaluate(X_val, y_val, verbose=0)
            accuracy = max(0, min(100, (1 - val_loss) * 100))
            
            print(f"\n🎉 Training Complete! Estimated Accuracy: {accuracy:.2f}%")
            
            self.model = model
            self.ensemble_models.append(model)
            
            return [model], {
                'loss': val_loss,
                'mae': val_mae,
                'accuracy': accuracy
            }
            
        except Exception as e:
            print(f"Error training model: {e}")
            return [], {}

    def predict_ensemble(self, X):
        try:
            if self.model is None or X.size == 0:
                return None
            
            predictions = self.model.predict(X, verbose=0)
            last_step_predictions = predictions[:, -1].reshape(-1, 1)
            actual_predictions = self.target_scaler.inverse_transform(last_step_predictions)
            
            return actual_predictions.flatten()
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    def predict(self, X):
        return self.predict_ensemble(X)

    def calculate_advanced_accuracy(self, y_true, y_pred):
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return 0
            
            y_true_direction = np.sign(np.diff(y_true))
            y_pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
            
            price_accuracy = np.mean(np.abs(y_true - y_pred) / y_true < 0.03) * 100
            r2 = r2_score(y_true, y_pred) * 100
            combined_accuracy = (directional_accuracy + price_accuracy + max(0, r2)) / 3
            
            return max(0, min(100, combined_accuracy))
            
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            return 0

    def generate_advanced_signals(self, current_price, predicted_price, confidence=0.0, technical_indicators=None):
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
            
            # Force signal generation based on price direction
            if price_change >= 0:
                signal['signal'] = 'BUY'
                signal['target_price'] = current_price * (1 + abs(price_change) * 2.0)
                signal['stop_loss'] = current_price * (1 - abs(price_change) * 0.7)
            else:
                signal['signal'] = 'SELL'
                signal['target_price'] = current_price * (1 + price_change * 2.0)
                signal['stop_loss'] = current_price * (1 - price_change * 0.7)
            
            # Set a default confidence (e.g., 50%) since it's forced
            signal['confidence'] = 50.0
            
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