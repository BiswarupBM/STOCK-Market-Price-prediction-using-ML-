import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import tempfile

# Import required TensorFlow components
keras = tf.keras
LSTM, GRU, Dense, Dropout, Input = keras.layers.LSTM, keras.layers.GRU, keras.layers.Dense, keras.layers.Dropout, keras.layers.Input
BatchNormalization, LeakyReLU, Concatenate, Model = keras.layers.BatchNormalization, keras.layers.LeakyReLU, keras.layers.Concatenate, keras.Model
EarlyStopping, ModelCheckpoint, ReduceLROnPlateau = keras.callbacks.EarlyStopping, keras.callbacks.ModelCheckpoint, keras.callbacks.ReduceLROnPlateau

class StockPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler_fitted = False
        self.model = None
        self.feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'SMA_20', 'EMA_20', 'Volume_EMA']

    def prepare_data_and_fit_scaler(self, df):
        """Fits the scaler to the data and prepares it for training."""
        try:
            if df.empty: return np.array([]), np.array([])
            data = df[self.feature_columns].values
            scaled_data = self.scaler.fit_transform(data)
            self._scaler_fitted = True
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i])
                y.append(scaled_data[i, 0])
            return np.array(X), np.array(y)
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return np.array([]), np.array([])

    def prepare_prediction_data(self, df):
        """Prepares new data for prediction using the existing scaler."""
        if not self._scaler_fitted: raise RuntimeError("Scaler has not been fitted. Call prepare_data_and_fit_scaler first.")
        try:
            if df.empty: return np.array([])
            data = df[self.feature_columns].values
            scaled_data = self.scaler.transform(data)
            X = []
            for i in range(self.sequence_length, len(scaled_data) + 1):
                 X.append(scaled_data[i-self.sequence_length:i])
            return np.array(X)
        except Exception as e:
            print(f"Error preparing prediction data: {e}")
            return np.array([])

    def build_model(self, input_shape):
        """Builds the hybrid LSTM-GRU model."""
        inputs = Input(shape=input_shape)
        x1 = LSTM(128, return_sequences=True)(inputs)
        x1 = BatchNormalization()(x1)
        x1 = LeakyReLU()(x1)
        x1 = Dropout(0.2)(x1)
        x2 = GRU(128, return_sequences=True)(inputs)
        x2 = BatchNormalization()(x2)
        x2 = LeakyReLU()(x2)
        x2 = Dropout(0.2)(x2)
        combined = Concatenate()([x1, x2])
        x = GRU(64)(combined)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.1)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        outputs = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        return model

    def train(self, X, y, epochs=50, batch_size=64):
        """Trains the model with time-series cross-validation."""
        from sklearn.model_selection import TimeSeriesSplit
        if self.model is None: self.model = self.build_model((X.shape[1], X.shape[2]))

        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, mode='min')]

        tscv = TimeSeriesSplit(n_splits=3)
        histories, fold_scores = [], []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]
            
            with tempfile.NamedTemporaryFile(suffix='.weights.h5', delete=True) as tmp:
                checkpoint = ModelCheckpoint(tmp.name, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=True)
                history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                                         callbacks=callbacks + [checkpoint], verbose=1)
                self.model.load_weights(tmp.name)
            scores = self.model.evaluate(X_val, y_val, verbose=0)
            fold_scores.append(scores)
            histories.append(history)

        avg_scores = np.mean(fold_scores, axis=0)
        scores_dict = dict(zip(self.model.metrics_names, avg_scores))
        
        # Calculate accuracy as (1 - loss) * 100
        scores_dict['accuracy'] = (1 - scores_dict['loss']) * 100
        return histories, scores_dict

    def calculate_accuracy(self, y_true, y_pred, tolerance=0.05):
        """Calculates a simple directional accuracy based on price movements."""
        y_true_move = np.sign(y_true[1:] - y_true[:-1])
        y_pred_move = np.sign(y_pred[1:] - y_pred[:-1])
        correct_predictions = np.sum(y_true_move == y_pred_move)
        total_predictions = len(y_true_move)
        return (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

    def predict(self, X):
        """Makes predictions on new data."""
        try:
            if self.model is None: raise ValueError("Model not trained yet.")
            if not self._scaler_fitted: raise ValueError("Scaler not fitted yet.")
            if X.size == 0: return np.array([])

            predictions = self.model.predict(X, verbose=0)
            dummy_array = np.zeros((len(predictions), len(self.feature_columns)))
            dummy_array[:, 0] = predictions.flatten()
            return self.scaler.inverse_transform(dummy_array)[:, 0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def generate_signals(self, current_price, predicted_price, confidence=0.0, technical_indicators=None):
        """Generates trading signals with risk management."""
        try:
            price_diff_pct = (predicted_price - current_price) / current_price
            signal = {'signal': None, 'entry_price': current_price, 'target_price': None, 'stop_loss': None, 'confidence': 0.0}

            tech_ok = False
            if technical_indicators:
                rsi, macd = technical_indicators.get('RSI', 50), technical_indicators.get('MACD', 0)
                if price_diff_pct > 0 and rsi < 70 and macd > 0: tech_ok = True  # Buy confirmation
                elif price_diff_pct < 0 and rsi > 30 and macd < 0: tech_ok = True  # Sell confirmation

            signal_confidence = abs(price_diff_pct) * 100
            if tech_ok and signal_confidence >= confidence:
                if price_diff_pct > 0:
                    signal.update({'signal': 'BUY', 'target_price': current_price * (1 + price_diff_pct * 2), 'stop_loss': current_price * (1 - price_diff_pct)})
                else:
                    signal.update({'signal': 'SELL', 'target_price': current_price * (1 + price_diff_pct * 2), 'stop_loss': current_price * (1 - price_diff_pct)})
                signal['confidence'] = signal_confidence
            return signal
        except Exception as e:
            print(f"Error generating signals: {e}")
            return None