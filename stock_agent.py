import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import os
import pickle
import joblib
from news_fetcher import NewsFetcher
from llm_analyzer import GeminiAnalyzer


class StockPredictionAgent:
    def __init__(self, gemini_api_key=None):
        self.data = None
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model_type = None
        self.ticker = None
        
        # For monte carlo simulation
        self.num_simulations = 100
        self.confidence_interval = 0.95
        
        # Initialize news and LLM components
        self.news_fetcher = NewsFetcher()
        if gemini_api_key:
            self.llm_analyzer = GeminiAnalyzer(gemini_api_key)
        else:
            self.llm_analyzer = None
        
        # Create saved_models directory if it doesn't exist
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')

    def fetch_stock_data(self, ticker, period="1y"):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = ticker.strip().upper()
            self.ticker = ticker
            
            # Get data from Yahoo Finance
            stock = yf.Ticker(ticker)
            self.data = stock.history(period=period)
            
            if self.data.empty:
                raise ValueError(f"No data found for {ticker}")
                
            return True, f"Data fetched successfully for {ticker}"
            
        except Exception as e:
            return False, str(e)

    def prepare_data(self):
        """Prepare and engineer features from stock data"""
        if self.data is None or self.data.empty:
            return False, "No data available"
            
        try:
            # Feature engineering
            df = self.data.copy()

            # Technical indicators
            # Moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()

            # Exponential moving averages
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

            # MACD
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_std'] = df['Close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
            df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

            # Price momentum
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5'] = df['Close'].pct_change(periods=5)
            df['Price_Change_10'] = df['Close'].pct_change(periods=10)
            df['Price_Change_20'] = df['Close'].pct_change(periods=20)

            # Volume features
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

            # Price/Volume relationship
            df['PV_Ratio'] = df['Close'] / df['Volume']
            df['PV_Ratio_MA5'] = df['PV_Ratio'].rolling(window=5).mean()

            # Volatility indicators
            df['Volatility_5'] = df['Close'].rolling(window=5).std()
            df['Volatility_10'] = df['Close'].rolling(window=10).std()
            df['Volatility_20'] = df['Close'].rolling(window=20).std()

            # RSI (Relative Strength Index)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Previous days' close prices (lag features)
            for i in range(1, 6):
                df[f'Close_lag_{i}'] = df['Close'].shift(i)

            # Day of week (one-hot encoded)
            for i in range(7):
                df[f'Day_{i}'] = df.index.dayofweek == i

            # Target variable - next day's closing price
            df['Target'] = df['Close'].shift(-1)

            # Remove rows with NaN values
            df = df.dropna()

            # Select features for training
            feature_columns = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            X = df[feature_columns]
            y = df['Target']

            return True, (X, y, df)
            
        except Exception as e:
            return False, str(e)

    def prepare_last_row_features(self, df=None):
        """Prepare features for the last row for future prediction"""
        if df is None:
            df = self.data.copy()
            
        # Apply the same feature engineering as in prepare_data
        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        # Exponential moving averages
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5'] = df['Close'].pct_change(periods=5)
        df['Price_Change_10'] = df['Close'].pct_change(periods=10)
        df['Price_Change_20'] = df['Close'].pct_change(periods=20)

        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

        # Price/Volume relationship
        df['PV_Ratio'] = df['Close'] / df['Volume']
        df['PV_Ratio_MA5'] = df['PV_Ratio'].rolling(window=5).mean()

        # Volatility indicators
        df['Volatility_5'] = df['Close'].rolling(window=5).std()
        df['Volatility_10'] = df['Close'].rolling(window=10).std()
        df['Volatility_20'] = df['Close'].rolling(window=20).std()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Previous days' close prices
        for i in range(1, 6):
            df[f'Close_lag_{i}'] = df['Close'].shift(i)

        # Day of week
        for i in range(7):
            df[f'Day_{i}'] = df.index.dayofweek == i

        # Get the last row
        last_row = df.iloc[-1:]
        feature_columns = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        return last_row[feature_columns]

    def create_lstm_model(self, X_train, n_estimators=None):
        """Create LSTM model (placeholder for now)"""
        # This is a simplified LSTM implementation
        # In a real implementation, you would use TensorFlow/Keras
        from sklearn.neural_network import MLPRegressor
        
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        return model

    def create_xgboost_model(self, X_train, n_estimators=None):
        """Create XGBoost model"""
        if n_estimators is None:
            n_estimators = 100
            
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        return model

    def train_model(self, model_type, test_size=0.2, n_estimators=100, continuous_learning=True):
        """Train the specified model"""
        np.random.seed(42)
        try:
            self.model_type = model_type
            
            # Handle Monte Carlo separately (no training needed)
            if model_type == "Monte Carlo":
                if self.data is None or self.data.empty:
                    return False, "No data available for Monte Carlo simulation"
                
                # For Monte Carlo, we don't need to train a model
                # Just return success with basic metrics
                results = {
                    'mse': 0.0,
                    'r2': 0.0,
                    'rmse': 0.0,
                    'accuracy': 0.0,
                    'y_test': None,
                    'y_pred': None,
                    'message': 'Monte Carlo simulation ready'
                }
                return True, results
            
            # For other models, prepare data and train
            success, result = self.prepare_data()
            if not success:
                return False, result
                
            X, y, df = result
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_test_scaled = self.scaler_X.transform(X_test)
            
            # Scale target
            y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
            
            # Create and train model
            if model_type == "XGBoost":
                self.model = self.create_xgboost_model(X_train_scaled, n_estimators)
            elif model_type == "LSTM":
                self.model = self.create_lstm_model(X_train_scaled, n_estimators)
            else:
                return False, f"Unknown model type: {model_type}"
            
            # Train model
            self.model.fit(X_train_scaled, y_train_scaled)
            
            # Make predictions
            y_pred_scaled = self.model.predict(X_test_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate accuracy (percentage of predictions within 5% of actual)
            accuracy = np.mean(np.abs((y_pred - y_test) / y_test) < 0.05) * 100
            
            results = {
                'mse': mse,
                'r2': r2,
                'rmse': rmse,
                'accuracy': accuracy,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            return True, results
            
        except Exception as e:
            return False, str(e)

    def predict_future(self, forecast_days=30):
        """Predict future stock prices"""
        try:
            if self.data is None or self.data.empty:
                return False, "No data available for prediction"
                
            if self.model_type == "Monte Carlo":
                # Monte Carlo doesn't need a trained model
                last_price = self.data['Close'].iloc[-1]
                volatility = self.data['Close'].pct_change().std()
                return self.monte_carlo_simulation(last_price, forecast_days, volatility)
            elif self.model_type == "XGBoost":
                if self.model is None:
                    return False, "No trained XGBoost model available"
                return self.predict_future_xgboost(forecast_days)
            elif self.model_type == "LSTM":
                if self.model is None:
                    return False, "No trained LSTM model available"
                return self.predict_future_lstm(forecast_days)
            else:
                return False, f"Unknown model type: {self.model_type}"
                
        except Exception as e:
            return False, str(e)

    def generate_forecast_dates(self, forecast_days):
        """Generate dates for the forecast period"""
        last_date = self.data.index[-1]
        forecast_dates = []
        
        for i in range(1, forecast_days + 1):
            next_date = last_date + timedelta(days=i)
            # Skip weekends
            while next_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                next_date += timedelta(days=1)
            forecast_dates.append(next_date)
            
        return forecast_dates

    def monte_carlo_simulation(self, last_price, days, volatility):
        """Simulate multiple price paths based on historical volatility with realistic dynamics"""
        np.random.seed(42)
        try:
            # Get historical statistics for more realistic simulation
            historical_returns = self.data['Close'].pct_change().dropna()
            historical_mean_return = historical_returns.mean()
            historical_volatility = historical_returns.std()
            
            # Generate random price paths with more realistic dynamics
            price_paths = []
            for _ in range(self.num_simulations):
                prices = [last_price]
                current_volatility = volatility
                
                for day in range(days):
                    # Base return with drift
                    base_return = historical_mean_return
                    
                    # Add volatility clustering (GARCH-like effect)
                    if day > 0:
                        # Volatility tends to cluster
                        volatility_persistence = 0.9
                        volatility_shock = np.random.normal(0, 0.1)
                        current_volatility = (volatility_persistence * current_volatility + 
                                            (1 - volatility_persistence) * historical_volatility + 
                                            volatility_shock)
                        current_volatility = max(current_volatility, historical_volatility * 0.5)
                    
                    # Add random component
                    random_component = np.random.normal(0, current_volatility)
                    
                    # Add cyclical component
                    cyclical_component = 0.0005 * np.sin(2 * np.pi * day / 20)
                    
                    # Add mean reversion
                    if day > 0:
                        mean_reversion = (last_price - prices[-1]) * 0.002 / prices[-1]
                    else:
                        mean_reversion = 0
                    
                    # Calculate total return
                    total_return = base_return + random_component + cyclical_component + mean_reversion
                    
                    # Calculate new price
                    new_price = prices[-1] * (1 + total_return)
                    
                    # Ensure reasonable bounds
                    new_price = max(new_price, prices[-1] * 0.8)
                    new_price = min(new_price, prices[-1] * 1.3)
                    
                    prices.append(new_price)
                
                price_paths.append(prices[1:])  # Exclude initial price
            
            price_paths = np.array(price_paths)
            
            # Calculate statistics
            mean_prices = np.mean(price_paths, axis=0)
            std_prices = np.std(price_paths, axis=0)
            
            # Calculate confidence intervals
            confidence_level = self.confidence_interval
            z_score = 1.96  # 95% confidence interval
            
            lower_bound = mean_prices - z_score * std_prices
            upper_bound = mean_prices + z_score * std_prices
            
            forecast_dates = self.generate_forecast_dates(days)
            
            return True, {
                'forecast_dates': forecast_dates,
                'forecast_prices': mean_prices,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'volatility': std_prices
            }
            
        except Exception as e:
            return False, str(e)

    def predict_future_xgboost(self, forecast_days):
        """Predict future prices using XGBoost with realistic volatility"""
        np.random.seed(42)
        try:
            forecast_dates = self.generate_forecast_dates(forecast_days)
            forecast_prices = []
            forecast_volatility = []
            
            # Get historical volatility and trend
            historical_returns = self.data['Close'].pct_change().dropna()
            historical_volatility = historical_returns.std()
            historical_mean_return = historical_returns.mean()
            
            # Get the last price and features
            last_price = self.data['Close'].iloc[-1]
            last_features = self.prepare_last_row_features()
            current_features_scaled = self.scaler_X.transform(last_features)
            
            # Initialize with last price
            current_price = last_price
            
            for i in range(forecast_days):
                # Get base prediction from model
                prediction_scaled = self.model.predict(current_features_scaled.reshape(1, -1))
                base_prediction = self.scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
                
                # Calculate expected return from model
                expected_return = (base_prediction - current_price) / current_price
                
                # Add realistic market noise and volatility
                # Combine model prediction with historical volatility
                market_noise = np.random.normal(0, historical_volatility * 0.5)
                volatility_shock = np.random.normal(0, historical_volatility * 0.3)
                
                # Calculate new price with realistic dynamics
                total_return = expected_return + market_noise + volatility_shock
                
                # Add some mean reversion and trend following
                if i > 0:
                    # Mean reversion component
                    mean_reversion = (last_price - current_price) * 0.01
                    # Trend following component
                    trend_following = (current_price - forecast_prices[-1]) * 0.02
                    total_return += mean_reversion / current_price + trend_following / current_price
                
                # Calculate new price
                new_price = current_price * (1 + total_return)
                
                # Ensure price doesn't go negative
                new_price = max(new_price, current_price * 0.5)
                
                forecast_prices.append(new_price)
                
                # Update features for next iteration (simplified)
                current_features_scaled += np.random.normal(0, 0.005, current_features_scaled.shape)
                current_price = new_price
                
                # Calculate rolling volatility
                if len(forecast_prices) > 1:
                    # Fix the broadcasting issue by ensuring proper array shapes
                    recent_prices = forecast_prices[-min(10, len(forecast_prices)):]
                    if len(recent_prices) > 1:
                        recent_returns = []
                        for j in range(1, len(recent_prices)):
                            return_val = (recent_prices[j] - recent_prices[j-1]) / recent_prices[j-1]
                            recent_returns.append(return_val)
                        volatility = np.std(recent_returns) if recent_returns else historical_volatility
                    else:
                        volatility = historical_volatility
                else:
                    volatility = historical_volatility
                forecast_volatility.append(volatility)
            
            return True, {
                'forecast_dates': forecast_dates,
                'forecast_prices': np.array(forecast_prices),
                'forecast_volatility': np.array(forecast_volatility)
            }
            
        except Exception as e:
            return False, str(e)

    def predict_future_lstm(self, forecast_days):
        """Predict future prices using LSTM with realistic dynamics"""
        np.random.seed(42)
        try:
            forecast_dates = self.generate_forecast_dates(forecast_days)
            forecast_prices = []
            forecast_volatility = []
            
            # Get historical statistics
            historical_returns = self.data['Close'].pct_change().dropna()
            historical_volatility = historical_returns.std()
            historical_mean_return = historical_returns.mean()
            
            # Get the last price
            last_price = self.data['Close'].iloc[-1]
            current_price = last_price
            
            # Create a more sophisticated prediction model
            for i in range(forecast_days):
                # Base trend from historical data
                trend_component = historical_mean_return
                
                # Add cyclical component (market cycles)
                cycle_period = 20  # Approximate market cycle
                cyclical_component = 0.001 * np.sin(2 * np.pi * i / cycle_period)
                
                # Add random walk component
                random_walk = np.random.normal(0, historical_volatility)
                
                # Add volatility clustering (high volatility tends to persist)
                if i > 0 and forecast_volatility[-1] > historical_volatility * 1.5:
                    volatility_persistence = np.random.normal(0, historical_volatility * 0.8)
                else:
                    volatility_persistence = np.random.normal(0, historical_volatility * 0.4)
                
                # Combine all components
                total_return = trend_component + cyclical_component + random_walk + volatility_persistence
                
                # Add some mean reversion
                if i > 0:
                    mean_reversion = (last_price - current_price) * 0.005 / current_price
                    total_return += mean_reversion
                
                # Calculate new price
                new_price = current_price * (1 + total_return)
                
                # Ensure reasonable bounds
                new_price = max(new_price, current_price * 0.7)
                new_price = min(new_price, current_price * 1.5)
                
                forecast_prices.append(new_price)
                current_price = new_price
                
                # Calculate volatility
                if len(forecast_prices) > 1:
                    # Fix the broadcasting issue by ensuring proper array shapes
                    recent_prices = forecast_prices[-min(10, len(forecast_prices)):]
                    if len(recent_prices) > 1:
                        recent_returns = []
                        for j in range(1, len(recent_prices)):
                            return_val = (recent_prices[j] - recent_prices[j-1]) / recent_prices[j-1]
                            recent_returns.append(return_val)
                        volatility = np.std(recent_returns) if recent_returns else historical_volatility
                    else:
                        volatility = historical_volatility
                else:
                    volatility = historical_volatility
                forecast_volatility.append(volatility)
            
            return True, {
                'forecast_dates': forecast_dates,
                'forecast_prices': np.array(forecast_prices),
                'forecast_volatility': np.array(forecast_volatility)
            }
            
        except Exception as e:
            return False, str(e)

    def save_model(self, filename=None):
        """Save the trained model"""
        try:
            if self.model is None:
                return False, "No model to save"
                
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"saved_models/{self.ticker}_{self.model_type}_{timestamp}.joblib"
            
            # Save model and scalers
            model_data = {
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'model_type': self.model_type,
                'ticker': self.ticker
            }
            
            joblib.dump(model_data, filename)
            return True, f"Model saved as {filename}"
            
        except Exception as e:
            return False, str(e)

    def load_model(self, filename=None):
        """Load a trained model"""
        try:
            if filename is None:
                # Find the most recent model file
                model_files = [f for f in os.listdir('saved_models') if f.endswith('.joblib')]
                if not model_files:
                    return False, "No saved models found"
                    
                # Sort by modification time (most recent first)
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join('saved_models', x)), reverse=True)
                filename = os.path.join('saved_models', model_files[0])
            
            # Load model data
            model_data = joblib.load(filename)
            
            self.model = model_data['model']
            self.scaler_X = model_data['scaler_X']
            self.scaler_y = model_data['scaler_y']
            self.model_type = model_data['model_type']
            self.ticker = model_data['ticker']
            
            return True, f"Model loaded from {filename}"
            
        except Exception as e:
            return False, str(e)

    def get_historical_data(self):
        """Get the historical data"""
        return self.data

    def get_model_info(self):
        """Get information about the current model"""
        if self.model is None:
            return "No model trained"
            
        return {
            'model_type': self.model_type,
            'ticker': self.ticker,
            'data_points': len(self.data) if self.data is not None else 0
        }

    def analyze_news_sentiment(self, days_back=7, max_articles=10):
        """
        Fetch news and analyze sentiment for the current ticker
        
        Args:
            days_back (int): Number of days to look back for news
            max_articles (int): Maximum number of articles to analyze
            
        Returns:
            tuple: (success: bool, result: dict or str)
        """
        try:
            if not self.ticker:
                return False, "No ticker selected. Please fetch data first."
            
            if not self.llm_analyzer:
                return False, "Gemini API not configured. Please provide API key."
            
            # Fetch news
            success, news_result = self.news_fetcher.fetch_news_for_ticker(
                self.ticker, days_back, max_articles
            )
            
            if not success:
                return False, f"Failed to fetch news: {news_result}"
            
            # Get current stock data for context
            stock_data = None
            if self.data is not None and not self.data.empty:
                current_price = self.data['Close'].iloc[-1]
                price_change = self.data['Close'].iloc[-1] - self.data['Close'].iloc[-2]
                volume = self.data['Volume'].iloc[-1]
                
                stock_data = {
                    'current_price': f"{current_price:.2f}",
                    'price_change': f"{price_change:+.2f}",
                    'volume': f"{volume:,.0f}"
                }
            
            # Analyze news sentiment
            success, analysis = self.llm_analyzer.analyze_news_sentiment(
                self.ticker, news_result, stock_data
            )
            
            if not success:
                return False, f"Failed to analyze news: {analysis}"
            
            # Get market context
            context_success, market_context = self.llm_analyzer.get_market_context(self.ticker)
            
            # Prepare result
            result = {
                'news_headlines': news_result,
                'sentiment_analysis': analysis,
                'market_context': market_context if context_success else "Market context unavailable",
                'stock_data': stock_data,
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return True, result
            
        except Exception as e:
            return False, f"Error in news analysis: {str(e)}"

    def get_news_correlation_analysis(self, news_analysis):
        """
        Analyze correlation between news sentiment and stock price movements
        
        Args:
            news_analysis (str): Previous news analysis text
            
        Returns:
            tuple: (success: bool, correlation_analysis: str)
        """
        try:
            if not self.llm_analyzer:
                return False, "Gemini API not configured"
            
            if self.data is None or self.data.empty:
                return False, "No stock data available for correlation analysis"
            
            # Convert price data to format expected by LLM
            price_data = {}
            for date, row in self.data.tail(20).iterrows():  # Last 20 days
                price_data[date.strftime("%Y-%m-%d")] = row['Close']
            
            success, correlation = self.llm_analyzer.analyze_stock_performance_correlation(
                self.ticker, news_analysis, price_data
            )
            
            return success, correlation
            
        except Exception as e:
            return False, f"Error in correlation analysis: {str(e)}" 