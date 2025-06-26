import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from stock_agent import StockPredictionAgent


class StockPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Predictor with Continuous Learning & News Analysis")
        self.root.geometry("1400x900")
        
        # Dark mode colors
        self.bg_color = "#2b2b2b"  # Dark background
        self.fg_color = "#ffffff"  # White text
        self.accent_color = "#4a9eff"  # Blue accent
        self.secondary_bg = "#3c3c3c"  # Slightly lighter background
        self.border_color = "#555555"  # Border color
        
        # Configure dark theme
        self.setup_dark_theme()
        
        # Initialize the agent with Gemini API key
        gemini_api_key = "AIzaSyCThrV8VdcZ4TEDTAVNKKq4hOMLVY3jErE"
        self.agent = StockPredictionAgent(gemini_api_key)

        # Variables
        self.ticker_var = tk.StringVar()
        self.period_var = tk.StringVar(value="1y")
        self.forecast_days_var = tk.StringVar(value="30")
        self.test_size_var = tk.StringVar(value="0.2")
        self.n_estimators_var = tk.StringVar(value="100")
        self.model_type_var = tk.StringVar(value="XGBoost")
        self.continuous_learning_var = tk.BooleanVar(value=True)

        # Create frames
        self.create_input_frame()
        self.create_chart_frame()
        self.create_result_frame()
        self.create_news_frame()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready")

    def setup_dark_theme(self):
        """Configure dark theme styling"""
        # Configure matplotlib for dark mode
        plt.style.use('dark_background')
        
        # Configure ttk styles for dark mode
        style = ttk.Style()
        style.theme_use('clam')  # Use clam theme as base
        
        # Configure colors for ttk widgets
        style.configure('TLabel', background=self.bg_color, foreground=self.fg_color)
        style.configure('TButton', background=self.accent_color, foreground=self.fg_color)
        style.configure('TEntry', fieldbackground=self.secondary_bg, foreground=self.fg_color)
        style.configure('TCombobox', fieldbackground=self.secondary_bg, foreground=self.fg_color)
        style.configure('TCheckbutton', background=self.bg_color, foreground=self.fg_color)
        style.configure('TLabelframe', background=self.bg_color, foreground=self.fg_color)
        style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.fg_color)
        style.configure('TFrame', background=self.bg_color)
        
        # Configure the root window
        self.root.configure(bg=self.bg_color)
        
        # Configure matplotlib figure colors
        plt.rcParams['figure.facecolor'] = self.bg_color
        plt.rcParams['axes.facecolor'] = self.bg_color
        plt.rcParams['axes.edgecolor'] = self.border_color
        plt.rcParams['axes.labelcolor'] = self.fg_color
        plt.rcParams['xtick.color'] = self.fg_color
        plt.rcParams['ytick.color'] = self.fg_color
        plt.rcParams['text.color'] = self.fg_color
        plt.rcParams['grid.color'] = self.border_color

    def create_input_frame(self):
        """Create the input parameters frame"""
        input_frame = ttk.LabelFrame(self.root, text="Input Parameters", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Ticker input
        ttk.Label(input_frame, text="Stock Ticker:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ticker_entry = ttk.Entry(input_frame, textvariable=self.ticker_var, width=10)
        ticker_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ticker_entry.insert(0, "AAPL")

        # Period selection
        ttk.Label(input_frame, text="Historical Period:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        period_combo = ttk.Combobox(input_frame, textvariable=self.period_var, width=5)
        period_combo['values'] = ('3mo', '6mo', '1y', '2y', '5y', '10y', 'max')
        period_combo.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        # Forecast days
        ttk.Label(input_frame, text="Forecast Days:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        forecast_entry = ttk.Entry(input_frame, textvariable=self.forecast_days_var, width=5)
        forecast_entry.grid(row=0, column=5, sticky=tk.W, padx=5, pady=5)

        # Test size
        ttk.Label(input_frame, text="Test Size:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        test_size_entry = ttk.Entry(input_frame, textvariable=self.test_size_var, width=5)
        test_size_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Model selection
        ttk.Label(input_frame, text="Model Type:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        model_combo = ttk.Combobox(input_frame, textvariable=self.model_type_var, width=15)
        model_combo['values'] = ('XGBoost', 'LSTM', 'Monte Carlo')
        model_combo.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # N estimators
        ttk.Label(input_frame, text="Trees/Iterations:").grid(row=1, column=4, sticky=tk.W, padx=5, pady=5)
        n_estimators_entry = ttk.Entry(input_frame, textvariable=self.n_estimators_var, width=5)
        n_estimators_entry.grid(row=1, column=5, sticky=tk.W, padx=5, pady=5)

        # Continuous learning checkbox
        continuous_cb = ttk.Checkbutton(input_frame, text="Continuous Learning",
                                        variable=self.continuous_learning_var)
        continuous_cb.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        # Buttons
        ttk.Button(input_frame, text="Fetch Data", command=self.fetch_data).grid(row=2, column=2, padx=5, pady=5)
        ttk.Button(input_frame, text="Train Model & Predict", command=self.train_and_predict).grid(row=2, column=3,
                                                                                                   padx=5, pady=5)
        ttk.Button(input_frame, text="Save Model", command=self.save_model).grid(row=2, column=4, padx=5, pady=5)
        ttk.Button(input_frame, text="Load Model", command=self.load_model).grid(row=2, column=5, padx=5, pady=5)

    def create_chart_frame(self):
        """Create the chart display frame"""
        self.chart_frame = ttk.LabelFrame(self.root, text="Stock Price Chart", padding=10)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create figure and canvas with dark theme
        self.fig = Figure(figsize=(12, 6), dpi=100, facecolor=self.bg_color)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(bg=self.bg_color)
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Add toolbar with dark theme
        toolbar_frame = ttk.Frame(self.chart_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.config(bg=self.bg_color)
        toolbar._message_label.config(bg=self.bg_color, fg=self.fg_color)
        toolbar.update()

    def create_result_frame(self):
        """Create the results display frame"""
        result_frame = ttk.LabelFrame(self.root, text="Model Results", padding=10)
        result_frame.pack(fill=tk.X, padx=10, pady=10)

        # Create a text widget for results with dark theme
        self.result_text = tk.Text(result_frame, height=5, width=80, 
                                  bg=self.secondary_bg, fg=self.fg_color,
                                  insertbackground=self.fg_color,
                                  selectbackground=self.accent_color,
                                  selectforeground=self.fg_color)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def create_news_frame(self):
        """Create the news analysis frame"""
        news_frame = ttk.LabelFrame(self.root, text="News Analysis", padding=10)
        news_frame.pack(fill=tk.X, padx=10, pady=10)

        # News analysis button
        ttk.Button(news_frame, text="Analyze News Sentiment", command=self.analyze_news).grid(row=0, column=0, padx=5, pady=5)

        # Create a text widget for news analysis with dark theme
        self.news_text = tk.Text(news_frame, height=8, width=80,
                                bg=self.secondary_bg, fg=self.fg_color,
                                insertbackground=self.fg_color,
                                selectbackground=self.accent_color,
                                selectforeground=self.fg_color)
        self.news_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Add scrollbar
        news_scrollbar = ttk.Scrollbar(news_frame, orient="vertical", command=self.news_text.yview)
        news_scrollbar.grid(row=1, column=2, sticky="ns")
        self.news_text.configure(yscrollcommand=news_scrollbar.set)

    def fetch_data(self):
        """Fetch stock data using the agent"""
        try:
            ticker = self.ticker_var.get().strip().upper()
            period = self.period_var.get()

            if not ticker:
                messagebox.showerror("Error", "Please enter a valid ticker symbol")
                return

            self.status_var.set(f"Fetching data for {ticker}...")
            self.root.update_idletasks()

            # Use agent to fetch data
            success, message = self.agent.fetch_stock_data(ticker, period)

            if not success:
                messagebox.showerror("Error", message)
                self.status_var.set("Error fetching data")
                return

            # Display historical data
            self.plot_historical_data()
            self.status_var.set(message)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error fetching data")

    def plot_historical_data(self):
        """Plot historical stock data"""
        data = self.agent.get_historical_data()
        if data is None or data.empty:
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Plot closing price with dark theme colors
        ax.plot(data.index, data['Close'], label='Close Price', color='#4a9eff', linewidth=2)

        # Format plot with dark theme
        ax.set_title(f"{self.ticker_var.get()} Historical Prices", color=self.fg_color, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', color=self.fg_color, fontsize=12)
        ax.set_ylabel('Price (USD)', color=self.fg_color, fontsize=12)
        ax.legend(facecolor=self.bg_color, edgecolor=self.border_color, labelcolor=self.fg_color)
        ax.grid(True, alpha=0.3, color=self.border_color)
        
        # Set axis colors
        ax.spines['bottom'].set_color(self.border_color)
        ax.spines['top'].set_color(self.border_color)
        ax.spines['left'].set_color(self.border_color)
        ax.spines['right'].set_color(self.border_color)
        ax.tick_params(colors=self.fg_color)

        # Format date axis
        self.fig.autofmt_xdate()

        self.canvas.draw()

    def train_and_predict(self):
        """Train model and make predictions"""
        try:
            # Get parameters
            model_type = self.model_type_var.get()
            test_size = float(self.test_size_var.get())
            n_estimators = int(self.n_estimators_var.get())
            forecast_days = int(self.forecast_days_var.get())
            continuous_learning = self.continuous_learning_var.get()

            if self.agent.data is None:
                messagebox.showerror("Error", "Please fetch data first")
                return

            self.status_var.set("Training model...")
            self.root.update_idletasks()

            # Train model using agent
            success, result = self.agent.train_model(model_type, test_size, n_estimators, continuous_learning)

            if not success:
                messagebox.showerror("Error", result)
                self.status_var.set("Error training model")
                return

            # Display training results
            self.display_training_results(result)

            # Make future predictions
            self.status_var.set("Making predictions...")
            self.root.update_idletasks()

            success, prediction_result = self.agent.predict_future(forecast_days)

            if not success:
                messagebox.showerror("Error", prediction_result)
                self.status_var.set("Error making predictions")
                return

            # Plot predictions
            self.plot_predictions(prediction_result)
            self.status_var.set("Training and prediction completed")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error in training and prediction")

    def display_training_results(self, results):
        """Display training results in the text widget"""
        self.result_text.delete(1.0, tk.END)
        
        model_type = self.model_type_var.get()
        
        if model_type == "Monte Carlo":
            result_text = f"""
Monte Carlo Simulation Results:
==============================
{results.get('message', 'Monte Carlo simulation ready')}

Model Type: {model_type}
Ticker: {self.ticker_var.get()}
Note: Monte Carlo simulation uses statistical methods and doesn't require traditional training.
        """
        else:
            result_text = f"""
Model Training Results:
======================
Mean Squared Error: {results['mse']:.4f}
Root Mean Squared Error: {results['rmse']:.4f}
R² Score: {results['r2']:.4f}
Accuracy (within 5%): {results['accuracy']:.2f}%

Model Type: {model_type}
Ticker: {self.ticker_var.get()}
        """
        
        self.result_text.insert(tk.END, result_text)

    def plot_predictions(self, prediction_result):
        """Plot predictions on the chart"""
        data = self.agent.get_historical_data()
        if data is None or data.empty:
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Plot historical data with dark theme colors
        ax.plot(data.index, data['Close'], label='Historical Close Price', color='#4a9eff', linewidth=2)

        # Plot predictions
        if 'forecast_dates' in prediction_result and 'forecast_prices' in prediction_result:
            forecast_dates = prediction_result['forecast_dates']
            forecast_prices = prediction_result['forecast_prices']
            
            ax.plot(forecast_dates, forecast_prices, label='Predicted Price', color='#ff6b6b', linestyle='--', linewidth=2)

            # Plot confidence intervals if available
            if 'lower_bound' in prediction_result and 'upper_bound' in prediction_result:
                lower_bound = prediction_result['lower_bound']
                upper_bound = prediction_result['upper_bound']
                
                ax.fill_between(forecast_dates, lower_bound, upper_bound, 
                               alpha=0.3, color='#ff6b6b', label='Confidence Interval')

        # Format plot with dark theme
        ax.set_title(f"{self.ticker_var.get()} Stock Price Prediction", color=self.fg_color, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', color=self.fg_color, fontsize=12)
        ax.set_ylabel('Price (USD)', color=self.fg_color, fontsize=12)
        ax.legend(facecolor=self.bg_color, edgecolor=self.border_color, labelcolor=self.fg_color)
        ax.grid(True, alpha=0.3, color=self.border_color)
        
        # Set axis colors
        ax.spines['bottom'].set_color(self.border_color)
        ax.spines['top'].set_color(self.border_color)
        ax.spines['left'].set_color(self.border_color)
        ax.spines['right'].set_color(self.border_color)
        ax.tick_params(colors=self.fg_color)

        # Format date axis
        self.fig.autofmt_xdate()

        self.canvas.draw()

    def save_model(self):
        """Save the trained model"""
        try:
            if self.agent.model is None:
                messagebox.showerror("Error", "No trained model to save")
                return

            success, message = self.agent.save_model()
            
            if success:
                messagebox.showinfo("Success", message)
                self.status_var.set("Model saved successfully")
            else:
                messagebox.showerror("Error", message)
                self.status_var.set("Error saving model")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error saving model")

    def load_model(self):
        """Load a trained model"""
        try:
            success, message = self.agent.load_model()
            
            if success:
                messagebox.showinfo("Success", message)
                self.status_var.set("Model loaded successfully")
                
                # Update GUI with loaded model info
                model_info = self.agent.get_model_info()
                if isinstance(model_info, dict):
                    self.ticker_var.set(model_info.get('ticker', ''))
                    self.model_type_var.set(model_info.get('model_type', ''))
                    
                    # Plot historical data if available
                    if self.agent.data is not None:
                        self.plot_historical_data()
            else:
                messagebox.showerror("Error", message)
                self.status_var.set("Error loading model")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error loading model")

    def analyze_news(self):
        """Analyze news sentiment"""
        try:
            ticker = self.ticker_var.get().strip().upper()
            if not ticker:
                messagebox.showerror("Error", "Please enter a valid ticker symbol")
                return

            self.status_var.set(f"Analyzing news sentiment for {ticker}...")
            self.root.update_idletasks()

            # Use agent to analyze news sentiment
            success, result = self.agent.analyze_news_sentiment()

            if not success:
                messagebox.showerror("Error", str(result))
                self.status_var.set("Error analyzing news sentiment")
                return

            # Display news sentiment analysis
            self.display_news_sentiment_analysis(result)
            self.status_var.set("News sentiment analysis completed")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error analyzing news sentiment")

    def display_news_sentiment_analysis(self, analysis_result):
        """Display news sentiment analysis in the news text widget"""
        self.news_text.delete(1.0, tk.END)
        
        if isinstance(analysis_result, dict):
            # Format the analysis result
            result_text = f"""
NEWS SENTIMENT ANALYSIS
=======================
Analysis Date: {analysis_result.get('analysis_date', 'N/A')}

STOCK DATA:
"""
            if analysis_result.get('stock_data'):
                stock_data = analysis_result['stock_data']
                result_text += f"Current Price: ${stock_data.get('current_price', 'N/A')}\n"
                result_text += f"Price Change: {stock_data.get('price_change', 'N/A')}\n"
                result_text += f"Volume: {stock_data.get('volume', 'N/A')}\n"
            
            result_text += f"""

NEWS HEADLINES:
"""
            for i, headline in enumerate(analysis_result.get('news_headlines', []), 1):
                result_text += f"{i}. {headline}\n"
            
            result_text += f"""

SENTIMENT ANALYSIS:
{analysis_result.get('sentiment_analysis', 'No analysis available')}

MARKET CONTEXT:
{analysis_result.get('market_context', 'No market context available')}
"""
        else:
            result_text = f"Analysis Result: {analysis_result}"
        
        self.news_text.insert(tk.END, result_text)

    def update_status(self, message):
        """Update the status bar"""
        self.status_var.set(message)
        self.root.update_idletasks() 