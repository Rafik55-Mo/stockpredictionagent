# Stock Price Predictor - Separated Architecture

This project has been refactored to separate the GUI from the agent code, following clean architecture principles.

## Project Structure

```
├── main.py              # Entry point - launches the application
├── stock_agent.py       # Core ML logic and data processing
├── stock_gui.py         # Tkinter GUI interface
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── saved_models/       # Directory for saved models (created automatically)
```

## Architecture Overview

### 1. StockPredictionAgent (`stock_agent.py`)
- **Purpose**: Core machine learning logic and data processing
- **Responsibilities**:
  - Fetch stock data from Yahoo Finance
  - Feature engineering and data preparation
  - Model training (XGBoost, LSTM, Monte Carlo)
  - Future price predictions
  - Model saving/loading
  - Technical indicators calculation

### 2. StockPredictorGUI (`stock_gui.py`)
- **Purpose**: User interface and visualization
- **Responsibilities**:
  - Tkinter GUI components
  - User input handling
  - Chart visualization with matplotlib
  - Results display
  - Status updates
  - Error handling and user feedback

### 3. Main Entry Point (`main.py`)
- **Purpose**: Application launcher
- **Responsibilities**:
  - Initialize the main window
  - Create GUI instance
  - Start the event loop

## Benefits of Separation

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Agent logic can be tested independently of GUI
3. **Reusability**: Agent can be used with different interfaces (CLI, web, etc.)
4. **Modularity**: Easy to modify or replace individual components
5. **Clean Code**: Clear separation of concerns

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

## Usage

1. **Fetch Data**: Enter a stock ticker (e.g., AAPL) and click "Fetch Data"
2. **Train Model**: Select model type and parameters, then click "Train Model & Predict"
3. **View Results**: Training metrics and predictions are displayed
4. **Save/Load Models**: Use the save/load buttons to persist models

## Model Types

- **XGBoost**: Gradient boosting algorithm for regression
- **LSTM**: Long Short-Term Memory neural network (simplified implementation)
- **Monte Carlo**: Statistical simulation based on historical volatility

## Features

- Real-time stock data fetching
- Advanced technical indicators
- Multiple ML algorithms
- Interactive charts with matplotlib
- Model persistence
- Confidence intervals for predictions
- Continuous learning support

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `yfinance`: Yahoo Finance data
- `matplotlib`: Chart visualization
- `scikit-learn`: Machine learning algorithms
- `xgboost`: Gradient boosting
- `joblib`: Model serialization
- `tkinter`: GUI framework (built-in)

## Future Enhancements

- Add more ML algorithms (Random Forest, SVM, etc.)
- Implement proper LSTM with TensorFlow/Keras
- Add more technical indicators
- Web-based interface option
- Real-time data streaming
- Portfolio optimization features 