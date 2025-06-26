#!/usr/bin/env python3
"""
Test script to visualize improved predictions
"""

import matplotlib.pyplot as plt
import numpy as np
from stock_agent import StockPredictionAgent

def test_realistic_predictions():
    """Test and visualize realistic predictions"""
    print("Testing Realistic Predictions...")
    
    # Create agent
    agent = StockPredictionAgent()
    
    try:
        # Fetch data
        print("1. Fetching data...")
        success, message = agent.fetch_stock_data("AAPL", "1y")
        if not success:
            print(f"❌ Data fetch failed: {message}")
            return False
        
        # Test different models
        models = ["XGBoost", "LSTM", "Monte Carlo"]
        results = {}
        
        for model_type in models:
            print(f"\n2. Testing {model_type}...")
            
            # Train model (skip for Monte Carlo)
            if model_type != "Monte Carlo":
                success, train_result = agent.train_model(model_type, test_size=0.2, n_estimators=50)
                if not success:
                    print(f"❌ {model_type} training failed: {train_result}")
                    continue
            
            # Make predictions
            success, pred_result = agent.predict_future(30)  # 30 days forecast
            if success:
                print(f"✅ {model_type} prediction successful")
                results[model_type] = pred_result
            else:
                print(f"❌ {model_type} prediction failed: {pred_result}")
        
        # Visualize results
        if results:
            visualize_predictions(agent.data, results)
            print("\n🎉 Prediction visualization complete!")
            return True
        else:
            print("\n❌ No successful predictions to visualize")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        return False

def visualize_predictions(historical_data, predictions):
    """Visualize historical data and predictions"""
    plt.style.use('dark_background')
    
    fig, axes = plt.subplots(len(predictions), 1, figsize=(12, 4*len(predictions)))
    if len(predictions) == 1:
        axes = [axes]
    
    colors = ['#4a9eff', '#ff6b6b', '#50c878']
    
    for i, (model_name, pred_data) in enumerate(predictions.items()):
        ax = axes[i]
        
        # Plot historical data
        ax.plot(historical_data.index, historical_data['Close'], 
                label='Historical', color='#ffffff', linewidth=2, alpha=0.8)
        
        # Plot predictions
        if 'forecast_dates' in pred_data and 'forecast_prices' in pred_data:
            forecast_dates = pred_data['forecast_dates']
            forecast_prices = pred_data['forecast_prices']
            
            ax.plot(forecast_dates, forecast_prices, 
                    label=f'{model_name} Prediction', color=colors[i], linewidth=2)
            
            # Plot confidence intervals for Monte Carlo
            if model_name == "Monte Carlo" and 'lower_bound' in pred_data and 'upper_bound' in pred_data:
                ax.fill_between(forecast_dates, pred_data['lower_bound'], pred_data['upper_bound'],
                               alpha=0.3, color=colors[i], label='Confidence Interval')
        
        ax.set_title(f'{model_name} Prediction', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Price (USD)', color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Style the plot
        ax.spines['bottom'].set_color('#555555')
        ax.spines['top'].set_color('#555555')
        ax.spines['left'].set_color('#555555')
        ax.spines['right'].set_color('#555555')
        ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_realistic_predictions() 