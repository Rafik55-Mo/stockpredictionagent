#!/usr/bin/env python3
"""
Test script to verify ML models are working correctly
"""

import sys
import traceback
from stock_agent import StockPredictionAgent

def test_agent():
    """Test the stock prediction agent"""
    print("Testing Stock Prediction Agent...")
    
    # Create agent
    agent = StockPredictionAgent()
    
    try:
        # Test 1: Fetch data
        print("\n1. Testing data fetching...")
        success, message = agent.fetch_stock_data("AAPL", "1y")
        if success:
            print(f"✅ Data fetch successful: {message}")
            print(f"   Data shape: {agent.data.shape}")
        else:
            print(f"❌ Data fetch failed: {message}")
            return False
        
        # Test 2: XGBoost model
        print("\n2. Testing XGBoost model...")
        success, results = agent.train_model("XGBoost", test_size=0.2, n_estimators=50)
        if success:
            print(f"✅ XGBoost training successful")
            print(f"   R² Score: {results['r2']:.4f}")
            print(f"   RMSE: {results['rmse']:.4f}")
            
            # Test prediction
            success, pred_results = agent.predict_future(10)
            if success:
                print(f"✅ XGBoost prediction successful")
                print(f"   Predicted {len(pred_results['forecast_prices'])} future prices")
            else:
                print(f"❌ XGBoost prediction failed: {pred_results}")
        else:
            print(f"❌ XGBoost training failed: {results}")
        
        # Test 3: Monte Carlo model
        print("\n3. Testing Monte Carlo simulation...")
        success, results = agent.train_model("Monte Carlo")
        if success:
            print(f"✅ Monte Carlo setup successful")
            
            # Test prediction
            success, pred_results = agent.predict_future(10)
            if success:
                print(f"✅ Monte Carlo prediction successful")
                print(f"   Predicted {len(pred_results['forecast_prices'])} future prices")
                print(f"   Confidence intervals available: {'lower_bound' in pred_results}")
            else:
                print(f"❌ Monte Carlo prediction failed: {pred_results}")
        else:
            print(f"❌ Monte Carlo setup failed: {results}")
        
        # Test 4: LSTM model
        print("\n4. Testing LSTM model...")
        success, results = agent.train_model("LSTM", test_size=0.2, n_estimators=50)
        if success:
            print(f"✅ LSTM training successful")
            print(f"   R² Score: {results['r2']:.4f}")
            print(f"   RMSE: {results['rmse']:.4f}")
            
            # Test prediction
            success, pred_results = agent.predict_future(10)
            if success:
                print(f"✅ LSTM prediction successful")
                print(f"   Predicted {len(pred_results['forecast_prices'])} future prices")
            else:
                print(f"❌ LSTM prediction failed: {pred_results}")
        else:
            print(f"❌ LSTM training failed: {results}")
        
        print("\n🎉 All tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent()
    sys.exit(0 if success else 1) 