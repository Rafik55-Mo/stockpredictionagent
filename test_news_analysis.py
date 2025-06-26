#!/usr/bin/env python3
"""
Test script to verify news analysis functionality
"""

import sys
import traceback
from stock_agent import StockPredictionAgent

def test_news_analysis():
    """Test the news analysis functionality"""
    print("Testing News Analysis with Gemini...")
    
    # Create agent with Gemini API key
    gemini_api_key = "AIzaSyCThrV8VdcZ4TEDTAVNKKq4hOMLVY3jErE"
    agent = StockPredictionAgent(gemini_api_key)
    
    try:
        # Test 1: Fetch stock data first
        print("\n1. Fetching stock data...")
        success, message = agent.fetch_stock_data("AAPL", "1y")
        if success:
            print(f"✅ Data fetch successful: {message}")
        else:
            print(f"❌ Data fetch failed: {message}")
            return False
        
        # Test 2: News analysis
        print("\n2. Testing news sentiment analysis...")
        success, result = agent.analyze_news_sentiment()
        
        if success:
            print("✅ News analysis successful!")
            print(f"   Analysis date: {result.get('analysis_date', 'N/A')}")
            print(f"   News headlines: {len(result.get('news_headlines', []))}")
            print(f"   Sentiment analysis length: {len(result.get('sentiment_analysis', ''))}")
            print(f"   Market context length: {len(result.get('market_context', ''))}")
            
            # Display a sample of the analysis
            print("\nSample Sentiment Analysis:")
            sentiment = result.get('sentiment_analysis', '')
            print(sentiment[:300] + "..." if len(sentiment) > 300 else sentiment)
            
        else:
            print(f"❌ News analysis failed: {result}")
            return False
        
        # Test 3: Correlation analysis
        print("\n3. Testing correlation analysis...")
        sentiment_analysis = result.get('sentiment_analysis', '')
        if sentiment_analysis:
            success, correlation = agent.get_news_correlation_analysis(sentiment_analysis)
            
            if success:
                print("✅ Correlation analysis successful!")
                print(f"   Correlation analysis length: {len(correlation)}")
                
                # Display a sample of the correlation analysis
                print("\nSample Correlation Analysis:")
                print(correlation[:300] + "..." if len(correlation) > 300 else correlation)
            else:
                print(f"❌ Correlation analysis failed: {correlation}")
        else:
            print("⚠️  Skipping correlation analysis - no sentiment analysis available")
        
        print("\n🎉 News analysis tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_news_analysis()
    sys.exit(0 if success else 1) 