import google.generativeai as genai
import json
from datetime import datetime


class GeminiAnalyzer:
    def __init__(self, api_key):
        """
        Initialize Gemini analyzer with API key
        
        Args:
            api_key (str): Google Gemini API key
        """
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            print(f"Warning: Could not initialize Gemini model: {e}")
            self.model = None
    
    def analyze_news_sentiment(self, ticker, news_headlines, stock_data=None):
        """
        Analyze news sentiment and correlate with stock performance
        
        Args:
            ticker (str): Stock ticker symbol
            news_headlines (list): List of news headlines
            stock_data (dict): Optional stock performance data
            
        Returns:
            tuple: (success: bool, analysis: str)
        """
        if not self.model:
            return False, "Gemini model not initialized"
            
        try:
            # Create the prompt
            prompt = self._create_analysis_prompt(ticker, news_headlines, stock_data)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            if response.text:
                return True, response.text
            else:
                return False, "No response generated from Gemini"
                
        except Exception as e:
            return False, f"Error analyzing news: {str(e)}"
    
    def _create_analysis_prompt(self, ticker, news_headlines, stock_data=None):
        """Create a comprehensive prompt for news analysis"""
        
        # Format news headlines
        news_text = "\n".join([f"{i+1}. {headline}" for i, headline in enumerate(news_headlines)])
        
        # Add stock performance context if available
        stock_context = ""
        if stock_data:
            current_price = stock_data.get('current_price', 'N/A')
            price_change = stock_data.get('price_change', 'N/A')
            volume = stock_data.get('volume', 'N/A')
            
            stock_context = f"""
Current Stock Performance:
- Current Price: ${current_price}
- Price Change: {price_change}
- Volume: {volume}
"""
        
        prompt = f"""
You are a financial analyst specializing in stock market analysis. Please analyze the following news headlines for {ticker} and provide insights on:

1. **Overall Sentiment Analysis**: What is the general sentiment of the news (positive, negative, neutral, or mixed)?

2. **Key Themes**: What are the main themes or topics emerging from these headlines?

3. **Potential Impact**: How might these news items affect {ticker}'s stock price in the short term (next few days to weeks)?

4. **Risk Factors**: What potential risks or concerns are highlighted in the news?

5. **Opportunities**: What positive developments or opportunities are mentioned?

6. **Market Context**: How do these news items fit into the broader market context?

7. **Recommendation**: Based on this news analysis, what would be your brief recommendation for investors?

Please provide a concise but comprehensive analysis in 3-4 paragraphs. Focus on actionable insights and clear reasoning.

{stock_context}

Recent News Headlines for {ticker}:
{news_text}

Please analyze the above news and provide your insights.
"""
        
        return prompt
    
    def analyze_stock_performance_correlation(self, ticker, news_analysis, price_data):
        """
        Analyze correlation between news sentiment and stock price movements
        
        Args:
            ticker (str): Stock ticker symbol
            news_analysis (str): Previous news analysis
            price_data (dict): Historical price data
            
        Returns:
            tuple: (success: bool, correlation_analysis: str)
        """
        if not self.model:
            return False, "Gemini model not initialized"
            
        try:
            # Extract key price metrics
            if price_data and len(price_data) > 0:
                recent_prices = list(price_data.values())[-10:]  # Last 10 data points
                price_trend = "increasing" if recent_prices[-1] > recent_prices[0] else "decreasing"
                volatility = "high" if max(recent_prices) - min(recent_prices) > recent_prices[0] * 0.1 else "low"
            else:
                price_trend = "unknown"
                volatility = "unknown"
            
            prompt = f"""
Based on the previous news analysis for {ticker}, please analyze the correlation between news sentiment and recent stock price movements.

Previous News Analysis Summary:
{news_analysis[:500]}...

Recent Price Trend: {price_trend}
Price Volatility: {volatility}

Please provide insights on:
1. How well does the news sentiment align with the price trend?
2. Are there any discrepancies between news sentiment and price movement?
3. What factors might explain any misalignment?
4. What should investors watch for in the coming days?

Provide a brief 2-3 paragraph analysis focusing on the correlation between news and price action.
"""
            
            response = self.model.generate_content(prompt)
            
            if response.text:
                return True, response.text
            else:
                return False, "No correlation analysis generated"
                
        except Exception as e:
            return False, f"Error analyzing correlation: {str(e)}"
    
    def get_market_context(self, ticker, sector=None):
        """
        Get broader market context for the stock
        
        Args:
            ticker (str): Stock ticker symbol
            sector (str): Optional sector information
            
        Returns:
            tuple: (success: bool, context: str)
        """
        if not self.model:
            return False, "Gemini model not initialized"
            
        try:
            prompt = f"""
Provide a brief market context analysis for {ticker} stock.

Consider:
1. Current market conditions and trends
2. Sector-specific factors affecting {sector if sector else 'this stock'}
3. Macroeconomic factors that might impact the stock
4. Key competitors and industry dynamics

Provide a concise 2-3 paragraph analysis focusing on the broader market context that could affect {ticker}'s performance.
"""
            
            response = self.model.generate_content(prompt)
            
            if response.text:
                return True, response.text
            else:
                return False, "No market context generated"
                
        except Exception as e:
            return False, f"Error getting market context: {str(e)}" 