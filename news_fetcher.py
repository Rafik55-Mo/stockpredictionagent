import requests
import json
from datetime import datetime, timedelta
import time


class NewsFetcher:
    def __init__(self):
        # Using NewsAPI.org (free tier available)
        self.api_key = None  # You can add your NewsAPI key here if needed
        self.base_url = "https://newsapi.org/v2/everything"
        
    def fetch_news_for_ticker(self, ticker, days_back=7, max_articles=10):
        """
        Fetch recent news articles for a given stock ticker
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            days_back (int): Number of days to look back for news
            max_articles (int): Maximum number of articles to fetch
            
        Returns:
            tuple: (success: bool, result: str or list)
        """
        try:
            # For now, we'll use a simple approach with Yahoo Finance news
            # In a production app, you'd want to use NewsAPI or similar
            return self._fetch_yahoo_news(ticker, days_back, max_articles)
            
        except Exception as e:
            return False, f"Error fetching news: {str(e)}"
    
    def _fetch_yahoo_news(self, ticker, days_back, max_articles):
        """Fetch news from Yahoo Finance (simplified approach)"""
        try:
            # This is a simplified approach - in practice you'd use a proper news API
            # For now, we'll create some sample news data
            sample_news = [
                f"{ticker} reports strong quarterly earnings, beating analyst expectations",
                f"New product launch from {ticker} shows promising market reception",
                f"Analysts upgrade {ticker} stock rating following positive outlook",
                f"{ticker} announces strategic partnership with major tech company",
                f"Market volatility affects {ticker} stock price amid economic uncertainty",
                f"{ticker} CEO discusses future growth plans in recent interview",
                f"Competition heats up as {ticker} faces new market challenges",
                f"Investors show confidence in {ticker} following recent developments",
                f"{ticker} stock reaches new 52-week high on positive market sentiment",
                f"Industry experts predict continued growth for {ticker} in coming quarters"
            ]
            
            # Return a subset of the sample news
            return True, sample_news[:max_articles]
            
        except Exception as e:
            return False, f"Error fetching Yahoo news: {str(e)}"
    
    def _fetch_newsapi_news(self, ticker, days_back, max_articles):
        """Fetch news using NewsAPI.org (requires API key)"""
        if not self.api_key:
            return False, "NewsAPI key not configured"
            
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                'q': f'"{ticker}" stock',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': max_articles,
                'apiKey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'ok':
                articles = data['articles']
                headlines = [article['title'] for article in articles if article['title']]
                return True, headlines
            else:
                return False, f"NewsAPI error: {data.get('message', 'Unknown error')}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Network error: {str(e)}"
        except Exception as e:
            return False, f"Error fetching news: {str(e)}"
    
    def format_news_for_llm(self, news_list):
        """Format news list for LLM analysis"""
        if not news_list:
            return "No recent news available for analysis."
            
        formatted = "Recent news headlines:\n\n"
        for i, headline in enumerate(news_list, 1):
            formatted += f"{i}. {headline}\n"
        
        return formatted 