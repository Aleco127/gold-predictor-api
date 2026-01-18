"""
News fetcher module for gold-related financial news.
Fetches news from NewsAPI to support sentiment analysis.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
import logging
import httpx

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a news article."""

    title: str
    description: Optional[str]
    source: str
    url: str
    published_at: datetime
    content: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "content": self.content,
        }


class NewsFetcher:
    """
    Fetches gold-related financial news from NewsAPI.

    Searches for news with keywords relevant to gold trading:
    - Gold, XAUUSD, precious metals
    - Fed, Federal Reserve, interest rates
    - Inflation, CPI, economic indicators
    - Dollar, USD strength
    """

    # Keywords for gold-related news
    GOLD_KEYWORDS = [
        "gold",
        "XAUUSD",
        "precious metals",
        "gold price",
        "gold market",
    ]

    MACRO_KEYWORDS = [
        "Federal Reserve",
        "Fed",
        "interest rate",
        "inflation",
        "CPI",
        "economic data",
        "dollar strength",
        "USD",
        "treasury yields",
    ]

    # NewsAPI base URL
    NEWSAPI_BASE_URL = "https://newsapi.org/v2"

    def __init__(
        self,
        api_key: str,
        cache_duration_minutes: int = 30,
        max_articles: int = 100,
    ):
        """
        Initialize news fetcher.

        Args:
            api_key: NewsAPI API key
            cache_duration_minutes: How long to cache articles
            max_articles: Maximum articles to store
        """
        self.api_key = api_key
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.max_articles = max_articles
        self._articles: list[NewsArticle] = []
        self._last_fetch: Optional[datetime] = None
        self._fetch_lock = asyncio.Lock()

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key and self.api_key != "your_newsapi_key")

    async def fetch_news(self, force_refresh: bool = False) -> list[NewsArticle]:
        """
        Fetch gold-related news from NewsAPI.

        Args:
            force_refresh: Force refresh even if cache is valid

        Returns:
            List of news articles
        """
        async with self._fetch_lock:
            # Check cache validity
            if not force_refresh and self._is_cache_valid():
                logger.debug("Returning cached news articles")
                return self._articles

            if not self.is_configured:
                logger.warning("NewsAPI key not configured, returning empty list")
                return []

            try:
                articles = await self._fetch_from_newsapi()
                self._articles = articles[: self.max_articles]
                self._last_fetch = datetime.now(timezone.utc)
                logger.info(f"Fetched {len(self._articles)} news articles")
                return self._articles
            except Exception as e:
                logger.error(f"Error fetching news: {e}")
                # Return cached articles if available
                return self._articles

    async def _fetch_from_newsapi(self) -> list[NewsArticle]:
        """Fetch articles from NewsAPI everything endpoint."""
        articles: list[NewsArticle] = []

        # Build query combining gold and macro keywords
        query = " OR ".join(self.GOLD_KEYWORDS[:3])  # Limit to avoid too long query

        # Calculate date range (last 24 hours for free tier, or 7 days for paid)
        from_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

        params: dict[str, str | int] = {
            "q": query,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": self.api_key,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch gold-specific news
            response = await client.get(
                f"{self.NEWSAPI_BASE_URL}/everything", params=params
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "ok":
                for article_data in data.get("articles", []):
                    article = self._parse_article(article_data)
                    if article:
                        articles.append(article)

            # Also fetch macro economic news
            macro_query = "Federal Reserve OR inflation OR interest rates"
            params["q"] = macro_query
            params["pageSize"] = 50

            response = await client.get(
                f"{self.NEWSAPI_BASE_URL}/everything", params=params
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "ok":
                for article_data in data.get("articles", []):
                    article = self._parse_article(article_data)
                    if article and article.url not in [a.url for a in articles]:
                        articles.append(article)

        # Sort by published date (newest first)
        articles.sort(key=lambda x: x.published_at, reverse=True)
        return articles

    def _parse_article(self, data: dict) -> Optional[NewsArticle]:
        """Parse NewsAPI article response into NewsArticle."""
        try:
            published_str = data.get("publishedAt", "")
            if published_str:
                # Parse ISO format datetime
                published_at = datetime.fromisoformat(
                    published_str.replace("Z", "+00:00")
                )
            else:
                published_at = datetime.now(timezone.utc)

            return NewsArticle(
                title=data.get("title", ""),
                description=data.get("description"),
                source=data.get("source", {}).get("name", "Unknown"),
                url=data.get("url", ""),
                published_at=published_at,
                content=data.get("content"),
            )
        except Exception as e:
            logger.warning(f"Error parsing article: {e}")
            return None

    def _is_cache_valid(self) -> bool:
        """Check if cached articles are still valid."""
        if not self._last_fetch or not self._articles:
            return False
        age = datetime.now(timezone.utc) - self._last_fetch
        return age < self.cache_duration

    def get_recent_headlines(
        self, hours: int = 4, max_count: int = 20
    ) -> list[NewsArticle]:
        """
        Get headlines from the last N hours.

        Args:
            hours: Number of hours to look back
            max_count: Maximum number of headlines to return

        Returns:
            List of recent news articles
        """
        if not self._articles:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = [a for a in self._articles if a.published_at >= cutoff]
        return recent[:max_count]

    def get_articles_by_keyword(
        self, keyword: str, max_count: int = 10
    ) -> list[NewsArticle]:
        """
        Filter articles by keyword in title or description.

        Args:
            keyword: Keyword to search for (case-insensitive)
            max_count: Maximum articles to return

        Returns:
            List of matching articles
        """
        keyword_lower = keyword.lower()
        matches = []
        for article in self._articles:
            title_match = keyword_lower in article.title.lower()
            desc_match = (
                article.description
                and keyword_lower in article.description.lower()
            )
            if title_match or desc_match:
                matches.append(article)
                if len(matches) >= max_count:
                    break
        return matches

    def get_status(self) -> dict:
        """Get news fetcher status information."""
        return {
            "configured": self.is_configured,
            "total_articles": len(self._articles),
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            "cache_valid": self._is_cache_valid(),
            "cache_duration_minutes": self.cache_duration.total_seconds() / 60,
        }

    async def get_headlines_for_sentiment(
        self, hours: int = 4, max_count: int = 20
    ) -> list[dict]:
        """
        Get headlines formatted for sentiment analysis.

        Args:
            hours: Hours to look back
            max_count: Maximum headlines

        Returns:
            List of dicts with title, source, and published_at
        """
        # Ensure we have fresh data
        await self.fetch_news()

        headlines = self.get_recent_headlines(hours=hours, max_count=max_count)
        return [
            {
                "title": h.title,
                "source": h.source,
                "published_at": h.published_at.isoformat(),
            }
            for h in headlines
        ]
