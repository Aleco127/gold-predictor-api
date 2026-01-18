"""
Sentiment Analyzer Module
=========================
FinBERT-based sentiment analysis for financial news headlines.
Provides sentiment scores for gold trading signals.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
import logging
import hashlib

logger = logging.getLogger(__name__)

# Lazy import for transformers to avoid slow startup if not needed
_pipeline = None
_tokenizer = None


def _load_finbert():
    """Lazy load FinBERT model and tokenizer."""
    global _pipeline, _tokenizer
    if _pipeline is None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            model_name = "ProsusAI/finbert"
            logger.info(f"Loading FinBERT model: {model_name}")

            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=_tokenizer,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise
    return _pipeline


@dataclass
class SentimentScore:
    """Sentiment analysis result for a single headline."""

    headline: str
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float  # 0.0 to 1.0
    score: float  # -1.0 (bearish) to +1.0 (bullish)
    source: Optional[str] = None
    published_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "headline": self.headline,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "score": self.score,
            "source": self.source,
            "published_at": self.published_at.isoformat() if self.published_at else None,
        }


@dataclass
class AggregateSentiment:
    """Aggregate sentiment for a time period."""

    average_score: float  # -1.0 to +1.0
    weighted_score: float  # Confidence-weighted average
    positive_count: int
    negative_count: int
    neutral_count: int
    total_headlines: int
    sentiment_label: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float  # Average confidence
    period_hours: int
    calculated_at: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "average_score": self.average_score,
            "weighted_score": self.weighted_score,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "total_headlines": self.total_headlines,
            "sentiment_label": self.sentiment_label,
            "confidence": self.confidence,
            "period_hours": self.period_hours,
            "calculated_at": self.calculated_at.isoformat(),
        }


class SentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial news.

    Uses ProsusAI/finbert model to analyze sentiment of news headlines
    and calculate aggregate sentiment scores for trading signals.
    """

    # Score mapping for FinBERT labels
    SENTIMENT_SCORES = {
        "positive": 1.0,
        "negative": -1.0,
        "neutral": 0.0,
    }

    # Thresholds for aggregate sentiment labels
    BULLISH_THRESHOLD = 0.15
    BEARISH_THRESHOLD = -0.15

    def __init__(
        self,
        cache_duration_minutes: int = 60,
        lazy_load: bool = True,
    ):
        """
        Initialize sentiment analyzer.

        Args:
            cache_duration_minutes: How long to cache sentiment scores
            lazy_load: Whether to lazy load the model (recommended for faster startup)
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self._score_cache: dict[str, tuple[SentimentScore, datetime]] = {}
        self._aggregate_cache: Optional[tuple[AggregateSentiment, datetime]] = None
        self._model_loaded = False
        self._cache_lock = asyncio.Lock()

        if not lazy_load:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> bool:
        """Ensure FinBERT model is loaded."""
        if not self._model_loaded:
            try:
                _load_finbert()
                self._model_loaded = True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False
        return True

    @property
    def is_ready(self) -> bool:
        """Check if the analyzer is ready (model loaded)."""
        return self._model_loaded or self._ensure_model_loaded()

    def _get_cache_key(self, headline: str) -> str:
        """Generate cache key for a headline."""
        return hashlib.md5(headline.encode()).hexdigest()

    def _is_cache_valid(self, cached_at: datetime) -> bool:
        """Check if cached result is still valid."""
        age = datetime.now(timezone.utc) - cached_at
        return age < self.cache_duration

    def analyze_headline(
        self,
        headline: str,
        source: Optional[str] = None,
        published_at: Optional[datetime] = None,
    ) -> SentimentScore:
        """
        Analyze sentiment of a single headline.

        Args:
            headline: News headline text
            source: News source name
            published_at: Publication timestamp

        Returns:
            SentimentScore with sentiment label, confidence, and score
        """
        # Check cache first
        cache_key = self._get_cache_key(headline)
        if cache_key in self._score_cache:
            cached_score, cached_at = self._score_cache[cache_key]
            if self._is_cache_valid(cached_at):
                # Update source/published_at if provided
                if source:
                    cached_score.source = source
                if published_at:
                    cached_score.published_at = published_at
                return cached_score

        # Ensure model is loaded
        if not self._ensure_model_loaded():
            # Return neutral score if model fails to load
            return SentimentScore(
                headline=headline,
                sentiment="neutral",
                confidence=0.0,
                score=0.0,
                source=source,
                published_at=published_at,
            )

        # Run sentiment analysis
        try:
            pipeline = _load_finbert()
            result = pipeline(headline)[0]

            sentiment = result["label"].lower()
            confidence = result["score"]

            # Map sentiment to score
            base_score = self.SENTIMENT_SCORES.get(sentiment, 0.0)
            # Scale by confidence
            score = base_score * confidence

            sentiment_score = SentimentScore(
                headline=headline,
                sentiment=sentiment,
                confidence=confidence,
                score=score,
                source=source,
                published_at=published_at,
            )

            # Cache the result
            self._score_cache[cache_key] = (sentiment_score, datetime.now(timezone.utc))

            return sentiment_score

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return SentimentScore(
                headline=headline,
                sentiment="neutral",
                confidence=0.0,
                score=0.0,
                source=source,
                published_at=published_at,
            )

    def analyze_headlines(
        self,
        headlines: list[dict],
    ) -> list[SentimentScore]:
        """
        Analyze sentiment of multiple headlines.

        Args:
            headlines: List of dicts with 'title', optional 'source', 'published_at'

        Returns:
            List of SentimentScore results
        """
        results = []
        for item in headlines:
            title = item.get("title", "")
            if not title:
                continue

            source = item.get("source")
            published_str = item.get("published_at")
            published_at = None
            if published_str:
                try:
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            score = self.analyze_headline(title, source, published_at)
            results.append(score)

        return results

    def calculate_aggregate(
        self,
        scores: list[SentimentScore],
        period_hours: int = 4,
    ) -> AggregateSentiment:
        """
        Calculate aggregate sentiment from individual scores.

        Args:
            scores: List of SentimentScore objects
            period_hours: Time period for the aggregate

        Returns:
            AggregateSentiment with averages and counts
        """
        if not scores:
            return AggregateSentiment(
                average_score=0.0,
                weighted_score=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                total_headlines=0,
                sentiment_label="NEUTRAL",
                confidence=0.0,
                period_hours=period_hours,
                calculated_at=datetime.now(timezone.utc),
            )

        # Filter by time period
        cutoff = datetime.now(timezone.utc) - timedelta(hours=period_hours)
        recent_scores = [
            s for s in scores
            if s.published_at is None or s.published_at >= cutoff
        ]

        if not recent_scores:
            recent_scores = scores  # Use all if none are recent

        # Count sentiments
        positive_count = sum(1 for s in recent_scores if s.sentiment == "positive")
        negative_count = sum(1 for s in recent_scores if s.sentiment == "negative")
        neutral_count = sum(1 for s in recent_scores if s.sentiment == "neutral")

        # Calculate averages
        total_score = sum(s.score for s in recent_scores)
        average_score = total_score / len(recent_scores)

        # Confidence-weighted average
        total_weight = sum(s.confidence for s in recent_scores)
        if total_weight > 0:
            weighted_score = sum(s.score * s.confidence for s in recent_scores) / total_weight
        else:
            weighted_score = average_score

        # Average confidence
        avg_confidence = sum(s.confidence for s in recent_scores) / len(recent_scores)

        # Determine sentiment label
        if weighted_score >= self.BULLISH_THRESHOLD:
            sentiment_label = "BULLISH"
        elif weighted_score <= self.BEARISH_THRESHOLD:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"

        return AggregateSentiment(
            average_score=round(average_score, 4),
            weighted_score=round(weighted_score, 4),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            total_headlines=len(recent_scores),
            sentiment_label=sentiment_label,
            confidence=round(avg_confidence, 4),
            period_hours=period_hours,
            calculated_at=datetime.now(timezone.utc),
        )

    async def get_current_sentiment(
        self,
        news_fetcher,
        period_hours: int = 4,
        force_refresh: bool = False,
    ) -> AggregateSentiment:
        """
        Get current aggregate sentiment from news fetcher.

        Args:
            news_fetcher: NewsFetcher instance to get headlines from
            period_hours: Time period to analyze
            force_refresh: Force refresh even if cache is valid

        Returns:
            AggregateSentiment with current market sentiment
        """
        async with self._cache_lock:
            # Check aggregate cache
            if not force_refresh and self._aggregate_cache:
                cached_agg, cached_at = self._aggregate_cache
                if self._is_cache_valid(cached_at) and cached_agg.period_hours == period_hours:
                    return cached_agg

            # Get headlines from news fetcher
            headlines = await news_fetcher.get_headlines_for_sentiment(
                hours=period_hours,
                max_count=30,  # Limit for performance
            )

            # Analyze all headlines
            scores = self.analyze_headlines(headlines)

            # Calculate aggregate
            aggregate = self.calculate_aggregate(scores, period_hours)

            # Cache the result
            self._aggregate_cache = (aggregate, datetime.now(timezone.utc))

            return aggregate

    def get_status(self) -> dict:
        """Get analyzer status information."""
        return {
            "model_loaded": self._model_loaded,
            "model_name": "ProsusAI/finbert",
            "cached_scores": len(self._score_cache),
            "cache_duration_minutes": self.cache_duration.total_seconds() / 60,
            "has_aggregate_cache": self._aggregate_cache is not None,
            "bullish_threshold": self.BULLISH_THRESHOLD,
            "bearish_threshold": self.BEARISH_THRESHOLD,
        }

    def clear_cache(self) -> dict:
        """Clear all cached scores."""
        cleared_scores = len(self._score_cache)
        self._score_cache.clear()
        self._aggregate_cache = None
        return {
            "cleared_scores": cleared_scores,
            "cleared_aggregate": True,
        }
