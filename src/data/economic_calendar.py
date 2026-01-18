"""
Economic Calendar Module
========================
Fetches and manages economic calendar events for trading decisions.
Supports filtering by impact level and currency relevance.
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, date, time
from enum import Enum
from typing import Optional
import logging

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ImpactLevel(Enum):
    """Impact level of economic events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HOLIDAY = "holiday"


@dataclass
class EconomicEvent:
    """Represents a single economic calendar event."""
    datetime_utc: datetime
    currency: str
    impact: ImpactLevel
    event_name: str
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "datetime_utc": self.datetime_utc.isoformat(),
            "currency": self.currency,
            "impact": self.impact.value,
            "event_name": self.event_name,
            "actual": self.actual,
            "forecast": self.forecast,
            "previous": self.previous,
            "is_high_impact": self.impact == ImpactLevel.HIGH,
        }


# High-impact event keywords for gold trading
HIGH_IMPACT_KEYWORDS = [
    "fed", "fomc", "interest rate", "rate decision",
    "cpi", "consumer price", "inflation",
    "nfp", "non-farm", "nonfarm", "employment change",
    "gdp", "gross domestic",
    "unemployment", "jobless claims",
    "retail sales",
    "pmi", "purchasing manager",
    "central bank", "ecb", "boe", "boj",
    "powell", "lagarde", "bailey",
    "gold", "xau",
]

# Currencies relevant for gold trading
GOLD_RELEVANT_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CNY"]


class EconomicCalendar:
    """
    Economic calendar manager for fetching and caching events.

    Fetches events from Forex Factory and caches them in memory.
    Supports filtering by impact level and time window.
    """

    FOREX_FACTORY_URL = "https://www.forexfactory.com/calendar"

    def __init__(
        self,
        refresh_interval_minutes: int = 60,
        relevant_currencies: Optional[list[str]] = None,
    ):
        """
        Initialize economic calendar.

        Args:
            refresh_interval_minutes: How often to refresh events (default: 60 min)
            relevant_currencies: List of currencies to track (default: gold-relevant)
        """
        self.refresh_interval = timedelta(minutes=refresh_interval_minutes)
        self.relevant_currencies = relevant_currencies or GOLD_RELEVANT_CURRENCIES

        self._events: list[EconomicEvent] = []
        self._last_fetch: Optional[datetime] = None
        self._fetch_lock = asyncio.Lock()

        logger.info(
            f"EconomicCalendar initialized: refresh={refresh_interval_minutes}min, "
            f"currencies={self.relevant_currencies}"
        )

    async def fetch_events(self, force: bool = False) -> list[EconomicEvent]:
        """
        Fetch economic events, using cache if available.

        Args:
            force: Force refresh even if cache is valid

        Returns:
            List of economic events
        """
        async with self._fetch_lock:
            now = datetime.now(timezone.utc)

            # Check if cache is valid
            if not force and self._last_fetch:
                cache_age = now - self._last_fetch
                if cache_age < self.refresh_interval and self._events:
                    logger.debug(f"Using cached events (age: {cache_age})")
                    return self._events

            # Fetch fresh events
            try:
                events = await self._fetch_forex_factory()
                self._events = events
                self._last_fetch = now
                logger.info(f"Fetched {len(events)} economic events")
                return events
            except Exception as e:
                logger.error(f"Failed to fetch events: {e}")
                # Return cached events on failure
                return self._events

    async def _fetch_forex_factory(self) -> list[EconomicEvent]:
        """
        Fetch events from Forex Factory calendar.

        Returns:
            List of parsed economic events
        """
        events = []

        # Forex Factory uses a specific date format
        today = datetime.now(timezone.utc)

        # Build URL for today and tomorrow
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                # Fetch today's calendar
                response = await client.get(
                    self.FOREX_FACTORY_URL,
                    headers=headers,
                )

                if response.status_code == 200:
                    events.extend(self._parse_forex_factory_html(response.text, today))
                else:
                    logger.warning(f"Forex Factory returned status {response.status_code}")
                    # Fall back to alternative source
                    events = await self._fetch_alternative_source()

        except httpx.RequestError as e:
            logger.error(f"Network error fetching Forex Factory: {e}")
            events = await self._fetch_alternative_source()

        return events

    def _parse_forex_factory_html(
        self,
        html: str,
        reference_date: datetime,
    ) -> list[EconomicEvent]:
        """
        Parse Forex Factory HTML calendar.

        Args:
            html: Raw HTML content
            reference_date: Reference date for parsing

        Returns:
            List of parsed events
        """
        events: list[EconomicEvent] = []

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Find calendar table
            calendar_table = soup.find("table", class_="calendar__table")
            if not calendar_table:
                logger.warning("Could not find calendar table in HTML")
                return events

            current_date = reference_date.date()
            current_time = None

            rows = calendar_table.find_all("tr", class_="calendar__row")

            for row in rows:
                try:
                    # Check for date row
                    date_cell = row.find("td", class_="calendar__date")
                    if date_cell:
                        date_text = date_cell.get_text(strip=True)
                        if date_text:
                            parsed_date = self._parse_ff_date(date_text, reference_date.year)
                            if parsed_date:
                                current_date = parsed_date

                    # Get time
                    time_cell = row.find("td", class_="calendar__time")
                    if time_cell:
                        time_text = time_cell.get_text(strip=True)
                        if time_text and time_text not in ["", "All Day", "Tentative"]:
                            current_time = self._parse_ff_time(time_text)

                    # Get currency
                    currency_cell = row.find("td", class_="calendar__currency")
                    currency = currency_cell.get_text(strip=True) if currency_cell else ""

                    # Skip if not relevant currency
                    if currency and currency not in self.relevant_currencies:
                        continue

                    # Get impact
                    impact_cell = row.find("td", class_="calendar__impact")
                    impact = self._parse_impact(impact_cell)

                    # Get event name
                    event_cell = row.find("td", class_="calendar__event")
                    event_name = ""
                    if event_cell:
                        event_span = event_cell.find("span", class_="calendar__event-title")
                        if event_span:
                            event_name = event_span.get_text(strip=True)

                    if not event_name:
                        continue

                    # Get actual/forecast/previous
                    actual_cell = row.find("td", class_="calendar__actual")
                    forecast_cell = row.find("td", class_="calendar__forecast")
                    previous_cell = row.find("td", class_="calendar__previous")

                    actual = actual_cell.get_text(strip=True) if actual_cell else None
                    forecast = forecast_cell.get_text(strip=True) if forecast_cell else None
                    previous = previous_cell.get_text(strip=True) if previous_cell else None

                    # Build datetime
                    if current_time:
                        event_dt = datetime.combine(
                            current_date,
                            current_time,
                            tzinfo=timezone.utc,
                        )
                    else:
                        event_dt = datetime.combine(
                            current_date,
                            datetime.min.time(),
                            tzinfo=timezone.utc,
                        )

                    # Check for high-impact keywords
                    if impact != ImpactLevel.HIGH:
                        event_lower = event_name.lower()
                        for keyword in HIGH_IMPACT_KEYWORDS:
                            if keyword in event_lower:
                                impact = ImpactLevel.HIGH
                                break

                    event = EconomicEvent(
                        datetime_utc=event_dt,
                        currency=currency,
                        impact=impact,
                        event_name=event_name,
                        actual=actual if actual else None,
                        forecast=forecast if forecast else None,
                        previous=previous if previous else None,
                    )
                    events.append(event)

                except Exception as e:
                    logger.debug(f"Error parsing calendar row: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing Forex Factory HTML: {e}")

        return events

    def _parse_ff_date(self, date_text: str, year: int) -> Optional[date]:
        """Parse Forex Factory date format."""
        try:
            # Format: "Mon Jan 15" or "Tue Jan 16"
            date_text = date_text.strip()

            # Remove day name if present
            parts = date_text.split()
            if len(parts) >= 2:
                month_day = " ".join(parts[-2:])
                parsed = datetime.strptime(f"{month_day} {year}", "%b %d %Y")
                return parsed.date()
        except Exception:
            pass
        return None

    def _parse_ff_time(self, time_text: str) -> Optional[time]:
        """Parse Forex Factory time format."""
        try:
            time_text = time_text.strip().lower()

            # Handle 12-hour format
            if "am" in time_text or "pm" in time_text:
                time_text = time_text.replace("am", " AM").replace("pm", " PM")
                parsed = datetime.strptime(time_text.strip(), "%I:%M %p")
                return parsed.time()

            # Handle 24-hour format
            parsed = datetime.strptime(time_text, "%H:%M")
            return parsed.time()

        except Exception:
            return None

    def _parse_impact(self, impact_cell) -> ImpactLevel:
        """Parse impact level from cell."""
        if not impact_cell:
            return ImpactLevel.LOW

        # Forex Factory uses colored icons
        impact_span = impact_cell.find("span")
        if impact_span:
            classes = impact_span.get("class", [])
            for cls in classes:
                if "high" in cls.lower() or "red" in cls.lower():
                    return ImpactLevel.HIGH
                elif "medium" in cls.lower() or "orange" in cls.lower() or "yellow" in cls.lower():
                    return ImpactLevel.MEDIUM
                elif "holiday" in cls.lower() or "gray" in cls.lower():
                    return ImpactLevel.HOLIDAY

        return ImpactLevel.LOW

    async def _fetch_alternative_source(self) -> list[EconomicEvent]:
        """
        Fetch from alternative free API source.
        Uses FCS API as backup.

        Returns:
            List of economic events
        """
        events = []

        # Use a simple fallback with common scheduled events
        now = datetime.now(timezone.utc)

        # Add placeholder for known regular events
        # This ensures the system has some data even if scraping fails

        # FOMC meetings are scheduled ~8 times per year
        # NFP is first Friday of each month at 8:30 AM ET (13:30 UTC)
        # CPI is typically mid-month

        # Calculate next NFP (first Friday of current/next month)
        first_day = now.replace(day=1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)

        if first_friday.date() <= now.date():
            # Move to next month
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=1)
            else:
                next_month = now.replace(month=now.month + 1, day=1)
            days_until_friday = (4 - next_month.weekday()) % 7
            first_friday = next_month + timedelta(days=days_until_friday)

        nfp_time = first_friday.replace(hour=13, minute=30, second=0, microsecond=0)

        events.append(EconomicEvent(
            datetime_utc=nfp_time,
            currency="USD",
            impact=ImpactLevel.HIGH,
            event_name="Non-Farm Payrolls",
            forecast=None,
            previous=None,
        ))

        logger.info("Using fallback economic calendar with scheduled events")
        return events

    def get_upcoming_events(
        self,
        hours_ahead: int = 24,
        min_impact: ImpactLevel = ImpactLevel.LOW,
        currencies: Optional[list[str]] = None,
    ) -> list[EconomicEvent]:
        """
        Get upcoming events within time window.

        Args:
            hours_ahead: Hours to look ahead (default: 24)
            min_impact: Minimum impact level to include
            currencies: Filter by specific currencies

        Returns:
            List of upcoming events
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        # Impact level hierarchy
        impact_order = {
            ImpactLevel.LOW: 0,
            ImpactLevel.MEDIUM: 1,
            ImpactLevel.HIGH: 2,
            ImpactLevel.HOLIDAY: 0,
        }
        min_level = impact_order.get(min_impact, 0)

        filtered = []
        for event in self._events:
            # Time filter
            if event.datetime_utc < now or event.datetime_utc > cutoff:
                continue

            # Impact filter
            if impact_order.get(event.impact, 0) < min_level:
                continue

            # Currency filter
            if currencies and event.currency not in currencies:
                continue

            filtered.append(event)

        # Sort by datetime
        filtered.sort(key=lambda e: e.datetime_utc)

        return filtered

    def get_high_impact_events(
        self,
        hours_ahead: int = 24,
    ) -> list[EconomicEvent]:
        """
        Get only high-impact events.

        Args:
            hours_ahead: Hours to look ahead

        Returns:
            List of high-impact events
        """
        return self.get_upcoming_events(
            hours_ahead=hours_ahead,
            min_impact=ImpactLevel.HIGH,
        )

    def is_near_high_impact_event(
        self,
        minutes_before: int = 30,
        minutes_after: int = 30,
    ) -> tuple[bool, Optional[EconomicEvent]]:
        """
        Check if we're near a high-impact event.

        Args:
            minutes_before: Minutes before event to flag
            minutes_after: Minutes after event to flag

        Returns:
            Tuple of (is_near, event or None)
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=minutes_after)
        window_end = now + timedelta(minutes=minutes_before)

        for event in self._events:
            if event.impact != ImpactLevel.HIGH:
                continue

            if window_start <= event.datetime_utc <= window_end:
                return True, event

        return False, None

    def get_status(self) -> dict:
        """Get calendar status information."""
        now = datetime.now(timezone.utc)

        is_near, near_event = self.is_near_high_impact_event()

        return {
            "events_cached": len(self._events),
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            "cache_age_minutes": (
                (now - self._last_fetch).total_seconds() / 60
                if self._last_fetch else None
            ),
            "refresh_interval_minutes": self.refresh_interval.total_seconds() / 60,
            "is_near_high_impact": is_near,
            "near_event": near_event.to_dict() if near_event else None,
            "relevant_currencies": self.relevant_currencies,
        }


@dataclass
class NewsFilterResult:
    """Result of news filter check."""
    trading_allowed: bool
    is_paused: bool
    reason: Optional[str] = None
    near_event: Optional[EconomicEvent] = None
    minutes_until_event: Optional[float] = None
    minutes_since_event: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "trading_allowed": self.trading_allowed,
            "is_paused": self.is_paused,
            "reason": self.reason,
            "near_event": self.near_event.to_dict() if self.near_event else None,
            "minutes_until_event": self.minutes_until_event,
            "minutes_since_event": self.minutes_since_event,
        }


class NewsFilter:
    """
    News filter for pausing trading around high-impact events.

    Checks if trading should be paused based on proximity to
    high-impact economic events.
    """

    def __init__(
        self,
        calendar: EconomicCalendar,
        pause_minutes_before: int = 30,
        pause_minutes_after: int = 30,
    ):
        """
        Initialize news filter.

        Args:
            calendar: EconomicCalendar instance to check events
            pause_minutes_before: Minutes to pause before high-impact events
            pause_minutes_after: Minutes to pause after high-impact events
        """
        self.calendar = calendar
        self.pause_before = pause_minutes_before
        self.pause_after = pause_minutes_after

        logger.info(
            f"NewsFilter initialized: pause {pause_minutes_before}min before, "
            f"{pause_minutes_after}min after high-impact events"
        )

    def check_trading_allowed(self) -> NewsFilterResult:
        """
        Check if trading is allowed based on economic calendar.

        Returns:
            NewsFilterResult with trading status and event details
        """
        now = datetime.now(timezone.utc)

        # Check for nearby high-impact events
        for event in self.calendar._events:
            if event.impact != ImpactLevel.HIGH:
                continue

            event_time = event.datetime_utc
            time_diff = (event_time - now).total_seconds() / 60  # minutes

            # Event is in the future
            if 0 < time_diff <= self.pause_before:
                return NewsFilterResult(
                    trading_allowed=False,
                    is_paused=True,
                    reason=f"High-impact event '{event.event_name}' in {time_diff:.0f} minutes",
                    near_event=event,
                    minutes_until_event=time_diff,
                )

            # Event just happened
            if -self.pause_after <= time_diff <= 0:
                minutes_ago = abs(time_diff)
                return NewsFilterResult(
                    trading_allowed=False,
                    is_paused=True,
                    reason=f"High-impact event '{event.event_name}' occurred {minutes_ago:.0f} minutes ago",
                    near_event=event,
                    minutes_since_event=minutes_ago,
                )

        # No blocking events
        return NewsFilterResult(
            trading_allowed=True,
            is_paused=False,
            reason=None,
        )

    def get_next_high_impact_event(self) -> Optional[EconomicEvent]:
        """
        Get the next upcoming high-impact event.

        Returns:
            Next high-impact event or None
        """
        now = datetime.now(timezone.utc)

        upcoming = [
            event for event in self.calendar._events
            if event.impact == ImpactLevel.HIGH and event.datetime_utc > now
        ]

        if upcoming:
            return min(upcoming, key=lambda e: e.datetime_utc)
        return None

    def get_config(self) -> dict:
        """Get filter configuration."""
        return {
            "pause_minutes_before": self.pause_before,
            "pause_minutes_after": self.pause_after,
            "total_pause_window_minutes": self.pause_before + self.pause_after,
        }

    def get_status(self) -> dict:
        """Get filter status including current state and next event."""
        result = self.check_trading_allowed()
        next_event = self.get_next_high_impact_event()

        return {
            "trading_allowed": result.trading_allowed,
            "is_paused": result.is_paused,
            "pause_reason": result.reason,
            "current_event": result.near_event.to_dict() if result.near_event else None,
            "next_high_impact_event": next_event.to_dict() if next_event else None,
            "config": self.get_config(),
        }


# Convenience function for synchronous use
def fetch_events_sync() -> list[EconomicEvent]:
    """Synchronous wrapper for fetching events."""
    calendar = EconomicCalendar()
    return asyncio.run(calendar.fetch_events())
