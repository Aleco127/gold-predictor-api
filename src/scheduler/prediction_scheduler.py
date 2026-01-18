"""
Prediction Scheduler Module
===========================
Automated scheduling of predictions with configurable intervals.
"""

import asyncio
from datetime import datetime
from typing import Callable, Optional

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from ..config import settings


class PredictionScheduler:
    """
    Scheduler for automated prediction generation.

    Runs predictions at configured intervals and optionally
    sends alerts via webhook (e.g., to n8n).
    """

    def __init__(
        self,
        interval_minutes: int = 5,
        api_url: str = "http://localhost:8000",
        api_key: str = None,
        webhook_url: Optional[str] = None,
    ):
        """
        Initialize prediction scheduler.

        Args:
            interval_minutes: Minutes between predictions
            api_url: Gold Predictor API URL
            api_key: API authentication key
            webhook_url: Optional webhook URL for alerts (n8n, Telegram, etc.)
        """
        self.interval_minutes = interval_minutes
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key or settings.api_key
        self.webhook_url = webhook_url

        self.scheduler = AsyncIOScheduler()
        self._running = False
        self._prediction_count = 0
        self._alert_count = 0
        self._last_prediction = None
        self._callbacks = []

    def add_callback(self, callback: Callable) -> None:
        """
        Add callback function to be called on each prediction.

        Args:
            callback: Function taking prediction dict as argument
        """
        self._callbacks.append(callback)

    async def _fetch_prediction(self) -> Optional[dict]:
        """Fetch prediction from API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/predict",
                    headers={"X-API-Key": self.api_key},
                    timeout=30.0,
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"API request failed: {e}")
            return None

    async def _send_webhook(self, prediction: dict) -> bool:
        """
        Send prediction to webhook.

        Args:
            prediction: Prediction data

        Returns:
            True if successful
        """
        if not self.webhook_url:
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=prediction,
                    timeout=10.0,
                )
                response.raise_for_status()
                logger.info(f"Webhook sent successfully: {prediction['signal']}")
                return True

        except httpx.HTTPError as e:
            logger.error(f"Webhook failed: {e}")
            return False

    async def _run_prediction_job(self) -> None:
        """Execute prediction job."""
        logger.info("Running scheduled prediction...")

        prediction = await self._fetch_prediction()

        if prediction is None:
            logger.warning("Failed to get prediction")
            return

        self._prediction_count += 1
        self._last_prediction = prediction

        logger.info(
            f"Prediction #{self._prediction_count}: "
            f"{prediction['signal']} (confidence: {prediction['confidence']:.2%})"
        )

        # Run callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(prediction)
                else:
                    callback(prediction)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        # Send alert if needed
        if prediction.get("should_alert"):
            self._alert_count += 1
            await self._send_webhook(prediction)

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        # Add prediction job
        self.scheduler.add_job(
            self._run_prediction_job,
            trigger=IntervalTrigger(minutes=self.interval_minutes),
            id="prediction_job",
            name="Gold Price Prediction",
            replace_existing=True,
            next_run_time=datetime.now(),  # Run immediately
        )

        self.scheduler.start()
        self._running = True
        logger.info(f"Scheduler started: running every {self.interval_minutes} minutes")

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self.scheduler.shutdown(wait=False)
        self._running = False
        logger.info("Scheduler stopped")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    @property
    def stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "running": self._running,
            "interval_minutes": self.interval_minutes,
            "prediction_count": self._prediction_count,
            "alert_count": self._alert_count,
            "last_prediction": self._last_prediction,
        }


async def run_scheduler(
    interval_minutes: int = 5,
    api_url: str = "http://localhost:8000",
    api_key: str = None,
    webhook_url: str = None,
):
    """
    Run prediction scheduler as standalone process.

    Args:
        interval_minutes: Minutes between predictions
        api_url: Gold Predictor API URL
        api_key: API authentication key
        webhook_url: Webhook URL for alerts
    """
    scheduler = PredictionScheduler(
        interval_minutes=interval_minutes,
        api_url=api_url,
        api_key=api_key,
        webhook_url=webhook_url,
    )

    scheduler.start()

    try:
        # Keep running
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down scheduler...")
        scheduler.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gold Prediction Scheduler")
    parser.add_argument("--interval", type=int, default=5, help="Interval in minutes")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--webhook", help="Webhook URL for alerts")
    args = parser.parse_args()

    asyncio.run(run_scheduler(
        interval_minutes=args.interval,
        api_url=args.api_url,
        api_key=args.api_key,
        webhook_url=args.webhook,
    ))
