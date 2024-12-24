from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from monitor import PageSnapshot

@dataclass
class URLMetrics:
    change_frequency: float  # Average changes per day
    avg_sentiment: float    # Average sentiment score
    sentiment_volatility: float  # Standard deviation of sentiment
    response_times: List[float]  # List of all response times
    status_codes: Dict[int, int]  # Count of each status code
    content_stability: float  # Measure of content stability (0-1)

class URLAnalytics:
    def __init__(self):
        self.metrics_cache: Dict[str, Tuple[datetime, URLMetrics]] = {}
        self.cache_duration = timedelta(hours=1)

    def analyze_url(self, snapshots: List[PageSnapshot]) -> URLMetrics:
        """Generate comprehensive metrics for a URL based on its history."""
        if not snapshots:
            raise ValueError("No snapshots provided for analysis")

        # Calculate time-based metrics
        total_days = (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 86400
        changes = sum(1 for i in range(1, len(snapshots))
                     if snapshots[i].hash != snapshots[i-1].hash)
        change_frequency = changes / total_days if total_days > 0 else 0

        # Sentiment analysis
        sentiment_scores = [s.sentiment_scores['compound'] for s in snapshots]
        avg_sentiment = np.mean(sentiment_scores)
        sentiment_volatility = np.std(sentiment_scores)

        # Response time analysis (assuming header information contains timing data)
        response_times = []
        status_codes = defaultdict(int)
        
        for snapshot in snapshots:
            status_codes[snapshot.status_code] += 1
            # Extract response time from headers if available
            if 'X-Response-Time' in snapshot.headers:
                try:
                    response_times.append(float(snapshot.headers['X-Response-Time']))
                except (ValueError, TypeError):
                    pass

        # Calculate content stability
        if len(snapshots) > 1:
            stability_scores = []
            for i in range(1, len(snapshots)):
                prev_len = len(snapshots[i-1].text_content)
                curr_len = len(snapshots[i].text_content)
                # Calculate relative size difference
                size_diff = abs(curr_len - prev_len) / max(prev_len, curr_len)
                stability_scores.append(1 - size_diff)
            content_stability = np.mean(stability_scores)
        else:
            content_stability = 1.0

        return URLMetrics(
            change_frequency=change_frequency,
            avg_sentiment=avg_sentiment,
            sentiment_volatility=sentiment_volatility,
            response_times=response_times,
            status_codes=dict(status_codes),
            content_stability=content_stability
        )

    def get_url_metrics(self, url: str, snapshots: List[PageSnapshot]) -> URLMetrics:
        """Get metrics for a URL, using cached values if available and fresh."""
        now = datetime.now()
        
        if url in self.metrics_cache:
            cache_time, metrics = self.metrics_cache[url]
            if now - cache_time < self.cache_duration:
                return metrics

        metrics = self.analyze_url(snapshots)
        self.metrics_cache[url] = (now, metrics)
        return metrics

    def get_comparative_analysis(self, snapshots_by_url: Dict[str, List[PageSnapshot]]) -> Dict[str, Dict[str, float]]:
        """Compare metrics across multiple URLs and generate relative scores."""
        if not snapshots_by_url:
            return {}

        # Calculate base metrics for all URLs
        base_metrics = {
            url: self.get_url_metrics(url, snapshots)
            for url, snapshots in snapshots_by_url.items()
        }

        # Calculate relative scores
        max_change_freq = max(m.change_frequency for m in base_metrics.values())
        max_volatility = max(m.sentiment_volatility for m in base_metrics.values())

        comparative_scores = {}
        for url, metrics in base_metrics.items():
            # Calculate normalized scores (0-1 scale)
            change_freq_score = metrics.change_frequency / max_change_freq if max_change_freq > 0 else 0
            volatility_score = metrics.sentiment_volatility / max_volatility if max_volatility > 0 else 0
            
            avg_response_time = np.mean(metrics.response_times) if metrics.response_times else 0
            success_rate = metrics.status_codes.get(200, 0) / sum(metrics.status_codes.values())

            comparative_scores[url] = {
                'dynamism_score': change_freq_score,
                'reliability_score': success_rate,
                'stability_score': metrics.content_stability,
                'sentiment_stability': 1 - volatility_score,
                'performance_score': 1 / (1 + avg_response_time) if avg_response_time > 0 else 1
            }

        return comparative_scores
