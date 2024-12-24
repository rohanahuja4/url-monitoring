from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from monitor import PageSnapshot

@dataclass
class URLMetrics:
    change_frequency: float
    avg_sentiment: float
    sentiment_volatility: float
    response_times: List[float]
    status_codes: Dict[int, int]
    content_stability: float
    
class URLAnalytics:
    def __init__(self):
        self.metrics_cache: Dict[str, Tuple[datetime, URLMetrics]] = {}
        self.cache_duration = timedelta(hours=1)
        
    def get_url_metrics(self, url: str, snapshots: List[PageSnapshot]) -> URLMetrics:
        """Retrieve or calculate metrics for a URL."""
        if url in self.metrics_cache:
            timestamp, metrics = self.metrics_cache[url]
            if datetime.now() - timestamp < self.cache_duration:
                return metrics
        
        metrics = self.analyze_url(snapshots)
        self.metrics_cache[url] = (datetime.now(), metrics)
        return metrics

    def analyze_url(self, snapshots: List[PageSnapshot]) -> URLMetrics:
        """Generate comprehensive metrics for a URL based on its history."""
        if not snapshots:
            raise ValueError("No snapshots provided for analysis")

        total_days = (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 86400
        changes = sum(1 for i in range(1, len(snapshots))
                    if snapshots[i].hash != snapshots[i-1].hash)
        change_frequency = changes / total_days if total_days > 0 else 0

        sentiment_scores = [s.sentiment_scores['compound'] for s in snapshots]
        avg_sentiment = np.mean(sentiment_scores)
        sentiment_volatility = np.std(sentiment_scores)

        response_times = []
        status_codes = defaultdict(int)
        
        for snapshot in snapshots:
            status_codes[snapshot.status_code] += 1
            if 'X-Response-Time' in snapshot.headers:
                try:
                    response_times.append(float(snapshot.headers['X-Response-Time']))
                except (ValueError, TypeError):
                    pass

        if len(snapshots) > 1:
            stability_scores = []
            for i in range(1, len(snapshots)):
                prev_len = len(snapshots[i-1].text_content)
                curr_len = len(snapshots[i].text_content)
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