import asyncio
import hashlib
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from difflib import SequenceMatcher
import aiohttp
from bs4 import BeautifulSoup
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

@dataclass
class PageSnapshot:
    url: str
    timestamp: datetime
    content: str
    hash: str
    text_content: str
    sentiment_scores: Dict[str, float]
    headers: Dict[str, str]
    status_code: int

class URLMonitor:
    def __init__(self, concurrency_limit: int = 5):
        self.concurrency_limit = concurrency_limit
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.history: Dict[str, List[PageSnapshot]] = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_nltk()
        
    def _initialize_nltk(self):
        """Initialize NLTK resources."""
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    async def add_urls(self, urls: Set[str]):
        """Add new URLs to the monitoring list."""
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_and_analyze(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for url, result in zip(urls, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to add URL {url}: {str(result)}")
                    continue
                self.history[url] = [result]

    async def _fetch_and_analyze(self, session: aiohttp.ClientSession, url: str) -> PageSnapshot:
        """Fetch a URL and create a snapshot with analysis."""
        async with self.semaphore:
            async with session.get(url) as response:
                content = await response.text()
                headers = dict(response.headers)
                status_code = response.status
                
                # Create BeautifulSoup object for text extraction
                soup = BeautifulSoup(content, 'html.parser')
                text_content = ' '.join(soup.stripped_strings)
                
                # Calculate content hash
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                
                # Perform sentiment analysis
                sentiment_scores = self.sentiment_analyzer.polarity_scores(text_content)
                
                return PageSnapshot(
                    url=url,
                    timestamp=datetime.now(),
                    content=content,
                    hash=content_hash,
                    text_content=text_content,
                    sentiment_scores=sentiment_scores,
                    headers=headers,
                    status_code=status_code
                )

    def get_changes(self, url: str, threshold: float = 0.1) -> List[Tuple[datetime, float]]:
        """Get a list of timestamps and similarity scores for content changes."""
        if url not in self.history or len(self.history[url]) < 2:
            return []
        
        snapshots = self.history[url]
        changes = []
        
        for i in range(1, len(snapshots)):
            prev_snapshot = snapshots[i-1]
            curr_snapshot = snapshots[i]
            
            similarity = SequenceMatcher(
                None,
                prev_snapshot.text_content,
                curr_snapshot.text_content
            ).ratio()
            
            if abs(1 - similarity) > threshold:
                changes.append((curr_snapshot.timestamp, similarity))
                
        return changes

    def get_sentiment_trend(self, url: str) -> List[Tuple[datetime, float]]:
        """Get the sentiment trend over time for a URL."""
        if url not in self.history:
            return []
        
        return [(snapshot.timestamp, snapshot.sentiment_scores['compound'])
                for snapshot in self.history[url]]

    async def monitor(self, interval: int = 3600):
        """Continuously monitor all URLs at the specified interval."""
        while True:
            urls = set(self.history.keys())
            if not urls:
                await asyncio.sleep(interval)
                continue
                
            async with aiohttp.ClientSession() as session:
                tasks = [self._fetch_and_analyze(session, url) for url in urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for url, result in zip(urls, results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Failed to monitor {url}: {str(result)}")
                        continue
                    
                    # Only store if content has changed
                    if not self.history[url] or result.hash != self.history[url][-1].hash:
                        self.history[url].append(result)
            
            await asyncio.sleep(interval)
