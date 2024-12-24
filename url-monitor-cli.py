import asyncio
import click
import logging
from typing import Set
from datetime import datetime, timedelta
import json
import sys
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from monitor import URLMonitor
from analytics import URLAnalytics

console = Console()

def setup_logging():
    """Configure logging with appropriate formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('url_monitor.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

class MonitoringCLI:
    def __init__(self):
        self.monitor = URLMonitor()
        self.analytics = URLAnalytics()
        setup_logging()
        self.logger = logging.getLogger(__name__)

    async def initialize_monitoring(self, urls: Set[str]):
        """Initialize URL monitoring with progress bar."""
        with Progress() as progress:
            task = progress.add_task("[cyan]Initializing monitoring...", total=len(urls))
            
            async def wrapped_add_urls():
                await self.monitor.add_urls(urls)
                progress.update(task, advance=1)
            
            await wrapped_add_urls()

    def display_metrics_table(self, url: str):
        """Display metrics for a URL in a formatted table."""
        if url not in self.monitor.history:
            console.print(f"[red]No data available for {url}[/red]")
            return

        snapshots = self.monitor.history[url]
        metrics = self.analytics.get_url_metrics(url, snapshots)

        table = Table(title=f"Metrics for {url}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Snapshots", str(len(snapshots)))
        table.add_row("Changes per Day", f"{metrics.change_frequency:.2f}")
        table.add_row("Average Sentiment", f"{metrics.avg_sentiment:.2f}")
        table.add_row("Content Stability", f"{metrics.content_stability:.2%}")
        
        if metrics.response_times:
            avg_response = sum(metrics.response_times) / len(metrics.response_times)
            table.add_row("Avg Response Time", f"{avg_response:.2f}s")

        status_summary = ", ".join(f"{k}: {v}" for k, v in metrics.status_codes.items())
        table.add_row("Status Codes", status_summary)

        console.print(table)

    def display_changes_table(self, url: str):
        """Display content changes in a formatted table."""
        changes = self.monitor.get_changes(url)
        
        if not changes:
            console.print(f"[yellow]No significant changes detected for {url}[/yellow]")
            return

        table = Table(title=f"Content Changes for {url}")
        table.add_column("Timestamp", style="cyan")
        table.add_column("Similarity Score", style="green")

        for timestamp, similarity in changes:
            table.add_row(
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                f"{similarity:.2%}"
            )

        console.print(table)

@click.group()
def cli():
    """URL Monitoring and Analysis Tool"""
    pass

@cli.command()
@click.argument('urls', nargs=-1, required=True)
@click.option('--interval', default=3600, help='Monitoring interval in seconds')
def monitor(urls: tuple, interval: int):
    """Start monitoring specified URLs."""
    cli_handler = MonitoringCLI()
    url_set = set(urls)

    async def run_monitoring():
        await cli_handler.initialize_monitoring(url_set)
        console.print("[green]Monitoring initialized successfully![/green]")
        
        try:
            await cli_handler.monitor.monitor(interval=interval)
        except KeyboardInterrupt:
            console.print("[yellow]Monitoring stopped by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Error during monitoring: {str(e)}[/red]")

    asyncio.run(run_monitoring())

@cli.command()
@click.argument('url')
@click.option('--days', default=7, help='Number of days of history to analyze')
def analyze(url: str, days: int):
    """Analyze monitoring data for a specific URL."""
    cli_handler = MonitoringCLI()
    
    if url not in cli_handler.monitor.history:
        console.print(f"[red]No monitoring data found for {url}[/red]")
        return

    cutoff_date = datetime.now() - timedelta(days=days)
    relevant_snapshots = [
        snapshot for snapshot in cli_handler.monitor.