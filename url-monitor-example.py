import asyncio
from analytics import URLAnalytics
from monitor import URLMonitor
from rich.console import Console

async def main():
    console = Console()
    monitor = URLMonitor(concurrency_limit=3)
    analytics = URLAnalytics()

    # Start monitoring some URLs
    urls = {
        "planetterp.com"
    }
    
    console.print("[cyan]Initializing monitoring...[/cyan]")
    await monitor.add_urls(urls)

    # Run monitor for 1 hour, checking every 10 minutes
    console.print("[cyan]Starting monitoring loop...[/cyan]")
    monitoring_task = asyncio.create_task(monitor.monitor(interval=200))
    
    try:
        await asyncio.sleep(3600)  # Run for 1 hour
        monitoring_task.cancel()
    except asyncio.CancelledError:
        pass

    # Display results
    for url in urls:
        console.print(f"\n[green]Analysis for {url}:[/green]")
        metrics = analytics.get_url_metrics(url, monitor.history[url])
        console.print(f"Changes per day: {metrics.change_frequency:.2f}")
        console.print(f"Average sentiment: {metrics.avg_sentiment:.2f}")
        console.print(f"Content stability: {metrics.content_stability:.2%}")

if __name__ == "__main__":
    asyncio.run(main())