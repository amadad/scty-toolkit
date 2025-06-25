#!/usr/bin/env python3
import argparse
import os
from scraper import extract_urls_from_markdown, crawl_urls_parallel

def main():
    """
    Command-line interface for the Firecrawl scraper.
    """
    parser = argparse.ArgumentParser(description='Scrape URLs from a markdown file using Firecrawl.')
    
    parser.add_argument(
        '--input', '-i',
        default='list.md',
        help='Path to the markdown file containing URLs (default: list.md)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='scraped_content',
        help='Directory to save scraped content (default: scraped_content)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=2,  # More conservative default for rate limits
        help='Number of parallel workers (default: 2)'
    )
    
    parser.add_argument(
        '--delay', '-d',
        type=int,
        default=30,  # More conservative delay
        help='Delay between URL crawls in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Ensure input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return 1
    
    # Extract URLs from the markdown file
    print(f"Extracting URLs from {args.input}...")
    urls = extract_urls_from_markdown(args.input)
    
    if not urls:
        print(f"No URLs found in {args.input}")
        return 1
    
    # Calculate estimated time
    total_urls = len(urls)
    estimated_minutes = (total_urls * 20) / 60  # 20 seconds between each URL
    
    print(f"Found {total_urls} URLs. Starting to crawl with {args.workers} worker(s)...")
    print(f"Using {args.delay} seconds delay between crawls (Hobby plan: 3 crawls/minute)")
    print(f"Estimated time to complete: {estimated_minutes:.1f} minutes")
    
    # Crawl URLs in parallel with the specified delay
    results = crawl_urls_parallel(
        urls, 
        args.output, 
        max_workers=args.workers,
        delay_between_urls=args.delay
    )
    
    # Print summary
    successful = sum(1 for _, _, success in results if success)
    print(f"\nSummary:")
    print(f"- Successfully crawled: {successful}/{total_urls} URLs")
    print(f"- Output directory: {os.path.abspath(args.output)}")
    
    # List failed URLs if any
    failed = [(title, url) for title, url, success in results if not success]
    if failed:
        print("\nFailed to crawl the following URLs:")
        for title, url in failed:
            print(f"- {title}: {url}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 