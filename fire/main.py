import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

# Import our custom scraper module
from scraper import extract_urls_from_markdown, crawl_urls_parallel

# Load environment variables
load_dotenv()

# Example 1: Basic Firecrawl usage
def example_basic_firecrawl():
    """
    Example of basic Firecrawl usage to crawl a single website.
    """
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    
    # Crawl a website:
    crawl_status = app.crawl_url(
        'https://firecrawl.dev', 
        params={
            'limit': 100, 
            'scrapeOptions': {'formats': ['markdown', 'html']}
        },
        poll_interval=30
    )
    
    print("Basic Firecrawl example result:")
    print(crawl_status)
    print("\n" + "-"*50 + "\n")

# Example 2: Using our custom scraper to process multiple URLs
def example_custom_scraper():
    """
    Example of using our custom scraper to process multiple URLs from a markdown file.
    """
    # Define input and output paths
    input_file = "fire/list.md"
    output_dir = "fire/scraped_content"
    
    # Extract URLs from the markdown file
    print(f"Extracting URLs from {input_file}...")
    urls = extract_urls_from_markdown(input_file)
    
    if not urls:
        print(f"No URLs found in {input_file}")
        return
    
    # Print the first 3 URLs as an example
    print(f"Found {len(urls)} URLs. Here are the first 3:")
    for i, (title, url) in enumerate(urls[:3]):
        print(f"{i+1}. {title}: {url}")
    
    print("\nTo crawl all these URLs, run:")
    print("python fire/run_scraper.py")
    print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    print("Firecrawl Examples\n" + "="*20 + "\n")
    
    # Run the basic Firecrawl example
    example_basic_firecrawl()
    
    # Run the custom scraper example
    example_custom_scraper()
    
    print("For more options, run: python fire/run_scraper.py --help")