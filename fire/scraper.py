import os
import re
import time
import concurrent.futures
from urllib.parse import urlparse
from pathlib import Path
import logging
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Firecrawl
app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

def extract_urls_from_markdown(markdown_file):
    """
    Extract all URLs from a markdown file.
    Returns a list of tuples containing (title, url)
    """
    urls = []
    with open(markdown_file, 'r') as file:
        content = file.read()
        # Find all markdown links [title](url)
        matches = re.findall(r'\[(.*?)\]\((https?://[^\s\)]+)\)', content)
        urls.extend(matches)
    
    logger.info(f"Extracted {len(urls)} URLs from {markdown_file}")
    return urls

def generate_filename(title, url):
    """
    Generate a filename based on the title and URL.
    Returns a sanitized filename with .md extension
    """
    # Parse the domain from the URL
    domain = urlparse(url).netloc
    
    # Create a base for the filename using the title
    if title:
        # Sanitize the title for use as a filename
        base_name = re.sub(r'[^\w\s-]', '', title).strip().lower()
        base_name = re.sub(r'[\s]+', '_', base_name)
    else:
        # If no title, use the domain
        base_name = domain.replace('.', '_')
    
    # Combine domain and title for a more unique filename
    filename = f"{domain.split('.')[0]}_{base_name}.md"
    
    return filename

def crawl_url_with_retry(title, url, output_dir, max_retries=3, retry_delay=20):
    """
    Crawl a URL using Firecrawl with retry logic.
    Saves the result as a markdown file in the output directory.
    """
    filename = generate_filename(title, url)
    output_path = os.path.join(output_dir, filename)
    
    # Check if file already exists (to avoid duplicate work)
    if os.path.exists(output_path):
        logger.info(f"File already exists for {url}, skipping: {output_path}")
        return True
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Crawling URL: {url} (Attempt {attempt+1}/{max_retries})")
            
            # Call Firecrawl API with v1 parameters
            crawl_result = app.scrape_url(
                url,
                params={
                    'formats': ['markdown'],
                    'location': {
                        'country': 'US',
                        'languages': ['en-US']
                    }
                }
            )
            
            # Check if the crawl was successful
            if crawl_result and 'markdown' in crawl_result:
                # Create the output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Write the markdown content to a file
                with open(output_path, 'w') as file:
                    # Add metadata at the top of the file
                    file.write(f"# {title}\n\n")
                    file.write(f"Source: {url}\n\n")
                    file.write(f"Crawled on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    file.write("---\n\n")
                    
                    # Write the crawled content
                    file.write(crawl_result['markdown'])
                
                logger.info(f"Successfully crawled and saved: {output_path}")
                return True
            else:
                logger.warning(f"Crawl completed but no data returned for {url}")
                
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            if attempt < max_retries - 1:
                # If it's a rate limit error, wait the specified time from the error message
                if "429" in str(e):
                    # Extract wait time from error message if available
                    try:
                        import re
                        wait_time = int(re.search(r'retry after (\d+)s', str(e)).group(1))
                        logger.info(f"Rate limit hit, waiting {wait_time} seconds as specified by API...")
                        time.sleep(wait_time)
                    except:
                        logger.info(f"Rate limit hit, waiting {retry_delay} seconds...")
                        time.sleep(retry_delay)
                else:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            else:
                logger.error(f"Failed to crawl {url} after {max_retries} attempts")
                return False
    
    return False

def crawl_urls_parallel(urls, output_dir, max_workers=2, delay_between_urls=30):
    """
    Crawl multiple URLs in parallel using a thread pool.
    Args:
        urls: List of (title, url) tuples to crawl
        output_dir: Directory to save output files
        max_workers: Number of parallel workers (default: 2 for safety)
        delay_between_urls: Delay in seconds between starting new URL crawls (default: 30)
    """
    results = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process URLs with parallel workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit crawling tasks for each URL
        future_to_url = {}
        for title, url in urls:
            # Add delay between submitting tasks
            if future_to_url:  # If not the first URL
                time.sleep(delay_between_urls)
            future = executor.submit(crawl_url_with_retry, title, url, output_dir)
            future_to_url[future] = (title, url)
            logger.info(f"Submitted task for URL: {url}")
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            title, url = future_to_url[future]
            try:
                success = future.result()
                results.append((title, url, success))
                logger.info(f"Completed task for URL: {url} (Success: {success})")
            except Exception as e:
                logger.error(f"Exception occurred while processing {url}: {str(e)}")
                results.append((title, url, False))
    
    return results

def main():
    """
    Main function to extract URLs from markdown and crawl them.
    """
    # Define input and output paths
    input_file = "list.md"
    output_dir = "scraped_content"
    
    # Extract URLs from the markdown file
    urls = extract_urls_from_markdown(input_file)
    
    if not urls:
        logger.error(f"No URLs found in {input_file}")
        return
    
    logger.info(f"Starting to crawl {len(urls)} URLs in parallel")
    
    # Crawl URLs in parallel
    start_time = time.time()
    results = crawl_urls_parallel(urls, output_dir)
    end_time = time.time()
    
    # Generate summary
    successful = sum(1 for _, _, success in results if success)
    logger.info(f"Crawling completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Successfully crawled {successful} out of {len(urls)} URLs")
    
    # List failed URLs
    failed = [(title, url) for title, url, success in results if not success]
    if failed:
        logger.warning("Failed to crawl the following URLs:")
        for title, url in failed:
            logger.warning(f"- {title}: {url}")

if __name__ == "__main__":
    main() 