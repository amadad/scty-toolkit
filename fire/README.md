# Firecrawl URL Scraper

This tool extracts URLs from a markdown file and uses Firecrawl to scrape their content, saving the results as individual markdown files.

## Features

- Extracts URLs from markdown files
- Crawls websites using Firecrawl API
- Saves content as markdown files
- Supports parallel processing
- Includes error handling and retries
- Generates detailed logs

## Prerequisites

- Python 3.6+
- Firecrawl API key

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install firecrawl-py python-dotenv
```

3. Create a `.env` file in the project root with your Firecrawl API key:

```
FIRECRAWL_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

Run the scraper with default settings:

```bash
python fire/run_scraper.py
```

This will:
- Extract URLs from `fire/list.md`
- Save scraped content to `fire/scraped_content/`
- Use 3 parallel workers

### Advanced Usage

Customize the scraper with command-line arguments:

```bash
python fire/run_scraper.py --input path/to/urls.md --output path/to/output --workers 5
```

### Command-line Arguments

- `--input`, `-i`: Path to the markdown file containing URLs (default: `fire/list.md`)
- `--output`, `-o`: Directory to save scraped content (default: `fire/scraped_content`)
- `--workers`, `-w`: Number of parallel workers (default: 3)
- `--verbose`, `-v`: Enable verbose output

## Output Format

Each scraped website is saved as a markdown file with the following structure:

```markdown
# Website Title

Source: https://example.com

Crawled on: YYYY-MM-DD HH:MM:SS

---

## Page: https://example.com

[Scraped content in markdown format]

---

## Page: https://example.com/subpage

[Scraped content in markdown format]

---
```

## Logging

Detailed logs are saved to `fire/scraper.log` and also displayed in the console.

## Limitations

- The scraper is limited to 5 pages per website by default
- Crawl depth is set to 2 by default
- Some websites may block scraping attempts

## License

This project is licensed under the MIT License - see the LICENSE file for details. 