import json
import requests
from bs4 import BeautifulSoup
import newspaper
from newspaper import Article
import ast
import os

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

def get_search_terms(topic):
    system_prompt = "You are a world-class journalist. Generate a list of 5 search terms to search for to research and write an article about the topic."
    messages = [
        {"role": "user", "content": f"Please provide a list of 5 search terms related to '{topic}' for researching and writing an article. Respond with the search terms separated by commas."},
    ]
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": 'claude-3-haiku-20240307',
        "max_tokens": 200,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": messages,
    }

    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
    response_text = response.json()['content'][0]['text']
    search_terms = [term.strip() for term in response_text.split(",")]
    return search_terms

def get_search_results(search_term):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": search_term})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    data = response.json()
    return data['organic']

def select_relevant_urls(search_results):
    system_prompt = "You are a journalist assistant. From the given search results, select the URLs that seem most relevant and informative for writing an article on the topic."
    search_results_text = "\n".join([f"{i+1}. {result['link']}" for i, result in enumerate(search_results)])
    messages = [
        {"role": "user", "content": f"Search Results:\n{search_results_text}\n\nPlease select the numbers of the URLs that seem most relevant and informative for writing an article on the topic. Respond with the numbers in a Python-parseable list, separated by commas."},
    ]
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": 'claude-3-haiku-20240307',
        "max_tokens": 200,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": messages,
    }
    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
    response_text = response.json()['content'][0]['text']

    numbers = ast.literal_eval(response_text)
    relevant_indices = [int(num) - 1 for num in numbers]
    relevant_urls = [search_results[i]['link'] for i in relevant_indices]

    return relevant_urls

def get_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def write_article(topic, article_texts):
    system_prompt = "You are a journalist. Write a high-quality, NYT-worthy article on the given topic based on the provided article texts. The article should be well-structured, informative, and engaging."
    combined_text = "\n\n".join(article_texts)
    messages = [
        {"role": "user", "content": f"Topic: {topic}\n\nArticle Texts:\n{combined_text}\n\nPlease write a high-quality, NYT-worthy article on the topic based on the provided article texts. The article should be well-structured, informative, and engaging. Ensure the length is at least as long as a NYT cover story -- at a minimum, 15 paragraphs."},
    ]
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": 'claude-3-opus-20240229',
        "max_tokens": 3000,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": messages,
    }
    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
    article = response.json()['content'][0]['text']
    return article

def edit_article(article):
    system_prompt = "You are an editor. Review the given article and provide suggestions for improvement. Focus on clarity, coherence, and overall quality."
    messages = [
        {"role": "user", "content": f"Article:\n{article}\n\nPlease review the article and provide suggestions for improvement. Focus on clarity, coherence, and overall quality."},
    ]
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": 'claude-3-opus-20240229',
        "max_tokens": 3000,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": messages,
    }
    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
    suggestions = response.json()['content'][0]['text']

    system_prompt = "You are an editor. Rewrite the given article based on the provided suggestions for improvement."
    messages = [
        {"role": "user", "content": f"Original Article:\n{article}\n\nSuggestions for Improvement:\n{suggestions}\n\nPlease rewrite the article based on the provided suggestions for improvement."},
    ]
    data = {
        "model": 'claude-3-opus-20240229',
        "max_tokens": 3000,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": messages,
    }
    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
    edited_article = response.json()['content'][0]['text']
    return edited_article

# User input
topic = input("Enter a topic to write about: ")
do_edit = input("After the initial draft, do you want an automatic edit? This may improve performance, but is slightly unreliable. Answer 'yes' or 'no'.")

# Generate search terms
search_terms = get_search_terms(topic)
print(f"\nSearch Terms for '{topic}':")
print(", ".join(search_terms))

# Perform searches and select relevant URLs
relevant_urls = []
for term in search_terms:
    search_results = get_search_results(term)
    urls = select_relevant_urls(search_results)
    relevant_urls.extend(urls)

print('Relevant URLs to read:', relevant_urls)


# Get article text from relevant URLs
article_texts = []
for url in relevant_urls:
  try:
    text = get_article_text(url)
    if len(text) > 75:
      article_texts.append(text)
  except:
    pass

print('Articles to reference:', article_texts)

print('\n\nWriting article...')
# Write the article
article = write_article(topic, article_texts)
print("\nGenerated Article:")
print(article)

if 'y' in do_edit:
  # Edit the article
  edited_article = edit_article(article)
  print("\nEdited Article:")
  print(edited_article)