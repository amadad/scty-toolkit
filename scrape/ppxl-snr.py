import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

pplx_api_key = os.getenv("PPXL_API_KEY")
if not pplx_api_key:
    raise ValueError("PPXL_API_KEY not found in .env file")

url = "https://api.perplexity.ai/chat/completions"

payload = {
    "model": "llama-3.1-sonar-huge-128k-online",
    "messages": [
        {
            "role": "system",
            "content": "You are a thorough researcher. Provide a detailed, organized list of key strategies with numbered points. Include a brief introduction before listing the points. Each point should be concise but informative, and where possible, provide relevant sources or citations in brackets, with url links."
        },
        {
            "role": "user",
            "content": input("Enter your query: ")
        }
    ],
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.9,
    "return_citations": True,
    "search_domain_filter": ["perplexity.ai"],
    "return_images": False,
    "return_related_questions": False,
    "search_recency_filter": "month",
    "top_k": 0,
    "stream": False,
    "presence_penalty": 0,
    "frequency_penalty": 1
}
headers = {
    "Authorization": f"Bearer {pplx_api_key}",
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, json=payload)
response.raise_for_status()

result = response.json()
with open('ppxl-0910-huge.md', 'w', encoding='utf-8') as f:
    f.write(result['choices'][0]['message']['content'])
print("Response content saved to ppxl-0910.md")