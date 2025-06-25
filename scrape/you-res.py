import os
import requests

# Research Mode API
url = "https://chat-api.you.com/research"

payload = {
    "query": input("Enter your query: "),
    "chat_id": "3c90c3cc-0d44-4b50-8888-8dd25736052a"
}
headers = {
    "X-API-Key": "0aa45074-a8aa-43e2-93a2-1b47041297f6<__>1Pv1obETU8N2v5f4sMIlfUDT",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

# Save the raw response content to a markdown file
with open('res-0910.md', 'w', encoding='utf-8') as f:
    f.write(response.text)

print("Response saved to res-0910.md")