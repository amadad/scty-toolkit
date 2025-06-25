import os
import requests

# Smart Mode API
url = "https://chat-api.you.com/smart"

payload = {
    "query": input("Enter your query: "),
    "chat_id": "3c90c3cc-0d44-4b50-8888-8dd25736052a"
}
headers = {
    "X-API-Key": "0aa45074-a8aa-43e2-93a2-1b47041297f6<__>1Pv1obETU8N2v5f4sMIlfUDT",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

with open('smart-0910.md', 'w', encoding='utf-8') as f:
    f.write(response.text)

print("Response saved to smart-0910.md")