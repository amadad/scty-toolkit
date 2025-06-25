"""
Simple Social Media Content Poster (Direct Approach)

This implementation focuses on direct posting functionality with predefined content.
It uses a straightforward approach without research integration.

Current Status:
- ✅ Image Generation: Successfully generates images using DALL-E
- ✅ Image Download: Successfully downloads and saves images locally
- ✅ Media Upload: Successfully uploads media to Twitter (gets media_id)
- ❌ LinkedIn Posting: Currently hitting rate limits (TOO_MANY_REQUESTS)
- ❌ Twitter Posting: Permission issues (403 Forbidden) - needs write permissions

Key Differences from main.py:
1. No research integration - uses predefined content
2. Simpler, more focused implementation
3. Direct posting without content generation complexity

Working Components:
1. DALL-E image generation
2. Local file handling
3. Media upload to Twitter

Known Issues:
1. LinkedIn: API rate limiting (will resolve after 24h)
2. Twitter: Needs proper write permissions in Developer Portal

Required Fixes:
1. Wait for LinkedIn API limit reset
2. Update Twitter app permissions to include tweet.write

Note: This implementation is more reliable for testing posting functionality
as it removes the complexity of content generation and research.
"""

from agno.agent.agent import Agent
from composio_agno import Action, App, ComposioToolSet
from agno.models.openai import OpenAIChat
from openai import OpenAI
from dotenv import load_dotenv
import json
import requests
import os
from pathlib import Path

load_dotenv()

# Initialize clients
toolset = ComposioToolSet()
openai_client = OpenAI()

def download_image(url: str) -> str:
    """Download image from URL and save locally."""
    response = requests.get(url)
    response.raise_for_status()
    
    # Create images directory if it doesn't exist
    Path("images").mkdir(exist_ok=True)
    
    # Save the image
    file_path = "images/temp.png"
    with open(file_path, "wb") as f:
        f.write(response.content)
    
    return os.path.abspath(file_path)

def generate_image(prompt: str) -> str:
    """Generate an image using DALL-E and return local file path."""
    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    url = response.data[0].url
    return download_image(url)

def post_content(topic: str):
    try:
        # First, generate image prompt
        image_prompt = f"Create a professional and engaging image about {topic}. The image should work well for both LinkedIn and Twitter."
        image_path = generate_image(image_prompt)
        
        # Now create and post content using Composio actions
        posting_agent = Agent(
            name="Social Media Poster",
            role="Post content to LinkedIn and Twitter with the provided image.",
            tools=toolset.get_tools(actions=[
                "LINKEDIN_CREATE_LINKED_IN_POST",
                "TWITTER_MEDIA_UPLOAD_MEDIA",
                "TWITTER_CREATION_OF_A_POST"
            ]),
            model=OpenAIChat("gpt-4o"),
            show_tool_calls=True
        )
        
        posting_prompt = f"""Create and post content about {topic} to both LinkedIn and Twitter. Follow these steps:

1. First, upload the image to Twitter using TWITTER_MEDIA_UPLOAD_MEDIA:
   - media: "{image_path}"

2. Then create a LinkedIn post using LINKEDIN_CREATE_LINKED_IN_POST:
   - author: "urn:li:organization:XXXX"
   - visibility: "PUBLIC"
   - lifecycleState: "PUBLISHED"
   - isReshareDisabledByAuthor: false
   - media: "{image_path}"
   Write professional, detailed content about {topic}

3. Finally, create a Twitter post using TWITTER_CREATION_OF_A_POST:
   - Use the media_id from step 1
   Write concise, engaging content with relevant hashtags about {topic}

Execute these steps in order."""

        posting_response = posting_agent.run(posting_prompt)
        
        return {
            "image_prompt": image_prompt,
            "image_path": image_path,
            "posting_result": str(posting_response)
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    result = post_content("The importance of self-care for caregivers")
    print(json.dumps(result, indent=2))
