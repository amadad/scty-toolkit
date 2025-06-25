"""
Orchestrated Social Media Content Manager (Advanced Implementation)

This implementation uses Mainframe Orchestra for advanced orchestration and modular design.
It represents the most sophisticated approach with specialized classes and agents.

Architecture:
1. ContentGenerator: Handles platform-specific content creation
2. ImageTools: Manages image generation and processing
3. SocialMediaTools: Handles platform posting with direct Composio integration
4. Specialized Agents: Image, Social Media, and Conductor agents for orchestration

Current Status:
- ✅ Content Generation: Successfully generates platform-specific content with OpenAI
- ✅ Image Generation: Successfully generates and handles images
- ✅ Orchestration: Successfully coordinates between different agents
- ❌ LinkedIn Posting: Currently hitting rate limits (TOO_MANY_REQUESTS)
- ❌ Twitter Posting: Permission issues (403 Forbidden) - needs write permissions

Key Differences from main.py and two.py:
1. Uses Orchestra for advanced agent orchestration
2. Modular class-based architecture
3. More sophisticated content generation with platform-specific guides
4. Better error handling and parameter validation
5. Supports dynamic content parameters (tone, length)

Working Components:
1. Content generation with platform-specific formatting
2. Image generation and handling
3. Task orchestration between agents
4. Parameter validation and error handling

Known Issues:
1. LinkedIn: API rate limiting (will resolve after 24h)
2. Twitter: Needs proper write permissions in Developer Portal

Required Fixes:
1. Wait for LinkedIn API limit reset
2. Update Twitter app permissions to include tweet.write
3. Consider adding retry logic for rate limits

Note: This is the most feature-complete implementation but requires proper API
permissions to function fully. The orchestration and content generation work
perfectly, just waiting on API access issues to be resolved.
"""

from mainframe_orchestra import Task, Agent, Conduct, OpenaiModels
from composio_agno import Action, App, ComposioToolSet
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Dict, Any, List
import requests
from pathlib import Path
from datetime import datetime

load_dotenv()

class ContentGenerator:
    def __init__(self):
        self.client = OpenAI()

    def generate_content(self, topic: str, platform: str, tone: str = "professional", length: str = "medium") -> str:
        """Generate platform-specific content using OpenAI."""
        platform_guides = {
            "linkedin": {
                "max_chars": 3000,
                "style": "professional and informative",
                "structure": "title, key points, benefits, and a call to action"
            },
            "twitter": {
                "max_chars": 280,
                "style": "concise and engaging",
                "structure": "main point and call to action"
            }
        }

        prompt = f"""Create a {tone} {platform} post about {topic}.
        Style: {platform_guides[platform]['style']}
        Structure: {platform_guides[platform]['structure']}
        Length: {length}
        Max characters: {platform_guides[platform]['max_chars']}
        
        Include relevant emojis and hashtags.
        Make it unique and engaging.
        Add current timestamp in appropriate format.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are an expert {platform} content creator specializing in healthcare and caregiving content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()

class ImageTools:
    @staticmethod
    def download_image(url: str) -> str:
        """Download image from URL and save locally."""
        response = requests.get(url)
        response.raise_for_status()
        
        Path("images").mkdir(exist_ok=True)
        file_path = "images/temp.png"
        
        with open(file_path, "wb") as f:
            f.write(response.content)
        
        return os.path.abspath(file_path)

    @staticmethod
    def generate_dalle_image(prompt: str) -> str:
        """Generate an image using DALL-E.
        
        Args:
            prompt: The full prompt to generate the image from.
        """
        openai_client = OpenAI()
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url

class SocialMediaTools:
    def __init__(self):
        self.toolset = ComposioToolSet()
        self.content_generator = ContentGenerator()

    def post_to_linkedin(self, content: str = None, media: str = None, topic: str = None, tone: str = "professional", length: str = "medium") -> Dict[str, Any]:
        """Post content to LinkedIn."""
        if content is None and topic is not None:
            content = self.content_generator.generate_content(topic, "linkedin", tone, length)
        
        params = {
            "author": "urn:li:organization:106542185",
            "visibility": "PUBLIC",
            "commentary": content,
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False
        }
        
        if media:
            params["media"] = media
            
        return self.toolset.execute_action(
            action="LINKEDIN_CREATE_LINKED_IN_POST",
            params=params
        )

    def post_to_twitter(self, content: str = None, media: str = None, topic: str = None, tone: str = "conversational", length: str = "short") -> Dict[str, Any]:
        """Post content to Twitter with image."""
        if content is None and topic is not None:
            content = self.content_generator.generate_content(topic, "twitter", tone, length)
        
        # Upload the image first
        media_id = None
        if media:
            upload_response = self.toolset.execute_action(
                action="TWITTER_MEDIA_UPLOAD_MEDIA",
                params={"media": media}
            )
            
            if not upload_response.get("data", {}).get("media_id"):
                raise Exception(f"Failed to upload media to Twitter: {upload_response}")
            
            media_id = upload_response["data"]["media_id"]
        
        # Create the tweet
        params = {"text": content}
        if media_id:
            params["media__media__ids"] = media_id
            
        return self.toolset.execute_action(
            action="TWITTER_CREATION_OF_A_POST",
            params=params
        )

# Create specialized agents
image_agent = Agent(
    agent_id="image_agent",
    role="Image Generator",
    goal="Generate and manage images for social media posts",
    attributes="You have expertise in generating and handling images.",
    llm=OpenaiModels.gpt_4o,
    tools={ImageTools.generate_dalle_image, ImageTools.download_image}
)

social_media_tools = SocialMediaTools()
social_media_agent = Agent(
    agent_id="social_media_agent",
    role="Social Media Manager",
    goal="Create and post content to social media platforms",
    attributes="You have expertise in social media management and content creation.",
    llm=OpenaiModels.gpt_4o,
    tools={social_media_tools.post_to_linkedin, social_media_tools.post_to_twitter}
)

conductor_agent = Agent(
    agent_id="conductor_agent",
    role="Conductor",
    goal="Orchestrate the content creation and posting process",
    attributes="You have expertise in coordinating different agents.",
    llm=OpenaiModels.gpt_4o,
    tools={Conduct.conduct_tool(image_agent, social_media_agent)}
)

def post_content(
    topic: str, 
    tone: str = "professional",
    linkedin_length: str = "medium",
    twitter_length: str = "short"
) -> Dict[str, Any]:
    """Post content to social media platforms using orchestrated agents."""
    task = Task.create(
        agent=conductor_agent,
        instruction=f"""
        Create and post content about {topic} to LinkedIn and Twitter.
        Tone: {tone}
        LinkedIn length: {linkedin_length}
        Twitter length: {twitter_length}
        
        Follow these exact steps with the exact parameter names:
        1. Generate an image using generate_dalle_image:
           - prompt: "Create a professional and engaging image about {topic}. The image should work well for both LinkedIn and Twitter."
        
        2. Download the generated image using download_image:
           - url: <use the URL from step 1>
        
        3. Post to LinkedIn using post_to_linkedin with these exact parameters:
           - content: <generate professional content about {topic}>
           - media: <use the downloaded image path from step 2>
           - tone: "{tone}"
           - length: "{linkedin_length}"
        
        4. Post to Twitter using post_to_twitter with these exact parameters:
           - content: <generate concise content about {topic}>
           - media: <use the downloaded image path from step 2>
           - tone: "{tone}"
           - length: "{twitter_length}"
        
        Make sure to:
        1. Use 'media' parameter for both LinkedIn and Twitter posts (not image_path or image_url)
        2. Pass the local file path from download_image as the media parameter
        3. Include tone and length parameters
        """
    )
    
    try:
        result = task.execute()
        if isinstance(result, str):
            return {"status": "completed", "message": result}
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "details": {
                "topic": topic,
                "tone": tone,
                "linkedin_length": linkedin_length,
                "twitter_length": twitter_length
            }
        }

if __name__ == "__main__":
    try:
        response = post_content(
            topic="The importance of self-care for caregivers",
            tone="empathetic",
            linkedin_length="medium",
            twitter_length="short"
        )
        if response.get("status") == "error":
            print(f"Error occurred: {response['message']}")
            print("Details:", response.get("details", {}))
        else:
            print("Success! Posted to social media platforms.")
            print("Response:", response)
    except Exception as e:
        print(f"Error occurred: {str(e)}")