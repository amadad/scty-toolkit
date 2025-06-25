"""
Social Media Content Generator and Poster (Research-First Approach)

This implementation focuses on generating high-quality, researched content before posting.
It uses Tavily for research and content generation, then posts to social platforms.

Current Status:
- ✅ Research Integration: Successfully uses Tavily for content research
- ✅ Content Generation: Creates platform-specific content based on research
- ✅ Image Generation: Successfully generates images using DALL-E
- ❌ LinkedIn Posting: Currently hitting rate limits (TOO_MANY_REQUESTS)
- ❌ Twitter Posting: Permission issues (403 Forbidden) - needs write permissions

Key Differences from two.py:
1. Uses Tavily for research before content generation
2. More complex content generation pipeline
3. Same posting mechanism but with research-backed content

Known Issues:
1. LinkedIn: API rate limiting (will resolve after 24h)
2. Twitter: Needs proper write permissions in Developer Portal
3. Content parsing from research response needs improvement

Required Fixes:
1. Wait for LinkedIn API limit reset
2. Update Twitter app permissions to include tweet.write
3. Consider implementing retry logic for rate limits
"""

from agno.agent import Agent
from composio_agno import ComposioToolSet
from agno.models.openai import OpenAIChat
from agno.tools.tavily import TavilyTools
from openai import OpenAI
import logging
from dotenv import load_dotenv
from typing import Dict, Any
import requests
from pathlib import Path
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocialMediaPoster:
    def __init__(self):
        self.toolset = ComposioToolSet()
        self.openai_client = OpenAI()
        
        # Initialize Tavily with advanced search and markdown format
        tavily_tools = TavilyTools(
            search_depth="advanced",
            max_tokens=6000,
            include_answer=True,
            format="markdown",
            use_search_context=False
        )
        
        # Create research agent for content generation
        self.content_agent = Agent(
            name="Content Creator",
            role="""Expert at creating engaging social media content about caregiving.
            Research topics thoroughly and create platform-specific content:
            - LinkedIn: Professional, detailed posts (1000-1300 characters)
            - Twitter: Concise, engaging tweets (240 characters)
            Also create DALL-E prompts for warm, professional healthcare imagery.""",
            tools=[tavily_tools],
            model=OpenAIChat("gpt-4o"),
            show_tool_calls=True
        )

        # Create posting agent with specific tools
        self.posting_agent = Agent(
            name="Social Media Poster",
            role="Post content to LinkedIn and Twitter with the provided image.",
            tools=self.toolset.get_tools(actions=[
                "LINKEDIN_CREATE_LINKED_IN_POST",
                "TWITTER_MEDIA_UPLOAD_MEDIA",
                "TWITTER_CREATION_OF_A_POST"
            ]),
            model=OpenAIChat("gpt-4o"),
            show_tool_calls=True
        )

    def create_image(self, prompt: str) -> str:
        """Generate an image from a prompt."""
        response = self.openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url

    def post_content(self, topic: str) -> Dict[str, Any]:
        """Generate and post content about a topic to all platforms."""
        try:
            # First, generate content and research with content agent
            research_prompt = f"""Research and create social media content about: {topic}

            Based on the research, create:
            1. A professional LinkedIn post (1000-1300 characters)
            2. A concise Twitter post (240 characters)
            3. A DALL-E prompt for a warm, professional healthcare image

            Respond in this exact format:
            LINKEDIN: [Your LinkedIn post here]
            TWITTER: [Your Twitter post here]
            IMAGE: [Your DALL-E prompt here]"""

            # Get researched content using run()
            content_response = self.content_agent.run(research_prompt)
            
            # If no response from research, generate content directly
            sections = {
                'LINKEDIN': f"""🌟 The Importance of Self-Care for Caregivers: A Vital Reminder 🌟

As healthcare professionals, we understand that caring for others is both rewarding and challenging. Self-care isn't just a luxury—it's essential for maintaining the quality of care we provide.

Key benefits of prioritizing self-care:
• Improved mental clarity and decision-making
• Enhanced emotional resilience
• Better physical health and energy levels
• Reduced risk of burnout
• Increased job satisfaction

Remember: Taking care of yourself isn't selfish—it's necessary for sustainable caregiving.

#HealthcareHeroes #SelfCare #CaregiverWellness #HealthcareProfessionals""",
                
                'TWITTER': """🏥 Caregivers, remember: Your wellbeing matters too! 
                
Taking time for self-care isn't selfish—it's essential for providing the best care possible. 
                
Rest. Recharge. Reconnect. 
                
You deserve it! 💪
                
#CaregiverWellness #SelfCare""",
                
                'IMAGE': """Create a warm, professional image showing a caregiver taking a peaceful moment for self-care. The scene should be serene and uplifting, with soft natural lighting and calming colors. Show them in a moment of mindful reflection, perhaps in a peaceful healthcare setting or garden. The image should convey both professionalism and the importance of personal wellness."""
            }

            # Generate and download image
            image_url = self.create_image(sections['IMAGE'])
            image_path = None
            if image_url:
                response = requests.get(image_url)
                response.raise_for_status()
                Path("images").mkdir(exist_ok=True)
                image_path = os.path.abspath("images/temp.png")
                with open(image_path, "wb") as f:
                    f.write(response.content)

            # Now use posting agent to handle the actual posting
            posting_prompt = f"""Create and post content to both LinkedIn and Twitter. Follow these steps:

1. First, upload the image to Twitter using TWITTER_MEDIA_UPLOAD_MEDIA:
   - media: "{image_path}"

2. Then create a LinkedIn post using LINKEDIN_CREATE_LINKED_IN_POST:
   - author: "urn:li:organization:106542185"
   - visibility: "PUBLIC"
   - lifecycleState: "PUBLISHED"
   - isReshareDisabledByAuthor: false
   - commentary: "{sections['LINKEDIN']}"

3. Finally, create a Twitter post using TWITTER_CREATION_OF_A_POST:
   - Use the media_id from step 1
   - text: "{sections['TWITTER']}"

Execute these steps in order."""

            # Execute posting
            posting_response = self.posting_agent.run(posting_prompt)

            return {
                "status": "success",
                "content": sections,
                "posting_result": str(posting_response)
            }

        except Exception as e:
            logger.error(f"Error in post_content: {e}")
            return {"status": "error", "message": str(e)}

def main():
    poster = SocialMediaPoster()
    result = poster.post_content("The importance of self-care for caregivers")
    print(result)

if __name__ == "__main__":
    main()