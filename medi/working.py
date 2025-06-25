from agno.agent.agent import Agent
from agno.team.team import Team
from composio_agno import ComposioToolSet
from agno.models.openai import OpenAIChat
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import requests

load_dotenv()

def download_image(url: str) -> str:
    """Download image from URL and save locally."""
    Path("images").mkdir(exist_ok=True)
    image_path = Path("images/temp.png").absolute()
    image_path.write_bytes(requests.get(url).content)
    return str(image_path)

# Create specialized agents
image_creator = Agent(
    name="Image Creator",
    role="Create engaging images using DALL-E",
    model=OpenAIChat("gpt-4"),
)

social_poster = Agent(
    name="Social Media Poster",
    role="Post content to LinkedIn and Twitter",
    tools=ComposioToolSet().get_tools(actions=[
        "LINKEDIN_CREATE_LINKED_IN_POST",
        "TWITTER_MEDIA_UPLOAD_MEDIA",
        "TWITTER_CREATION_OF_A_POST"
    ]),
    model=OpenAIChat("gpt-4"),
)

# Create team
social_team = Team(
    name="Social Media Team",
    mode="coordinate",
    members=[image_creator, social_poster],
    model=OpenAIChat("gpt-4"),
    instructions=[
        "1. Have Image Creator generate a DALL-E prompt",
        "2. Generate image using DALL-E",
        "3. Have Social Poster upload and post content with the image"
    ],
    markdown=True,
    show_tool_calls=True,
    show_members_responses=True,
)

def post_content(topic: str):
    try:
        # First get image prompt from image creator
        prompt_response = image_creator.run(
            f"Create a professional and engaging DALL-E prompt for content about: {topic}"
        )
        
        # Generate image
        image_url = OpenAI().images.generate(
            model="dall-e-3",
            prompt=str(prompt_response),
            size="1024x1024",
            n=1
        ).data[0].url
        
        # Download image
        image_path = download_image(image_url)
        
        # Have team handle the posting
        return social_team.run(f"""Post about {topic} to LinkedIn and Twitter:
1. Upload image to Twitter (TWITTER_MEDIA_UPLOAD_MEDIA):
   - media: "{image_path}"
2. Post to LinkedIn (LINKEDIN_CREATE_LINKED_IN_POST):
   - author: "urn:li:organization:106542185"
   - visibility: "PUBLIC"
   - lifecycleState: "PUBLISHED"
   - media: "{image_path}"
3. Post to Twitter (TWITTER_CREATION_OF_A_POST) using media_id from step 1""")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    social_team.print_response("The importance of self-care for caregivers", stream=True)
