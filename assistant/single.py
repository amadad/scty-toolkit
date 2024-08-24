import json
import os
from typing import Optional
from pydantic import BaseModel
from openai import AzureOpenAI

class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini-care"
    instructions: str = "You are a helpful customer service agent."
    tools: list = []

class Response(BaseModel):
    agent: Optional[Agent]
    messages: list

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = 'gpt-4o-mini-care'

def run_full_turn(agent, messages):
    current_agent = agent
    messages = messages.copy()

    while True:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "system", "content": current_agent.instructions}] + messages
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:
            print(f"{current_agent.name}:", message.content)

        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, {tool.__name__: tool for tool in current_agent.tools})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return Response(agent=current_agent, messages=messages[len(messages):])

def execute_tool_call(tool_call, tools):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    return tools[name](**args)

def escalate_to_human(summary):
    print("Escalating to human agent...")
    print(f"Summary: {summary}")
    exit()

customer_service_agent = Agent(
    name="Customer Service Agent",
    instructions="You are a customer service agent for ACME Inc.",
    tools=[escalate_to_human],
)

agent = customer_service_agent
messages = []

while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})

    response = run_full_turn(agent, messages)
    agent = response.agent
    messages.extend(response.messages)