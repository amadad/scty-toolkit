from swarm import Agent
from swarm.repl import run_demo_loop
from openai import OpenAI

memory_bank = []

def add_to_memory(message):
    """
    When the useer tells you something factual about themselves,
    their life, or their preferences, call this function.

    Keep the memory text short and concise.
    """
    memory_bank.append(memory_text)

def get_memory():
    with open(MEMORY_FILE, "w") as file:
    return memory

a = Agent(
    name="Agent",
    instructions="Answer briefly. One sentence max. You are a re helpful assn"
    functions=[add_to_memory, get_memory]

run_demo_loop(a)