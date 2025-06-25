from swarm import Agent
from swarm.repl import run_demo_loop

agent = Agent(
    name="Build",
    instructions="",
    functions=[],
)

run_demo_loop(agent)