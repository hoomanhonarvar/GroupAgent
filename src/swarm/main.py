from typing import Any, Dict , List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination , TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from autogen_core.models import ModelFamily, ModelInfo
import asyncio
def refund_flight(flight_id:str)->str:
    """Refund a flight."""
    return f"Refunding flight {flight_id}."

load_dotenv()
API_KEY=os.getenv("API_KEY")
BASE_URL=os.getenv("BASE_URL")
MODEL_NAME=os.getenv("MODEL_NAME")

model_info=ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    family=ModelFamily.UNKNOWN,
    structured_output=True,
    )

model = OpenAIChatCompletionClient(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        model_info=model_info
    )


# Agents
travel_agent=AssistantAgent(
    "travel_agent",
    model_client=model,
    handoffs=["flights_refunder","user"],
    system_message="""You are a travel agent.
    The flights_refunder is in charge of refunding flights.
    If you need information from the user, you must first send your message, then you can handoff to the user.
    Use TERMINATE when the travel planning is complete.
"""

)

flights_refunder=AssistantAgent(
    "flights_refunder",
    model_client=model,
    handoffs=["travel_agent","user"],
    tools=[refund_flight],
    system_message="""You are an agent specialized in refunding flights.
    You only need flight reference numbers to refund a flight.
    You have the ability to refund a flight using the refund_flight tool.
    If you need information from the user, you must first send your message, then you can handoff to the user.
    When the transaction is complete, handoff to the travel agent to finalize.
"""

)
termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")
team = Swarm([travel_agent, flights_refunder], termination_condition=termination)

task = "I need to refund my flight."


async def run_team_stream() -> None:
    input("Press enter to start the conversation.")
    task_result = await Console(team.run_stream(task=task))
    input("Press enter to continue the conversation.")
    last_message = task_result.messages[-1]

    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        task_result = await Console(
            team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
        )
        last_message = task_result.messages[-1]


asyncio.run(run_team_stream())
asyncio.run(model.close())