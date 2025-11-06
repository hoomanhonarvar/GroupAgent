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
    You must hand-off to the flights_refunder to refund flights.
    The flights_refunder agent is in charge of refunding flights.
    only flights_refunder can refund the flights.
    If user wants to refund a flight you must hand-off to the flights_refunder to refund the flight.
    If you need information from the user, you must first send your message, then you must hand-off to the user.
    Use TERMINATE when the travel planning is complete.
"""

)

flights_refunder=AssistantAgent(
    "flights_refunder",
    model_client=model,
    handoffs=["travel_agent","user"],
    tools=[refund_flight],
    system_message="""You are an agent specialized in refunding flights.
    first send your message.
    You only need flight reference numbers to refund a flight.
    You have the ability to refund a flight using the refund_flight tool.
    You have to be in conversation with the user until you get the flight_ID from user by hand offing to the user that means after each message from you handoff to user.
    If you need information from the user, you must get it from user by handoffing to the user.
    When the transaction is complete, hand off to the travel agent to finalize.
"""

)
termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")
team = Swarm([travel_agent, flights_refunder], termination_condition=termination)

task = "I need to refund my flight."


async def run_team_stream() -> None:
    input("Press enter to start the conversation.")
    task_result = await Console(team.run_stream(task=task))
    last_message = task_result.messages[-1]

    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        task_result = await Console(
            team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
        )
        last_message = task_result.messages[-1]
    model.close()

asyncio.run(run_team_stream())

