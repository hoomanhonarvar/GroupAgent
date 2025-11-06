from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from autogen_core.models import ModelFamily, ModelInfo
import asyncio

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
writer=AssistantAgent(
    "Writer",
    model_client=model,
    system_message="""You are a writer. Draft a short paragraph on climate change."""
)
reviewer=AssistantAgent(
    "Reviewer",
    model_client=model,
    system_message="""You are a reviewer. Review the draft and suggest improvment"""
)

builder=DiGraphBuilder()
builder.add_node(writer).add_node(reviewer)
builder.add_edge(writer,reviewer)

graph=builder.build()
flow=GraphFlow([writer,reviewer],graph)


async def main():
    stream=flow.run_stream(task="Write a short paragraph about Climate change.")
    async for event in stream:
        print(event)

asyncio.run(main())