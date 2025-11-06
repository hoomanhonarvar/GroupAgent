#Message filtering is a separate feature that allows you to filter the messages received by each agent and limiting their model context to only the relevant information.
#  The set of message filters defines the message graph in the flow.
from autogen_agentchat.agents import AssistantAgent,MessageFilterAgent,PerSourceFilter,MessageFilterConfig
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from autogen_core.models import ModelFamily, ModelInfo
import asyncio
from autogen_agentchat.ui import Console

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
researcher=AssistantAgent(
    'reseacher',
    model_client=model,
    system_message="Summarize key facts about climate change."
)
analyst=AssistantAgent("analyst",
                       model_client=model,
                       system_message="Review the summary and suggest improvements.")
presenter =AssistantAgent(
    "presenter",
    model_client=model,
    system_message="Prepare a presentation slide based on the final summary."
)

# message filltering

filtered_analyst =MessageFilterAgent(
    name="analyst",
    wrapped_agent=analyst,
    filter=MessageFilterConfig(per_source=[PerSourceFilter(source="reseacher",position="last",count=1)])
)
filtered_presenter=MessageFilterAgent(
    name="presenter",
    wrapped_agent=presenter,
    filter=MessageFilterConfig(per_source=[PerSourceFilter(source="analyst",position="last",count=1)])
)
builder=DiGraphBuilder()
builder.add_node(researcher).add_node(filtered_analyst).add_node(filtered_presenter)
builder.add_edge(researcher,filtered_analyst)
builder.add_edge(filtered_analyst,filtered_presenter)


graph=builder.build()
flow=GraphFlow(
    participants=builder.get_participants(),
    graph=graph
)


async def main():
    await Console(flow.run_stream(task="Summarize key facts about climate change."))

asyncio.run(main())