from autogen_agentchat.agents import AssistantAgent,MessageFilterAgent,MessageFilterConfig,PerSourceFilter
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from autogen_core.models import ModelFamily, ModelInfo
import asyncio
from autogen_agentchat.conditions import MaxMessageTermination

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


# a loop betwwen generator and reviewer (which exits when reviewr says "APPROVE")
# A summarizer agent that only sees the first user input and lat reviwer message
generator =AssistantAgent(
    "generator",
    model_client=model,
    system_message="Generate a list of creative ideas."
)
reviewer=AssistantAgent(
    "reviewer",
    model_client=model,
    system_message="Review ideas and provide feedbacks, or just 'APPROVE' for final approval."
)
summarizer_core=AssistantAgent(
    "summary",
    model_client=model,
    system_message="Summarize the user request and the final feedback"
)


#filter
filtered_summarizer=MessageFilterAgent(
    name="summary",
    wrapped_agent=summarizer_core,
    filter=MessageFilterConfig(
        per_source=[
            PerSourceFilter(source="user",position="first",count=1),
            PerSourceFilter(source="reviewer",position="last",count=1)
        ]
    )
)
builder=DiGraphBuilder()
builder.add_node(generator).add_node(reviewer).add_node(filtered_summarizer)
builder.add_edge(generator,reviewer)
builder.add_edge(reviewer,filtered_summarizer,condition=lambda msg: "APPROVE" in msg.to_model_text())
builder.add_edge(reviewer,generator,condition=lambda msg: "APPROVE" not in msg.to_model_text())

builder.set_entry_point(generator)
graph=builder.build()
termination_condition=MaxMessageTermination(10)
flow=GraphFlow(
    participants=builder.get_participants(),
    graph=graph,
    termination_condition=termination_condition
)


async def main():
    stream=flow.run_stream(task="Brainstorm ways to reduce plastic waste")
    async for event in stream:
        print(event)

asyncio.run(main())