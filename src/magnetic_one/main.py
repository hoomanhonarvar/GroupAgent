import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily, ModelInfo
from dotenv import load_dotenv
import os
from autogen_ext.agents.web_surfer import MultimodalWebSurfer


async def main() -> None:
    
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

    assistant = AssistantAgent(
        "Assistant",
        model_client=model,
    )
    surfer = MultimodalWebSurfer(
        "WebSurfer",
        model_client=model,
    )
    team = MagenticOneGroupChat([surfer], model_client=model)
    await Console(team.run_stream(task="What is the UV index in Melbourne today?"))
    await model.close()


asyncio.run(main())