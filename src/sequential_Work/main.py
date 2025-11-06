from dataclasses import dataclass
from autogen_core import (
    SingleThreadedAgentRuntime,
    TopicId,
)
from autogen_core import (
    SingleThreadedAgentRuntime,
    TopicId,
)
from autogen_core.models import ModelFamily, ModelInfo
from dotenv import load_dotenv
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from agents import *
import asyncio

import logging

from autogen_core import TRACE_LOGGER_NAME,EVENT_LOGGER_NAME

# logging.basicConfig(level=logging.WARNING)
# logger = logging.getLogger(EVENT_LOGGER_NAME)
# logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.DEBUG)

# logger = logging.getLogger(EVENT_LOGGER_NAME + ".my_module")
# logger.info(Message("content"))







joker_topic_type="jokerAgent"
reacter_topic_type="ReacterAgent"
format_proof_topic_type="FormatProofAgent"
user_topic_type="User"



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
runtime=SingleThreadedAgentRuntime()

async def main():
    await jokerAgent.register(runtime,type=joker_topic_type,factory=lambda: jokerAgent(model_client=model))

    await ReacterAgent.register(runtime,type=reacter_topic_type,factory=lambda: ReacterAgent(model_client=model))

    await FormatProofAgent.register(runtime,type=format_proof_topic_type,factory=lambda: FormatProofAgent(model_client=model))

    await UserAgent.register(runtime,type=user_topic_type,factory=lambda: UserAgent())
    runtime.start()
    await runtime.publish_message(
        Message(content="Tell a joke a bout programming!"),
        topic_id=TopicId(type=joker_topic_type,source="defualt")
    )
    await runtime.stop_when_idle()
    await model.close()

asyncio.run(main())
