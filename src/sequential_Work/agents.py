from dataclasses import dataclass
from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
    type_subscription,
)
from autogen_core.models import ChatCompletionClient , SystemMessage, UserMessage
joker_topic_type="jokerAgent"
reacter_topic_type="ReacterAgent"
format_proof_topic_type="FormatProofAgent"
user_topic_type="User"

@dataclass
class Message :
    content: str

@type_subscription(topic_type=joker_topic_type)
class jokerAgent(RoutedAgent):
    def __init__(self, model_client:ChatCompletionClient)->None:
        super().__init__("A joker Agent")
        self._system_message=SystemMessage(
            content=("""You are a joker Agent tell a awsome joke related to the concept!"""))

        self._model_client=model_client

    @message_handler
    async def handle_retrive_messages(self,message:Message,ctx:MessageContext )->None:
        print("recieved")
        prompt=f"recieved concept to joking:{message.content}"
        llm_result=await self._model_client.create(
            messages=[
                self._system_message,
                UserMessage(content=prompt,source=self.id.key)
            ],
            cancellation_token=ctx.cancellation_token,
        )
        response=llm_result.content
        assert isinstance(response,str)
        print(f"{'-'*80}\n{self.id.type}:\n{response}")

        await self.publish_message(Message(response),topic_id=TopicId(reacter_topic_type,source=self.id.key))



@type_subscription(topic_type=reacter_topic_type)
class ReacterAgent(RoutedAgent):
    def __init__(self,model_client:ChatCompletionClient)->None:
        super().__init__("A Reacter Agent.")
        self._system_message=SystemMessage(
            content=("""You are a Reacter Agent whom reacts to the joke."""))

        self._model_client=model_client
    @message_handler
    async def handle_generate_messages(self,message:Message,ctx:MessageContext)->None:
        prompt=f"you should react to this joke  :{message.content}"
        llm_result=await self._model_client.create(
            messages=[
                self._system_message,
                UserMessage(content=prompt,source=self.id.key)
            ],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        print(f"{'-'*80}\n{self.id.type}:\n{response}")

        await self.publish_message(Message(response), topic_id=TopicId(format_proof_topic_type, source=self.id.key))

@type_subscription(topic_type=format_proof_topic_type)
class FormatProofAgent(RoutedAgent):
    def __init__(self,model_client:ChatCompletionClient)->None:
        super().__init__("A Format Proof Agent")
        self._system_message=SystemMessage(
            content=("""
            You are an editor. Given the draft copy, correct grammar, improve clarity, ensure consistent tone, 
            give format and make it polished. Output the final improved copy as a single text block.
                     """))

        self._model_client=model_client

    @message_handler
    async def handle_format_proof(self,message:Message,ctx:MessageContext)->None:
        prompt=f"Draft Copy:{message.content}"
        llm_result=await self._model_client.create(
            messages=[
                self._system_message,
                UserMessage(content=prompt,source=self.id.key)
            ],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        print(f"{'-'*80}\n{self.id.type}:\n{response}")

        await self.publish_message(Message(response), topic_id=TopicId(user_topic_type, source=self.id.key))

    
@type_subscription(topic_type=user_topic_type)
class UserAgent(RoutedAgent):
    def __init__(self)->None:
        super().__init__("A user agent that outputs the final copy to the user.")
        

    @message_handler
    async def handle_user_message(self,message:Message,ctx:MessageContext)->None:
        print(f"\n{'-'*80}\n{self.id.type} received final copy:\n{message.content}")

