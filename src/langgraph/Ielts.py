from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict,Literal,Annotated
import operator
from langchain.messages import AnyMessage,HumanMessage
from langgraph.graph import START,END,StateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel,Field

load_dotenv()
BASEURL=os.getenv("BASE_URL")
API_KEY=os.getenv("API_KEY")
MODEL_NAME=os.getenv("MODEL_NAME")
class user_info(BaseModel):
    username: str|None = Field(description="The name of the person. if he/she doesn't mention should be null")
    user_intent:Literal["writing","Grammar","vocabulary","None"]|None = Field(description="the Intent of user like writing, Grammar, vocabulary or if he/she doesn't mentioned directly should be None")
    ideal_score: int|None = Field(description="the score target by user if he/she doesn't mention directly 0")
    summary: str|None = Field(description="the summary of user message ")

base_model= ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASEURL,
)
class UserIntent(TypedDict):
    user_intent=Literal["writing","Grammar","vocabulary"]
    summary=str
    ideal_score=int
    

class tutorials(TypedDict):
    intent:UserIntent | None
    messages:Annotated[list[AnyMessage],operator.add]
    name:str | None
    stage:Literal["greeting"]
    user_id:str
    



def greeting(state:tutorials)->dict:
    print("state in greeting: ",state["intent"])

    Message=state["messages"][-1]
    print(Message.content)
    structured_output=base_model.with_structured_output(user_info)
    search_prompt=f"""
    search for user intent, Ideal score, username, summary
    find this information when directly mentioned in message 
    user_intent should be one of writing, Grammar, vocabulary or None and until user don't mention it, you should not filled it
    ideal score can be analysis from message and should be int you must not use your imagination. when it mentioned by user clearly and some descriptions about it set this score, otherwise it should be 0
    Human message: {Message.content}
    """

    greeting_prompt_intent=f"""
    you are a helpful assistant for IELTS.
    greeting to user and ask for user intent for practice like writing, Grammar, vocabulary or ideal score
    """


    prompt_score=f"""
    you are a helpful assistant for IELTS.
     ask for user her/his ideal score
    """
    



    llm_answer=structured_output.invoke(search_prompt)
    print("user_info:  ",llm_answer)
    # print("type: ",type(llm_answer))
    if llm_answer.user_intent!="" and llm_answer.user_intent!=None and  llm_answer.user_intent!="None" or state["intent"]["user_intent"]:
        print("ok")
        state["intent"]["user_intent"]=llm_answer.user_intent
        if llm_answer.ideal_score!=0:
            state["intent"]["ideal_score"]=llm_answer.ideal_score
            state["intent"]["summary"]=llm_answer.summary
            print("1: ",state["intent"]["user_intent"])
            print("2:  ",state["intent"]["ideal_score"])
            prompt_recieved_intent=f"""
            you are a helpful assistant for IELTS.
            tanks user for sharing his/her information.
            ask user if he/she wants to pracitce {state["intent"]["user_intent"]} and is her/his ideal score {state["intent"]["ideal_score"]}?
            just to be sure
            """
            result=base_model.invoke([{"role":"user","content":prompt_recieved_intent}]) 
            state["messages"].append({"role":"user","content":result})

        else:
            result=base_model.invoke([{"role":"user","content":prompt_score}]) 
            state["messages"].append({"role":"user","content":result})
    
        
    else:
        result=base_model.invoke([{"role":"user","content":greeting_prompt_intent}]) 
        state["messages"].append({"role":"user","content":result})
    state['name']=llm_answer.username
    
    return state

def create_workflow():

    workflow=StateGraph(tutorials)
    workflow.add_node("greeting",greeting)
    workflow.add_edge(START,"greeting")
    workflow.add_edge("greeting",END)

    app=workflow.compile()
    return app
# message=input("start a chat...")
# start_state=tutorials({"messages":[HumanMessage(content=message)]})
# result=app.invoke(start_state)
# print(result)
# print(len(result["messages"]))

    
    
