from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from Ielts import *


workflow=create_workflow()
app=FastAPI()
Sessions:Dict[str,tutorials] ={}
class ielts_request(BaseModel):
    messages: str
    user_id:str


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
    


def get_user_state(user_id)->tutorials:
    if user_id not in Sessions:
        Sessions[user_id]=tutorials({
        "intent": UserIntent(
            user_intent=None,
            summary="",
            ideal_score=0
        ),
        "messages": Annotated[list[AnyMessage],operator.add],
        "name": None,
        "stage": "greeting",
        "user_id": user_id,})
        Sessions[user_id]["messages"]=[]
        print("nabood")
    
    return Sessions[user_id]
        

@app.post("/ielts")
def chat(request:ielts_request):
    state=get_user_state(request.user_id)
    state["messages"]=state["messages"]+[ HumanMessage(content=request.messages)]
    new_state=workflow.invoke(state)    
    Sessions[request.user_id]=new_state
    # print(new_state["messages"][-1],"\n\n\n\n")
    return{
        "response":new_state["messages"][-1]["content"].content,
        "stage":new_state["stage"],
        "user_id":new_state["user_id"],
        "intent":new_state["intent"],
        "user_name":new_state["name"]
    }