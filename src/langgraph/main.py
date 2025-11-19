from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from graph import *
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from langchain_core.messages import messages_from_dict
load_dotenv()
mongo_url=os.getenv("MONGO_URL")
mongo_port=os.getenv("MONGO_PORT")
uri=f"mongodb://{mongo_url}:{mongo_port}"
client=MongoClient(uri)
DB=client["ChatBot"]
sessions= DB["sessions"]


def change_Message_type(session):
    return tutorials(
        {
            "user_id":session["user_id"],
            "intent":session["intent"],
            "messages":messages_from_dict(session["messages"]),
            "name":session["name"],
            "stage":session["stage"],
        }
    )
    


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
    


    


def get_user_state(user_id)->tutorials:
    if not sessions.find_one({"user_id":user_id}):
        
        sessions.insert_one({
        "user_id": user_id,
        "intent": UserIntent(
            user_intent="",
            summary="",
            ideal_score=0
        ),
        # "messages": Annotated[list[AnyMessage],operator.add],
        "messages": [],
        "name": "",
        "stage":"greeting",
        })
    session_user=sessions.find_one({"user_id":user_id})
    return session_user
        

@app.post("/ielts")
def chat(request:ielts_request):
    state=get_user_state(request.user_id)
    
    state["messages"].append(request.messages)
    new_state=workflow.invoke(state)    
    sessions.update_one({"user_id": new_state["user_id"]}, {"$set": {
       
        "intent": new_state["intent"],
        "messages": new_state["messages"],
        "name": new_state["name"],
        "stage": new_state["stage"],
    }
    })
    return{
        "response":new_state["messages"][-1],
        "stage":new_state["stage"],
        "user_id":new_state["user_id"],
        "intent":new_state["intent"],
        "user_name":new_state["name"]
    }