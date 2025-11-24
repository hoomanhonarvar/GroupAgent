from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from graph import *
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from langchain_core.messages import messages_from_dict
from bson import ObjectId
from langgraph.checkpoint.mongodb import MongoDBSaver  
import ormsgpack
from langgraph.checkpoint.base import get_checkpoint_id  # Optional, for debugging
from contextlib import asynccontextmanager
from langgraph.checkpoint.base import Checkpoint

load_dotenv()
mongo_url=os.getenv("MONGO_URL")
mongo_port=os.getenv("MONGO_PORT")
db_name=os.getenv("DB_NAME")
uri=f"mongodb://{mongo_url}:{mongo_port}"
client=MongoClient(uri)
class ielts_request(BaseModel):
    messages: str
    user_id:str


graph=None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("starting...")
    

    db = client[db_name]
    checkpoint_collection_name="checkpoints"
    session_collection_name="sessions"
    if checkpoint_collection_name not in db.list_collection_names():
        db.create_collection(checkpoint_collection_name)
    if session_collection_name not in db.list_collection_names():
        db.create_collection(session_collection_name)
    
    # db["checkpoints"].insert_one({"init": True, "thread_id": "__init__"})
    print(f"db '{db_name} is ready")
    checkpointer = MongoDBSaver(
    client=client, database_name=db_name, collection_name=checkpoint_collection_name
        )
    workflow=create_workflow()
    global graph
    graph = workflow.compile(checkpointer=checkpointer)
    print("graph compiled!")
    yield
    print("shutting down...")
    client.close()
    print("database closed!")

def get_session(user_id: str):
    sessions=client[db_name]["sessions"]
    if not sessions.find_one({"user_id":user_id}):
        
        sessions.insert_one({
        "user_id": user_id,
        
        "user_intent":"",
        "summary":"",
        "ideal_score":0,
        # "messages": Annotated[list[AnyMessage],operator.add],
        "messages": [],
        "name": "",
        "stage":"greeting",
        })
    session_user=sessions.find_one({"user_id":user_id})
    return session_user

app=FastAPI(lifespan=lifespan)
@app.post("/conversation")
def chat(request:ielts_request):
    config = {
        "configurable": {
            "thread_id": request.user_id  # Consistent per user
        }
    }
    result=graph.invoke({"messages": [request.messages]},config=config)
    return result
@app.post("/new_ielts")
def chat(request:ielts_request):
    config = {
        "configurable": {
            "thread_id": request.user_id
        }
    }
    result=graph.invoke({"messages": [request.messages],"stage":"greeting"},config=config)
    return result["messages"][-1]

@app.post("/ielts")
def chat(request:ielts_request):
    
    config = {
        "configurable": {
            "thread_id": request.user_id  # Consistent per user
        }
    }
    state=get_session(request.user_id)
    state["messages"].append(request.messages)
    state["_id"] = str(state["_id"])
    new_state = graph.invoke(state, config)
    sessions=client[db_name]["sessions"]
    sessions.update_one({"user_id": new_state["user_id"]}, {"$set": {
       
        "user_intent":new_state["user_intent"],
        "summary":new_state["summary"],
        "ideal_score":new_state["ideal_score"],
        "messages": new_state["messages"],
        "name": new_state["name"],
        "stage": new_state["stage"],
    }
    })
    return{
        "response":new_state["messages"][-1],
        "stage":new_state["stage"],
        "user_id":new_state["user_id"],
        "user_intent":new_state["user_intent"],
        "summary":new_state["summary"],
        "ideal_score":new_state["ideal_score"],
        "user_name":new_state["name"]
    }