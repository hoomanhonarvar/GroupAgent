from fastapi import FastAPI
from pydantic import BaseModel
from graph import *
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from langgraph.checkpoint.mongodb import MongoDBSaver  
from contextlib import asynccontextmanager

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



app=FastAPI(lifespan=lifespan)
@app.post("/english_assistance")
def chat(request:ielts_request):
    config = {
        "configurable": {
            "thread_id": request.user_id
        }
    }
    result=graph.invoke({"messages": [request.messages],"stage":"greeting"},config=config)
    return result["messages"][-1]

