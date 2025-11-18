from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict,Literal,Annotated
import operator
from langchain.messages import AnyMessage,HumanMessage,SystemMessage
from langgraph.graph import START,END,StateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel,Field
from tools import *
from langgraph.types import Command

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
    stage:Literal["greeting","writing","Grammar","vocabulary",END]
    user_id:str
    


writing_tools=[]
Grammar_tools=[correct_grammar]
vocabulary_tools=[syn_ant]
writing_model=base_model.bind_tools(writing_tools)
Grammar_model=base_model.bind_tools(Grammar_tools)
vocabulary_model=base_model.bind_tools(vocabulary_tools)

def writing(state:tutorials)->dict:
    if state["stage"]=="writing":
        if "role" in state["messages"][-1] :
            return state
        print("in writing")
        Message_content=state["messages"][-1].content
        structured_output=base_model.with_structured_output(user_info)
        search_prompt=f"""
        In this part user has entered into {state["stage"]} 
        search if user has changed his/her mind or if he/she wants you to practice
        Human message: {Message_content}
        """

        result=structured_output.invoke([SystemMessage(content=search_prompt)])
        if result.user_intent=="writing":
            print("state in writing: ",state["stage"])
            prompt=f"""
                    answer the question of user about {state['messages'][-1].content}
                    """
            result=base_model.invoke([SystemMessage(content=prompt)])
            state["messages"].append({"role":"user","content":result})
            return state
        elif result.user_intent!="" and result.user_intent!=None and  result.user_intent!="None" :
            prompt=f"""
              you are a helpfull assistant for IELTS in  {state['messages'][-1].content}
              aware user that his/her stage has been changed from {state['stage']} to {result.user_intent}"""
            result=base_model.invoke([SystemMessage(content=prompt)])
            state["messages"].append({"role":"user","content":result})
            state["stage"]=result.user_intent
            state["intent"]["user_intent"]=result.user_intent
            return state


def Grammar(state:tutorials)->dict:
    
    if state["stage"]=="Grammar":
        # print("in grammar")
        # print("type message :   ",type(state["messages"][-1]))
        # print("message ::  ",state["messages"][-1]["role"])
        if "role" in state["messages"][-1] :
            return state
        Message_content=state["messages"][-1].content
        structured_output=base_model.with_structured_output(user_info)
        search_prompt=f"""
        In this part user has entered into {state["stage"]} 
        search if user has changed his/her mind or if he/she wants you to practice
        Human message: {Message_content}
        """

        result=structured_output.invoke([SystemMessage(content=search_prompt)])
        if result.user_intent=="Grammar":
            print("ok ok")
            prompt=f"""
                    you are a helpfull assistant for IELTS in  {state['messages'][-1].content}
                    answer the question of user {state['messages'][-1].content}
                    if he/she wants you to correct grammar of sentence you can use correct grammar tool 
                    """
            result=Grammar_model.invoke([SystemMessage(content=prompt)])
            print("tool call :",result.tool_calls)
            print("result  ",result)
            state["messages"].append({"role":"user","content":result})
            return state
        elif result.user_intent!="" and result.user_intent!=None and  result.user_intent!="None" :
            
            prompt=f""" aware user that his/her stage has been changed from {state['stage']} to {result.user_intent}"""
            result=base_model.invoke([SystemMessage(content=prompt)])
            state["messages"].append({"role":"user","content":result})
            state["stage"]=result.user_intent
            state["intent"]["user_intent"]=result.user_intent
            return state

def vocabulary(state:tutorials)->dict:
    if state["stage"]=="vocabulary":
        if "role" in state["messages"][-1] :
            return state
        print("vocabulary")
        Message_content=state["messages"][-1].content
        structured_output=base_model.with_structured_output(user_info)
        search_prompt=f"""
        In this part user has entered into {state["stage"]} 
        search if user has changed his/her mind or if he/she wants you to practice
        Human message: {Message_content}
        """

        result=structured_output.invoke([SystemMessage(content=search_prompt)])
        if result.user_intent=="vocabulary":
            prompt=f"""
                    you are a helpfull assistant for IELTS in  {state['messages'][-1].content}
                    answer the question of user {state['messages'][-1].content}
                    if he/she wants you to create list of synonyms and antonyms of given word you can use syn-ant tool
                    """
            result=vocabulary_model.invoke([SystemMessage(content=prompt)])
            state["messages"].append({"role":"user","content":result})
            return state
        elif result.user_intent!="" and result.user_intent!=None and  result.user_intent!="None" :
            prompt=f""" aware user that his/her stage has been changed from {state['stage']} to {result.user_intent}"""
            result=base_model.invoke([SystemMessage(content=prompt)])
            state["messages"].append({"role":"user","content":result})
            state["stage"]=result.user_intent
            state["intent"]["user_intent"]=result.user_intent
            return state


def greeting(state:tutorials)->dict:
    
    if state["stage"]=="Grammar":
        return state
    elif state["stage"]=="writing":
        return state
    elif state["stage"]=="vocabulary":
        return state
    state["stage"]="greeting"
    print("I am in: ",state["stage"])
    Message_content=state["messages"][-1].content
    structured_output=base_model.with_structured_output(user_info)
    search_prompt=f"""
    search for user intent, Ideal score, username, summary
    find this information when directly mentioned in message 
    user_intent should be one of writing, Grammar, vocabulary or None and until user don't mention it, you should not filled it
    ideal score can be analysis from message and should be int you must not use your imagination. when it mentioned by user clearly and some descriptions about it set this score, otherwise it should be 0
    Human message: {Message_content}
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
    if llm_answer.user_intent!="" and llm_answer.user_intent!=None and  llm_answer.user_intent!="None" :
        state["intent"]["user_intent"]=llm_answer.user_intent
        
    

        if state["intent"]["ideal_score"]==0 and llm_answer.ideal_score!=0:
            state["intent"]["ideal_score"]=llm_answer.ideal_score
            state["intent"]["summary"]=llm_answer.summary
            prompt_recieved_intent=f"""
            you are a helpful assistant for IELTS.
            tanks user for sharing his/her information.
            ask user if he/she wants to pracitce {state["intent"]["user_intent"]} and is her/his ideal score {state["intent"]["ideal_score"]}?
            just to be sure
            """
            result=base_model.invoke([{"role":"user","content":prompt_recieved_intent}]) 
            state["messages"].append({"role":"user","content":result})
            print("doneeeeeeeee")

            state["stage"]=state["intent"]["user_intent"]
            print(state["stage"],"     ",state["intent"]["user_intent"])
        else:
            result=base_model.invoke([{"role":"user","content":prompt_score}]) 
            state["messages"].append({"role":"user","content":result})
            state["stage"]=END
    elif llm_answer.ideal_score!=0:
        state["intent"]["ideal_score"]=llm_answer.ideal_score
        state["intent"]["summary"]=llm_answer.summary
        prompt_recieved_intent=f"""
        you are a helpful assistant for IELTS.
        tanks user for sharing his/her information.
        ask user what does he/she want to pracitce ?
        """
        result=base_model.invoke([{"role":"user","content":prompt_recieved_intent}]) 
        state["messages"].append({"role":"user","content":result})
        state["stage"]=state["intent"]["user_intent"]

    else:
        result=base_model.invoke([{"role":"user","content":greeting_prompt_intent}]) 
        state["messages"]=[{"role":"user","content":result}]
        print("helllooooooooo")
        state["stage"]=END
    state['name']=llm_answer.username
    print("stage eeee :",state["stage"])

        
    return state



def create_workflow():

    workflow=StateGraph(tutorials)
    workflow.add_node("greeting",greeting)
    workflow.add_node("writing",writing)
    workflow.add_node("Grammar",Grammar)
    workflow.add_node("vocabulary",vocabulary)

    workflow.add_edge(START,"greeting")
    workflow.add_edge("greeting",END)
    workflow.add_edge("greeting","writing")
    workflow.add_edge("greeting","Grammar")
    workflow.add_edge("greeting","vocabulary")
    workflow.add_edge("vocabulary",END)
    workflow.add_edge("writing",END)
    workflow.add_edge("Grammar",END)

    

    app=workflow.compile()
    return app
# message=input("start a chat...")
# start_state=tutorials({"messages":[HumanMessage(content=message)]})
# result=app.invoke(start_state)
# print(result)
# print(len(result["messages"]))

    
    
