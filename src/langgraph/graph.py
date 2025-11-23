from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict,Literal,Annotated
import operator
from langchain.messages import AnyMessage,HumanMessage,SystemMessage,ToolMessage
from langgraph.graph import START,END,StateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel,Field
from tools import *
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
API_KEY=os.getenv("API_KEY_GPT")
MODEL_NAME=os.getenv("MODEL_NAME_GPT")
class user_info(BaseModel):
    username: str = Field(description="The name of the person. if he/she doesn't mention should be null")
    user_intent:Literal["writing","Grammar","vocabulary","None"]|None = Field(description="the Intent of user like writing, Grammar, vocabulary or if he/she doesn't mentioned directly should be None")
    ideal_score: int|None = Field(description="the score target by user if he/she doesn't mention directly 0")
    summary: str|None = Field(description="the summary of user message ")





base_model= ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    max_tokens=40

)
class UserIntent(TypedDict):
    user_intent=Literal["writing","Grammar","vocabulary"]
    summary=str
    ideal_score=int
    

class tutorials(TypedDict):
    user_id:str
    intent:UserIntent | None
    messages:list
    name:str 
    stage:Literal["greeting","writing","Grammar","vocabulary",END]
    user_id:str
    


writing_tools=[]
Grammar_tools=[correct_grammar,print_hello]
vocabulary_tools=[syn_ant,]
writing_model=base_model.bind_tools(writing_tools)
Grammar_model=base_model.bind_tools(Grammar_tools)
vocabulary_model=base_model.bind_tools(vocabulary_tools)
tools_by_name={tool.name:tool for tool in writing_tools+Grammar_tools+vocabulary_tools}
print(tools_by_name)
def writing(state:tutorials)->dict:
    if state["stage"]=="writing":
        if "role" in state["messages"][-1] :
            return state
        Message_content=state["messages"][-1]
        structured_output=base_model.with_structured_output(user_info)

        search_prompt=f"""
        In this part you should set your output based on what user exacly says
        search if user has changed his/her mind
        the intent of user should be one of writing, Grammar, vocabulary tasks

        Human message: {Message_content}
        """

        result=structured_output.invoke(search_prompt)
        print(result.user_intent)
        if result.user_intent=="writing"or result.user_intent==None or result.user_intent=="" or result.user_intent=="None":
            prompt=f"""
                    you are a helpfull persian assistant for IELTS in  {state["intent"]["user_intent"]} task
                    answer in farsi the question of user about {state['messages'][-1]}
                    """
            writing_result=base_model.invoke([SystemMessage(content=prompt)])
            state["messages"].append(writing_result.content)
            return state
        elif result.user_intent!="" and result.user_intent!=None and  result.user_intent!="None" :
            prompt=f"""
              you are a helpfull persian assistant for IELTS in  {state["intent"]["user_intent"]}
              you should talk farsi
              aware user that his/her stage has been changed from {state['stage']} to {result.user_intent}"""
            writing_result=base_model.invoke([SystemMessage(content=prompt)])
            state["messages"].append(writing_result.content)
            state["stage"]=result.user_intent
            state["intent"]["user_intent"]=result.user_intent
            return state


def Grammar(state:tutorials)->dict:
    
    if state["stage"]=="Grammar":
        if "role" in state["messages"][-1] :
            return state
        Message_content=state["messages"][-1]
        structured_output=base_model.with_structured_output(user_info)
        search_prompt=f"""

        In this part user has entered into {state["stage"]} 
        search if user has changed his/her mind or if he/she wants you to practice
        Human message: {Message_content}
        """

        result=structured_output.invoke(search_prompt,{"configurable": {"thread_id": "1"}})
        if result.user_intent=="Grammar" or result.user_intent==None or result.user_intent=="" or result.user_intent=="None":
            Grammar_prompt=f"""
                    you are a helpful persian assistant for IELTS. in  {state['intent']["user_intent"]}
                    you should talk in farsi.   
                    answer the question of user: {state['messages'][-1]}
                    if user wants you to correct grammar of his/her sentence use correct-grammar tool
                    """
            grammar_model_response=Grammar_model.invoke( Grammar_prompt)
            state["messages"].append(grammar_model_response.content)

            for toold_call in grammar_model_response.tool_calls:
                tool=tools_by_name[toold_call["name"]]
                observation=tool.invoke(toold_call["args"])
                state["messages"].append(ToolMessage(content=observation, tool_call_id=toold_call["id"]).content)
            return state
        elif result.user_intent!="" and result.user_intent!=None and  result.user_intent!="None" :
            
            prompt=f""" aware user that his/her stage has been changed from {state['stage']} to {result.user_intent}
                        you should speak in farsi"""
            grammar_result=base_model.invoke([SystemMessage(content=prompt)])
            state["messages"].append(grammar_result.content)
            state["stage"]=result.user_intent
            state["intent"]["user_intent"]=result.user_intent
            return state

def vocabulary(state:tutorials)->dict:
    if state["stage"]=="vocabulary":
        if "role" in state["messages"][-1] :
            return state
        Message_content=state["messages"][-1]
        structured_output=base_model.with_structured_output(user_info)
        search_prompt=f"""
        In this part user has entered into {state["stage"]} 
        search if user has changed his/her mind or if he/she wants you to practice
        Human message: {Message_content}
        """

        result=structured_output.invoke(search_prompt)
        if result.user_intent=="vocabulary"or result.user_intent==None or result.user_intent=="":
            vocab_prompt=f"""
                    you are a helpfull persian assistant for IELTS in  {state["intent"]["user_intent"]}
                    answer the question of user: {state['messages'][-1]}
                    if he/she wants you to create list of synonyms and antonyms of given word you can use syn-ant tool
                    """
            vocab_result=vocabulary_model.invoke([SystemMessage(content=vocab_prompt)])
            state["messages"].append(vocab_result.content)
            for toold_call in vocab_result.tool_calls:
                tool=tools_by_name[toold_call["name"]]
                observation=tool.invoke(toold_call["args"])
                state["messages"].append(ToolMessage(content=observation, tool_call_id=toold_call["id"]).content)
            return state
        elif result.user_intent!="" and result.user_intent!=None and  result.user_intent!="None" :
            prompt=f""" aware user that his/her stage has been changed from {state['stage']} to {result.user_intent}
                        you should speak in farsi"""
            vocab_result=base_model.invoke([SystemMessage(content=prompt)])
            state["messages"].append(vocab_result.content)
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
    Message_content=state["messages"][-1]
    structured_output=base_model.with_structured_output(user_info)
    search_prompt=f"""
    search for user intent, Ideal score, username, summary
    find this information when directly mentioned in message 
    user_intent should be one of writing, Grammar, vocabulary or None and until user don't mention it, you should not filled it
    ideal score can be analysis from message and should be int you must not use your imagination. when it mentioned by user clearly and some descriptions about it set this score, otherwise it should be 0
    the highest score is 9 and lowest score is 0
    Human message: {Message_content}
    """

    greeting_prompt_intent=f"""
    
    you are a helpful persian assistant for IELTS.
    you should talk in farsi.
    greeting to user and ask for user intent for practice like writing, Grammar, vocabulary or ideal score
    """


    prompt_score=f"""
    you are a helpful persian assistant for IELTS.
    you should talk in farsi.
     ask for user her/his ideal score
     the highest score is 9 and lowest score is 0
    """


    llm_answer=structured_output.invoke(search_prompt)
    if llm_answer.user_intent!="" and llm_answer.user_intent!=None and  llm_answer.user_intent!="None" :
        state["intent"]["user_intent"]=llm_answer.user_intent
        
    

        if state["intent"]["ideal_score"]==0 and llm_answer.ideal_score!=0:
            state["intent"]["ideal_score"]=llm_answer.ideal_score
            state["intent"]["summary"]=llm_answer.summary
            prompt_recieved_intent=f"""
            you are a helpful persian assistant for IELTS.
    you should talk in farsi.
            tanks user for sharing his/her information.
            ask user if he/she wants to pracitce {state["intent"]["user_intent"]} and is her/his ideal score {state["intent"]["ideal_score"]}?
            just to be sure
            """
            result=base_model.invoke([{"role":"user","content":prompt_recieved_intent}]) 
            state["messages"].append(result.content)

            state["stage"]=state["intent"]["user_intent"]
        else:
            result=base_model.invoke([{"role":"user","content":prompt_score}]) 
            state["messages"].append(result.content)
    elif llm_answer.ideal_score!=0:
        state["intent"]["ideal_score"]=llm_answer.ideal_score
        state["intent"]["summary"]=llm_answer.summary
        prompt_recieved_intent=f"""
        you are a helpful persian assistant for IELTS.
    you should talk in farsi.
        tanks user for sharing his/her information.
        ask user what does he/she want to pracitce ?
        """
        result=base_model.invoke([{"role":"user","content":prompt_recieved_intent}]) 
        state["messages"].append(result.content)
        state["stage"]=state["intent"]["user_intent"]

    else:
        result=base_model.invoke([{"role":"user","content":greeting_prompt_intent}]) 
        state["messages"].append(result.content)
        
    
    state['name']=llm_answer.username
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

    
    
