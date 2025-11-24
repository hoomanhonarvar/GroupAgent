from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict,Literal,Annotated
import operator
from langchain.messages import AnyMessage,HumanMessage,SystemMessage,ToolMessage
from langgraph.graph import START,END,StateGraph
from pydantic import BaseModel,Field
from tools import *


load_dotenv()
API_KEY=os.getenv("API_KEY_OSS")
MODEL_NAME=os.getenv("MODEL_NAME_OSS")
MODEL_URL=os.getenv("BASE_URL_OSS")

class user_info(BaseModel):
    user_intent:Literal["writing","Grammar","vocabulary","None"]|None = Field(description="the Intent of user like writing, Grammar, vocabulary or if he/she doesn't mentioned directly should be None")





base_model= ChatOpenAI(
    model=MODEL_NAME,
    base_url=MODEL_URL,
    api_key=API_KEY,

)   
class tutorials(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    stage:Literal["greeting","writing","Grammar","vocabulary"]
    


writing_tools=[]
Grammar_tools=[correct_grammar]
vocabulary_tools=[syn_ant,]
writing_model=base_model.bind_tools(writing_tools)
Grammar_model=base_model.bind_tools(Grammar_tools)
vocabulary_model=base_model.bind_tools(vocabulary_tools)
tools_by_name={tool.name:tool for tool in writing_tools+Grammar_tools+vocabulary_tools}

def writing(state:tutorials)->dict:
    print(state["stage"])
    print("hello you are in writing")
    prompt=f"""you are helpfull writing assistent.
            aware user that has enterned into {state["stage"]} stage
            answer user question in farsi

            """
    return {
        "messages":[writing_model.invoke([SystemMessage(
                            content=prompt
                        )]+state["messages"]
                        )
        ],
    }
    

def tool_node(state: dict):
    print("you are in tool call")
    result=[]
    for tool_call in state["messages"][-1].tool_calls:
        print("find a tool calling",tool_call["name"])
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
   
    return {"messages": result}


def Grammar(state:tutorials)->dict:
    print(state["stage"])
    print("hello you are in grammar")
    prompt=f"""you are helpfull grammar assistent.
            aware user that has enterned into {state["stage"]} stage
            answer user question in farsi
            you should call correc_grammar tool if user wants you to correct grammar

            """
    result=Grammar_model.invoke([SystemMessage(
                            content=prompt    
                        )]+state["messages"]
    )
    print(result)
    return {
        "messages":[Grammar_model.invoke([SystemMessage(
                            content=prompt
                        )]+state["messages"]
                        )
        ],
    }
def vocabulary(state:tutorials)->dict:
    print(state["stage"])
    print("hello you are in vocab")
    prompt=f"""you are helpfull vocabulary assistent.
            aware user that has enterned into {state["stage"]} stage
            answer user question in farsi

            """
    return {
        "messages":[vocabulary_model.invoke([SystemMessage(
                            content=prompt
                        )]+state["messages"]
                        )
        ],
    }
    

def greeting(state:tutorials)->dict:

    print("hello you are in greeting")
    Message_content=state["messages"][-1]
    structured_output=base_model.with_structured_output(user_info)
    search_prompt=f"""
    search for user intent 
    user_intent should be one of writing, Grammar, vocabulary or None and until user  
    Human message: {Message_content}
    """

    greeting_prompt_intent=f"""
    
    you are a helpful persian assistant for IELTS.
    you should talk in farsi.
    greeting to user and ask for user intent for practice like writing, Grammar, vocabulary or ideal score
    """

    llm_answer=structured_output.invoke(search_prompt)
    print("user intent is",llm_answer.user_intent)
    if llm_answer.user_intent!="" and llm_answer.user_intent!=None and  llm_answer.user_intent!="None" :
        print("user intent is",llm_answer.user_intent)
        return {
            "stage":llm_answer.user_intent
        }
    else:
        return {
            "messages": [
                base_model.invoke(
                    [
                        SystemMessage(
                            content=greeting_prompt_intent
                        )
                    ]
                    + state["messages"]
                )
            ],
        }
    

def router(state) -> Literal["writing", "Grammar", "vocabulary",END]:
    print(state["stage"])
    stage = state["stage"]
    print(stage)
    if stage == "writing":
        return "writing"
    elif stage == "Grammar":
        return "Grammar"
    elif stage == "vocabulary":
        return "vocabulary"
    else:
        return END

def create_workflow():
    
    workflow=StateGraph(tutorials)
    workflow.add_node("greeting",greeting)
    workflow.add_node("writing",writing)
    workflow.add_node("Grammar",Grammar)
    workflow.add_node("vocabulary",vocabulary)
    workflow.add_node("tool_node",tool_node)

    workflow.add_edge(START,"greeting")
    workflow.add_edge("greeting",END)
    workflow.add_conditional_edges("greeting",
    router,  
    {
        "writing": "writing",
        "Grammar": "Grammar",
        "vocabulary": "vocabulary",
        END: END
    }
)
    workflow.add_edge("vocabulary","tool_node")
    workflow.add_edge("Grammar","tool_node")
    workflow.add_edge("vocabulary",END)
    workflow.add_edge("writing",END)
    workflow.add_edge("Grammar",END)
    workflow.add_edge("tool_node",END)
    
    return workflow

    
    
