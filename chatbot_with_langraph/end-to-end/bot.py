#importing libraries

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq

from typing import Annotated, Literal, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

import os
from dotenv import load_dotenv
load_dotenv()

class chatbot:
    def __init__(self):
        GROQ_API_KEY=os.getenv("GROQ_API_KEY")
        TAVILY_API_KEY=os.getenv("TAVILY_API_KEY") # print(GOOGLE_API_KEY)
        LANGCHAIN_API_KEY=os.getenv("LANGSMITH_API_KEY")
        LANGCHAIN_PROJECT=os.getenv("LANGSMITH_PROJECT")
        LANGCHAIN_TRACING_V2 = os.getenv("LANGSMITH_TRACING")
        LANGCHAIN_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
        os.environ["GOOGLE_API_KEY"] = GROQ_API_KEY
        os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
        os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
        os.environ["LANGCHAIN_PROJECT"]=LANGCHAIN_PROJECT
        
        self.llm = ChatGroq(model_name = 'gemma2-9b-it')
        # self.call_tool()

       # Prepare the tool node once
        tool = TavilySearchResults(max_results=2)
        self.tool_node = ToolNode(tools=[tool])
        self.llm_with_tool = self.llm.bind_tools([tool])

    def call_tool(self):
        tool = TavilySearchResults(max_results = 2)
        tools = [tool]
        self.tool_node = ToolNode(tools=tools)
        self.llm_with_tool = self.llm.bind_tools(tools)
       
       

    def call_model(self, state: MessagesState):
        messages = state['messages']
        response = self.llm_with_tool.invoke(messages)
        return { "messages" : [response]}
    
    
    def router_function(self, state: MessagesState) -> Literal['tools', END]:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    def __call__(self):
        
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        # workflow.add_node("tools", self.call_tool)
        workflow.add_edge(START, 'agent')
        workflow.add_conditional_edges('agent', self.router_function, { "tools": "tools", END:END})
        workflow.add_edge("tools", "agent")

        self.app = workflow.compile()
        return self.app
    
if __name__ == "__main__":
    mybot = chatbot()
    workflow = mybot()
    reponse = workflow.invoke({"messages": ["who is a current prime minister of UAE?"]})
    print(reponse['messages'][-1].content)