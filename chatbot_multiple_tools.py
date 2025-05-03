#!/usr/bin/env python
# coding: utf-8

# In[2]:


##tools
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper


# In[4]:


api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, verbose=True,description="Query Arxiv papers")
print(arxiv.description)


# In[6]:


# arxiv.invoke("What is the latest ai tool?")


# In[7]:


api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki, verbose=True,description="Query Wikipedia")
# print(wiki.description)


# In[9]:


#wiki.invoke("Which is the latest ai model in openai tool?")


# In[10]:


#wiki.invoke("Who is the father of nation india ?")


# In[11]:


from dotenv import load_dotenv
import os
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


# In[12]:


from langchain_community.tools.tavily_search import TavilySearchResults
tavily = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), top_k_results=2, doc_content_chars_max=500)


# In[13]:


#tavily.invoke("What is the latest ai tool?")


# In[14]:


tools = [arxiv, wiki, tavily]


# In[21]:


from langchain_groq import ChatGroq
llm = ChatGroq(model="qwen-qwq-32b")


# In[23]:


#answer = llm.invoke("What is AI?")


# In[24]:


#answer = str(answer)


# In[26]:


# answer.replace("\n","")


# # In[27]:


# answer 


# In[28]:


llm_with_tools = llm.bind_tools(tools=tools)


# In[29]:


#llm_with_tools.invoke("WHat is the latest ai news?")


# In[30]:


#llm_with_tools.invoke("What is the latest research paper on AI?")


# In[33]:


#llm_with_tools.invoke("What is the India?")


# In[34]:


from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated
from langgraph.graph.message import add_messages


# In[35]:


class State(TypedDict):
    messages : Annotated[list[AnyMessage],add_messages]



# In[36]:


from IPython.display import Image,display
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


# In[37]:


def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}



# In[44]:


# graphbuilder = StateGraph(State)
# graphbuilder.add_node("tool_calling_llm",tool_calling_llm)
# graphbuilder.add_node("tools",ToolNode(tools))


# In[45]:


# graphbuilder.add_edge(START,"tool_calling_llm")
# graphbuilder.add_conditional_edges("tool_calling_llm",tools_condition)
# graphbuilder.add_edge("tools",END)


# In[60]:


graphbuilder2 = StateGraph(State)
graphbuilder2.add_node("tool_calling_llm",tool_calling_llm)
graphbuilder2.add_node("tools",ToolNode(tools))

graphbuilder2.add_edge(START,"tool_calling_llm")
graphbuilder2.add_conditional_edges("tool_calling_llm",tools_condition)
graphbuilder2.add_edge("tools","tool_calling_llm")
graphbuilder2.add_edge("tool_calling_llm",END)

graph2 = graphbuilder2.compile()
# display(Image(graph2.get_graph().draw_mermaid_png()))


# In[63]:


from langchain_core.messages import HumanMessage

# messages = graph2.invoke({"messages": "HI Good Morning?"})
# for m in messages['messages']:
#     m.pretty_print()


# In[47]:


# graph = graphbuilder.compile()


# In[49]:


# display(Image(graph.get_graph().draw_mermaid_png()))


# In[62]:


# from langchain_core.messages import HumanMessage

# messages = graph.invoke({"messages": "Hi Good Morning?"})
# for m in messages['messages']:
#     m.pretty_print()


# In[2]:


import streamlit as st 
st.header("Agentic AI")
st.write("Ask me anything about AI")
st.write("I can search Arxiv, Wikipedia and Tavily for you")
user_input = st.text_input("Enter your question:")  
if user_input:
    messages = graph2.invoke({"messages": user_input})
    for m in messages['messages']:
        st.write(m.content)


