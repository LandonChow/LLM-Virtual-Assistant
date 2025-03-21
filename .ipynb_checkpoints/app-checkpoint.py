import os
from typing_extensions import Annotated, TypedDict
from typing import Sequence

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, trim_messages, BaseMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

import streamlit as st

from config import config
from prompt import prompt_template

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

model = ChatOllama(model = "llama3.2", temperature=0.8)

workflow = StateGraph(state_schema=MessagesState)

trimmer = trim_messages(
    max_tokens=200,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

def call_model(state: State):
    #todo implement better memory management
    trimed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({"messages": trimed_messages})
    response = model.invoke(prompt)
    return {"messages": response}

def stream_wrapper(app_stream):
    for chunk, metadata in app_stream:
        if isinstance(chunk, AIMessage):
            yield chunk

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


st.title("Langchain Ollama")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    
    response = app.invoke(
        {"messages": st.session_state.messages}, 
        config,
        #stream_mode="messages",
    )
    msg = response["messages"][-1].content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    

    #todo implement streaming
    """
     stream = app.stream(
        {"messages": st.session_state.messages},
        config,
        stream_mode="messages",
    )
    st.session_state.messages.append({"role": "assistant", "content": stream})
    st.chat_message("assistant").write_stream(stream)   
    """






