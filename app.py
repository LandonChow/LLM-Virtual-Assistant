import os
import uuid

from typing_extensions import Annotated, TypedDict
from typing import List, Sequence

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, trim_messages, BaseMessage, get_buffer_string, ToolMessage
from langchain_community.vectorstores import FAISS

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langgraph.prebuilt import ToolNode

import streamlit as st

from config import config
from prompt import prompt_template
from modules.vectorstore import db, save_path
from modules.stt import record_audio

class State(MessagesState):
    recall_memories: List[str]

#nodes
def call_model(state: State) -> State:
    #todo implement better memory management
    bound = prompt_template | model_with_tools
    recall = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    response = bound.invoke(
        {
            "messages": state["messages"],
            "recall_memories": recall,
        }
    )
    return {"messages": [response]}

def load_memory(state: State, config: RunnableConfig) -> State:
    human_messages = get_buffer_string(state["messages"])
    recall = search_recall_memories.invoke(human_messages, config)
    return {"recall_memories": recall}


#utils
def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id

#tools
@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save recall memories to build a comprehensive understanding to this user.
    Args:
        memory: describle user preferences in third person
    """
    user_id = get_user_id(config)
    document = Document(
        page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
    )
    db.add_documents([document])
    db.save_local(save_path)
    return memory

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)

    def _filter_function(metadata: dict) -> bool:
        return metadata["user_id"] == user_id

    documents = db.similarity_search(
        query, k=3, filter=_filter_function
    )
    if (documents != None):
        return [document.page_content for document in documents]
    else:
        return []

def route_tools(state: State):

    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"

    return END


#workflow
model = ChatOllama(model = "llama3.2", temperature=0.8)
tools = [save_recall_memory]
model_with_tools = model.bind_tools(tools)

trimmer = trim_messages(
    max_tokens=200,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

workflow = StateGraph(state_schema=State)
workflow.add_node("load_memory", load_memory)
workflow.add_node("model", call_model)
workflow.add_node("tools", ToolNode(tools))
#workflow.add_node("tools", tool_node)

workflow.add_edge(START, "load_memory")
workflow.add_edge("load_memory", "model")
workflow.add_conditional_edges("model", route_tools, ["tools", END])
workflow.add_edge("tools", "model")

#workflow.add_conditional_edges("model", should_continue, ["tools", END])
#workflow.add_edge("tools", "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

img = Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
with open("graph.png", "wb") as png:
    png.write(img.data)


#start

def pretty_print_stream_chunk(chunk):
    for node, updates in chunk.items():
        print(f"Update from node: {node}")
        if "messages" in updates:
            updates["messages"][-1].pretty_print()
            msg = updates["messages"][-1]
            if ((not isinstance(msg, ToolMessage)) and msg.content):
                print ("Message detected from node:" + node)
                st.session_state.messages.append({"role": "assistant", "content": msg.content})
                st.chat_message("assistant").write(msg.content)
        else:
            print(updates)
        print("\n")


# while True:
#     user_input = input("You: ")
#     response = app.stream(
#         {"messages": user_input}, 
#         config,
#     )
#     for chunk in response:
#         pretty_print_stream_chunk(chunk)

    

#streamlit

with st.sidebar:
    user_input = record_audio()

st.title("Langchain Ollama")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    
    response = app.stream(
        {"messages": st.session_state.messages}, 
        config,
        #stream_mode="messages",
    )
    for chunk in response:
        pretty_print_stream_chunk(chunk)



    # msg = response["messages"][-1].content
    # st.session_state.messages.append({"role": "assistant", "content": msg})
    # st.chat_message("assistant").write(msg)
    #st.chat_message("assistant").write(app.get_state(config=config))
    






