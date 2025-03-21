from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os

save_path = "AppStorage"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
index = IndexFlatL2(len(embeddings.embed_query("hello world")))

db = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

if os.path.exists(save_path):
    db = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    print("Loaded user information from disk.")
else:
    print("No user information found.")



# history_retriver = db.as_retriever()

# history_retriver_tool = create_retriever_tool(
#     history_retriver,
#     "retrieve_past_conversation",
#     "Search and return most relevant past conversations based on current query",
# )
