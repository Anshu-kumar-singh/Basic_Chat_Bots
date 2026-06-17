from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq                          # ← changed from openai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ===========================================================
# LLM MEMORY — import InMemoryStore
# InMemoryStore stores facts across the entire app lifetime
# (all threads share the same store object)
# ===========================================================
from langgraph.store.memory import InMemoryStore

import requests

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=os.getenv("GROQ_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
_ddg = DuckDuckGoSearchAPIWrapper(region="us-en")

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Use this for current events or general knowledge questions."""
    return _ddg.run(query)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


# ===========================================================
# LLM MEMORY — Tool 1: save_memory
# The LLM calls this tool when it wants to remember something
# about the user (name, preferences, facts etc.)
# Facts are stored in InMemoryStore under the user's thread_id
# ===========================================================
@tool
def save_memory(fact: str, thread_id: Optional[str] = None) -> str:
    """
    Save an important fact about the user to long-term memory.
    Call this when the user shares personal info, preferences, or
    anything worth remembering across conversations.
    Example facts: 'User's name is Anshu', 'User prefers Python over Java'
    """
    if not thread_id:
        return "Cannot save memory: no thread_id provided."

    # namespace groups memories by user (thread_id acts as user id here)
    namespace = ("user_memory", thread_id)

    # each fact gets a unique key based on how many facts exist already
    existing = memory_store.search(namespace)
    key = f"fact_{len(existing)}"

    memory_store.put(namespace, key, {"fact": fact})
    return f"Saved to memory: {fact}"


# ===========================================================
# LLM MEMORY — Tool 2: get_memories
# The LLM calls this tool at the start of a conversation
# to recall what it knows about the user
# ===========================================================
@tool
def get_memories(thread_id: Optional[str] = None) -> str:
    """
    Retrieve all saved facts about the user from long-term memory.
    Call this at the beginning of a conversation to recall user preferences.
    """
    if not thread_id:
        return "No thread_id provided."

    namespace = ("user_memory", thread_id)
    memories = memory_store.search(namespace)

    if not memories:
        return "No memories saved for this user yet."

    # join all facts into a readable string for the LLM
    facts = [item.value["fact"] for item in memories]
    return "Known facts about the user:\n" + "\n".join(f"- {f}" for f in facts)


tools = [web_search, get_stock_price, calculator, rag_tool, save_memory, get_memories]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF.\n\n"
            # ===========================================================
            # LLM MEMORY — instructions added to system prompt
            # Tells the LLM when to use the memory tools
            # ===========================================================
            "MEMORY INSTRUCTIONS:\n"
            "- At the start of each conversation, call `get_memories` with the "
            f"thread_id `{thread_id}` to recall what you know about the user.\n"
            "- Whenever the user shares personal info (name, preferences, goals), "
            "call `save_memory` to store it for future conversations.\n"
            "- Use recalled memories to personalise your responses."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer (SQLite — short term, per session)
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ===========================================================
# LLM MEMORY — InMemoryStore (long term, across sessions)
# checkpointer  → saves message history per thread (short term)
# memory_store  → saves user facts across ALL threads (long term)
# ===========================================================
memory_store = InMemoryStore()

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# ===========================================================
# LLM MEMORY — pass store= to compile so LangGraph knows
# about the long-term store alongside the checkpointer
# ===========================================================
chatbot = graph.compile(checkpointer=checkpointer, store=memory_store)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


# ===========================================================
# LLM MEMORY — helper so frontend can show saved memories
# in the sidebar (optional but useful for demo/debugging)
# ===========================================================
def get_user_memories(thread_id: str) -> list[str]:
    """Return all saved memory facts for a thread as a plain list."""
    namespace = ("user_memory", str(thread_id))
    items = memory_store.search(namespace)
    return [item.value["fact"] for item in items]