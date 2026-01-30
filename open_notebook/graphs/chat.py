import sqlite3
from typing import Annotated, Optional
from ai_prompter import Prompter
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from open_notebook.ai.provision import provision_langchain_model
from open_notebook.config import LANGGRAPH_CHECKPOINT_FILE
from open_notebook.domain.notebook import Notebook
from open_notebook.utils import clean_thinking_content

class ThreadState(TypedDict):
    messages: Annotated[list, add_messages]
    notebook: Optional[Notebook]
    context: Optional[str]
    context_config: Optional[dict]
    model_override: Optional[str]


async def call_model_with_messages(state: ThreadState, config: RunnableConfig) -> dict:
    system_prompt = Prompter(prompt_template="chat/system").render(data=state)
    payload = [SystemMessage(content=system_prompt)] + state.get("messages", [])
    model_id = config.get("configurable", {}).get("model_id") or state.get("model_override")

    # Provisionamos el modelo (aquí se mantenía el error antes)
    model = await provision_langchain_model(str(payload), model_id, "chat", max_tokens=8192)

    # Forzamos que no haya streaming para mayor estabilidad
    if hasattr(model, "streaming"):
        model.streaming = False

   
    ai_message = await model.ainvoke(payload)

    content = ai_message.content if isinstance(ai_message.content, str) else str(ai_message.content)
    cleaned_content = clean_thinking_content(content)
    cleaned_message = ai_message.model_copy(update={"content": cleaned_content})

    return {"messages": cleaned_message}

conn = sqlite3.connect(LANGGRAPH_CHECKPOINT_FILE, check_same_thread=False)
memory = SqliteSaver(conn)
agent_state = StateGraph(ThreadState)
agent_state.add_node("agent", call_model_with_messages)
agent_state.add_edge(START, "agent")
agent_state.add_edge("agent", END)
graph = agent_state.compile(checkpointer=memory)
