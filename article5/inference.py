from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import os
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory1.db")


load_dotenv()

os.environ["COHERE_API_KEY"]  = os.getenv("cohere_api_key") 
model = ChatCohere()


system_template = "You are an helpful coding assistant that can rely on this code: {context} and on the previous message history as context, and from that you build a context and history-aware reply to this (DO NOT mention the fact that you are starting from a code snippet):"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")]
)

chain = prompt_template | model

runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def infer_reply(context, text, sessionid):
    global chain
    r = runnable_with_history.invoke(
        {"context": context, "input": text},
        config={"configurable": {"session_id": sessionid}}
    )
    return r.content

