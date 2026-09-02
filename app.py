import streamlit as st
from langchain_groq import ChatGroq

from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper
)

from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun
)

from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv
import os

load_dotenv()


# --------------------------------------------------
# Wikipedia Tool
# --------------------------------------------------

api_wrapper_wiki = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=250
)

wiki = WikipediaQueryRun(
    api_wrapper=api_wrapper_wiki
)


# --------------------------------------------------
# Arxiv Tool
# --------------------------------------------------

api_wrapper_arxiv = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=250
)

arxiv = ArxivQueryRun(
    api_wrapper=api_wrapper_arxiv
)


# --------------------------------------------------
# DuckDuckGo Search Tool
# --------------------------------------------------

search = DuckDuckGoSearchRun(
    name="Search"
)


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("LangChain - Chat with Search")


# Sidebar
st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter your Groq API key:",
    type="password"
)


# --------------------------------------------------
# Chat History
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm a chatbot who can search the web. How can I help you?"
        }
    ]


# Display previous messages
for msg in st.session_state.messages:

    st.chat_message(msg["role"]).write(
        msg["content"]
    )


# --------------------------------------------------
# User Input
# --------------------------------------------------

if prompt := st.chat_input(
    placeholder="What is machine learning?"
):

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.chat_message("user").write(prompt)


    # --------------------------------------------------
    # LLM
    # --------------------------------------------------

    llm = ChatGroq(
        groq_api_key=api_key,
        model="openai/gpt-oss-20b",
        streaming=True
    )


    # --------------------------------------------------
    # Tools
    # --------------------------------------------------

    tools = [
        search,
        wiki,
        arxiv
    ]


    # --------------------------------------------------
    # Agent
    # --------------------------------------------------

    search_agent = create_react_agent(
        model=llm,
        tools=tools
    )


    # --------------------------------------------------
    # Invoke Agent
    # --------------------------------------------------

    with st.chat_message("assistant"):

        response = search_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )

        answer = response["messages"][-1].content

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.write(answer)