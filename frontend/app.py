"""Streamlit chat UI for the US population assistant. It only talks to the web API (POST /chat).

make api        # in one terminal
make frontend   # in another
"""

import os

import httpx
import streamlit as st
from dotenv import dotenv_values

# Read only API_URL from .env, so the other secrets there do not reach this process.
API_URL = os.environ.get("API_URL") or dotenv_values(".env").get("API_URL") or "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 150  # a bit more than the API's own turn limit
EXAMPLES = [
    "How many people live in Travis County, Texas?",
    "What is the poverty rate in Mississippi?",
    "Which 10 states have the highest share of renters?",
    "Compare the share of adults with a bachelor's degree in Colorado vs the whole US",
]

st.set_page_config(page_title="US Population Assistant", page_icon="📊")


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.thread_id = None


def ask(question: str, api_key: str) -> str:
    """Send one question to the API and return the answer; raises RuntimeError with a message for the user."""
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        response = httpx.post(
            f"{API_URL}/chat",
            json={"message": question, "thread_id": st.session_state.thread_id},
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError("The request timed out. Please try again.") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError("Cannot reach the service. Please try again later.") from exc

    if response.status_code == 401:
        raise RuntimeError("Invalid or missing access key. Enter it in the sidebar.")
    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        raise RuntimeError(detail if isinstance(detail, str) else "Something went wrong. Please try again.")

    body = response.json()
    st.session_state.thread_id = body["thread_id"]
    return body["answer"]


if "messages" not in st.session_state:
    reset_chat()

with st.sidebar:
    st.header("Access")
    api_key = st.text_input("Access key", type="password", help="The key you received with the link.")
    st.button("New conversation", on_click=reset_chat, use_container_width=True)
    st.caption("Data: American Community Survey (ACS) 2020 5-year estimates, US Census Bureau.")

st.title("US Population Assistant")
st.write("Ask questions about the US population, households, income, housing, education and more.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about US Census data")

examples_slot = st.empty()
if not st.session_state.messages:
    with examples_slot.container():
        st.caption("Or try one of these:")
        for example in EXAMPLES:
            if st.button(example, key=example):
                question = example

if question:
    examples_slot.empty()  # the examples were drawn before this run knew a question was coming
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Looking up the data..."):
                answer = ask(question, api_key)
        except RuntimeError as exc:
            st.error(str(exc))
            st.session_state.messages.pop()  # the question got no answer, so it does not enter the history
        else:
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
