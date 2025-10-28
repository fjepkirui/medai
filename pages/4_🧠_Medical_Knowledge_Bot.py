import json

import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Medical Knowledge Bot", page_icon="🧠", layout="centered"
)

# Sidebar
st.sidebar.info(
    "Ask medical questions and get responses from a local AI model via Ollama (e.g., Mistral)."
)

# Title
st.title("🧠 Medical Knowledge Bot")


def ollama_stream(prompt: str, model: str = "tinyllama"):
    """
    Streams a response from an Ollama model.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            stream=True,
            timeout=30,
        )

        for line in response.iter_lines():
            if line:
                try:
                    data = line.decode("utf-8").removeprefix("data: ")
                    json_data = json.loads(data)
                    content_piece = json_data.get("message", {}).get("content", "")
                    yield content_piece
                except json.JSONDecodeError:
                    yield "\n[Error parsing stream: Invalid JSON]\n"
                    break
    except requests.exceptions.RequestException as e:
        yield f"\n[Connection error: {str(e)}]\n"


def save_feedback(index):
    """
    Saves thumbs feedback to the message at a given index.
    """
    st.session_state.history[index]["feedback"] = st.session_state.get(
        f"feedback_{index}"
    )


# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

## Display previous messages in chat history
for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            feedback = message.get("feedback", None)
            st.session_state[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
            )

# Get user input
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.history.append({"role": "user", "content": user_input})

    # Assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()
        FULL_RESPONSE = ""

        # Get response generator
        response_generator = ollama_stream(user_input)

        # Show spinner until the first chunk arrives
        with st.spinner("Thinking..."):
            try:
                first_chunk = next(response_generator)
            except StopIteration:
                first_chunk = "[No response received.]"

        # Start rendering streamed response
        FULL_RESPONSE += first_chunk
        response_container.markdown(FULL_RESPONSE, unsafe_allow_html=True)

        for chunk in response_generator:
            FULL_RESPONSE += chunk
            response_container.markdown(FULL_RESPONSE, unsafe_allow_html=True)

        # Store feedback
        assistant_index = len(st.session_state.history)
        FEEDBACK_KEY = f"feedback_{assistant_index}"
        st.feedback(
            "thumbs",
            key=FEEDBACK_KEY,
            on_change=save_feedback,
            args=(assistant_index,),
        )

    # Append assistant response to history
    st.session_state.history.append({"role": "assistant", "content": FULL_RESPONSE})
