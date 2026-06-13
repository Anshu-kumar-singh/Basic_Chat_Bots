import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# thread/session config
thread_id = '1'
CONFIG = {'configurable': {'thread_id': thread_id}}

st.title('LangGraph Chatbot')

# session state to hold chat history
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# render existing chat history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# user input
user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
    # this is the code for streaming 
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
