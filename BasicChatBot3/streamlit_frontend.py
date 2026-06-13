import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

#this is made so your current talking conversation dont remove when u ask new question
# thread/session config
thread_id = '1'
CONFIG = {'configurable': {'thread_id': thread_id}}

st.title('LangGraph Chatbot') # this will make a heading in the web page

# session state to hold chat history for display and st.session state bec streamlit run after evry execution
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = [] # a dictionay type container to store the message 

# render existing chat history(displaying the stored chat history again and agin bec it is reruning )
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# user input
user_input = st.chat_input('Type here')

if user_input:

    # add user message to history and display it
    st.session_state['message_history'].append({'role': 'user', 'content': user_input}) #this line store the user message 
    with st.chat_message('user'): # this display the user message 
        st.text(user_input)

    # invoke the chatbot (state persisted via checkpointer + thread_id)
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_input)]}, # sending the user message to the llm 
        config=CONFIG # what it do in this line 
    )
    ai_message = response['messages'][-1].content
    #-----------------X--------------------------
    #if this was the message from llm 
    # {
    #    "messages":[
    #       HumanMessage("Hi"),
    #       AIMessage("Hello")
    #      ]
    # }    
    # with help of above line we taking the last meassage which is the ai message  with -1 and .content so we can remove the meta data 
              

    # add ai message to history and display it updating the sesion state of ai with this code 
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)
    # with st.chat_message('assistant') → opens a chat bubble with the assistant icon
    # st.text(ai_message) → displays the full AI response text inside that bubble
