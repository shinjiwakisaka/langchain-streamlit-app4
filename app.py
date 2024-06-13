import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder


def create_agent_chain():
    chat = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=os.environ["OPENAI_API_TEMPERATURE"],
        streaming=True,
    )
    
    # OpenAI Functions AgentのプロンプトにMemoryの会話履歴を追加するための設定
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    # OpenAI Functions Agentが使える設定でMemoryを初期化
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    
    tools = load_tools(["ddg-search", "wikipedia"])
    return initialize_agent(
        tools,
        chat,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs=agent_kwargs,  # 追加
        memory=memory,              # 追加
    )
    
load_dotenv()

st.title("langchain-stremlit-app")

if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

if "messages" not in st.session_state:  # st.session_stateにmessagesがない場合
    st.session_state.messages = []      # st.session_state.messagesを空のリストで初期化
    
for message in st.session_state.messages:  # st.session_state.messagesでループ
    with st.chat_message(message["role"]): # ロールごとに
        st.markdown(message["content"])    # 保存されているテキストを表示

prompt = st.chat_input("What is up?")

if prompt:
    # ユーザの入力内容をst.session_state.messagesに追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):  # ユーザのアイコンで
        st.markdown(prompt)        # promptをマークダウンとして整形して表示
        
    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        # agent_chain = create_agent_chain()
        response = st.session_state.agent_chain.run(prompt, callbacks=[callback])
        st.markdown(response)
        
    with st.chat_message("assistant"):  # AIのアイコンで
        # response = "こんにちは"         # 固定の応答を用意して
        # st.markdown(response)           # 応答をマークダウンとして整形して表示
        chat = ChatOpenAI(
            model_name=os.environ["OPENAI_API_MODEL"],
            temperature=os.environ["OPENAI_API_TEMPERATURE"],
        )
        messages = [HumanMessage(content=prompt)]
        response = chat(messages)
        st.markdown(response.content)
        
    # 応答をst.session_state.messagesに追加
    st.session_state.messages.append({"role": "assistant", "content": response})