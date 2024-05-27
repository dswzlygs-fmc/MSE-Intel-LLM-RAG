import json
import requests
import streamlit as st
from streamlit_chat import message


st.set_page_config(
    page_title="ChatApp",
    page_icon=" ",
    layout="wide",
)
st.title(" Neo4j 查询辅助工具 ")


# 给对话增加history属性，将历史对话信息储存下来
if "history" not in st.session_state:
    st.session_state.history = []

# 显示历史信息
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 这里是你的大模型生成的回复
# 需要根据自己的情况进行实现
# 我这里不仅想要显示大模型的回复，还想展示检索信息，所以将检索的内容也一起返回
def get_response_material(query, history):
    
    # 构建请求的URL        
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": "qwen:7b",
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "stream": False
    }
    headers = {
        "Content-Type": "",
        "content-type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    
    # 假设API返回的是一个JSON对象，其中包含'response'和'material'两个字段
    api_response = response.json()
    
    # 提取所需的值
    chat_response = api_response.get('message')
    material = api_response.get('model')
    
    # 返回两个值
    return chat_response, material


def get_response_Neo4jNote(query, history):
   
    # 构建请求的URL        
    url = "http://127.0.0.1:8000/predict"

    payload = {
        "data": query
    }

    response = requests.request("POST", url, json=payload)
    
    # 假设API返回的是一个JSON对象，其中包含'response'和'material'两个字段
    api_response = response.json()
    
    # 提取所需的值
    prediction = api_response.get('prediction')
    chat_response = "Result: \n" + prediction.get('result')
    material = "Ollama+Neo4j"
    
    # 返回两个值
    return chat_response, material


# user_input接收用户的输入
if user_input := st.chat_input("Chat with Alex: "):
    # 在页面上显示用户的输入
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # get_response_material用来获取模型生成的回复，这个是需要根据自己的情况去实现
    # response为大模型生成的回复，material为RAG的检索的内容
    if user_input.startswith("Neo4j") :
        response, material = get_response_Neo4jNote(user_input, st.session_state.history)    
    else:
        response, material = get_response_material(user_input, st.session_state.history)    

    # 使用一个左侧框，展示检索到的信息，如果不需要显示检索信息删掉即可
    with st.sidebar:
        st.markdown(':balloon::tulip::cherry_blossom::rose: :green[**检索内容：**] :hibiscus::sunflower::blossom::balloon:')
        st.text(material)
    
    # 将用户的输入加入历史
    st.session_state.history.append({"role": "user", "content": user_input})
    # 在页面上显示模型生成的回复
    with st.chat_message("assistant"):
        st.markdown(response)
    # 将模型的输出加入到历史信息中
    st.session_state.history.append({"role": "assistant", "content": response})
    
    # 只保留十轮对话，这个可根据自己的情况设定，我这里主要是会把history给大模型，context有限，轮数不能太多
    if len(st.session_state.history) > 20:
        st.session_state.messages = st.session_state.messages[-20:]