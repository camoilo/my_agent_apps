from openai import OpenAI
import streamlit as st
import os
import datetime
import json

def save_session():
    """保存会话数据"""
    if st.session_state.session_name and st.session_state.messages:
        session_data = {
            "assistant_name": st.session_state.assistant_name,
            "assistant_character": st.session_state.assistant_character,
            "session_name": st.session_state.session_name,
            "messages": st.session_state.messages
        }
        os.makedirs("streamlit_app/sessions", exist_ok=True)
        with open(f"streamlit_app/sessions/{st.session_state.session_name}.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

def load_sessions():
    """加载会话列表"""
    session_list = []
    if os.path.exists("streamlit_app/sessions"):
        with os.scandir("streamlit_app/sessions") as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".json"):
                    file_name = entry.name[:-5]
                    session_list.append(file_name)
    return sorted(session_list, reverse=True)   # 反转列表顺序，使最近的会话在前

def load_session(session):
    """加载会话数据"""
    try:
        if os.path.exists(f"streamlit_app/sessions/{session}.json"):
            with open(f"streamlit_app/sessions/{session}.json", "r", encoding="utf-8") as f:
                session_data = json.load(f)
            for key, value in session_data.items():
                st.session_state[key] = value
    except Exception as e:
        st.error(f"加载会话数据时出错：{e}")

def delete_session(session):
    """删除会话数据"""
    try:
        if os.path.exists(f"streamlit_app/sessions/{session}.json"):
            os.remove(f"streamlit_app/sessions/{session}.json")
        if session == st.session_state.session_name:
            st.session_state.clear()
    except Exception as e:
        st.error(f"删除会话数据时出错：{e}")

# 大模型配置
WorkspaceId = os.getenv("WorkspaceId")
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=f"https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
# 页面信息配置
st.set_page_config(
    page_title="AI智能助手",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={}
)
st.title("AI智能助手")
st.write("欢迎来到AI智能助手，在下方输入即可与其对话。")

# 会话状态初始化，用于记录对话相关的历史信息
if "assistant_name" not in st.session_state:
    st.session_state.assistant_name = "小优"
if "assistant_character" not in st.session_state:
    st.session_state.assistant_character = "活泼开朗，幽默风趣"
if "session_name" not in st.session_state:
    st.session_state.session_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
if "messages" not in st.session_state:
    st.session_state.messages = []
st.text(f"当前会话名称：{st.session_state.session_name}")

# 侧边栏信息
with st.sidebar:
    # 历史会话管理
    st.subheader("历史会话管理", text_alignment="center")
    if st.button("新建对话",width="stretch"):
        save_session()
        st.session_state.clear()
        st.rerun()
    st.text("历史会话",text_alignment="center")
    for session in load_sessions():
        col1, col2 = st.columns([5,1])
        with col1:
            if st.button(session, width="stretch", key=f"show_{session}",
                         type="primary" if session==st.session_state.session_name else "secondary"):
                load_session(session)
                st.rerun()
        with col2:
            if st.button("", key=f"delete_{session}", icon="✖️", width="stretch"):
                delete_session(session)
                st.rerun()
    st.divider()
    # 助手信息
    st.subheader("助手信息", text_alignment="center")
    assistant_name = st.text_input("助手名称", value=st.session_state.assistant_name, placeholder="请设置助手名称")
    if assistant_name:
        st.session_state.assistant_name = assistant_name
    assistant_character = st.text_area("助手性格", value=st.session_state.assistant_character, placeholder="请设置助手性格")
    if assistant_character:
        st.session_state.assistant_character = assistant_character

# 显示对话
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

system_prompt = f"""你叫{st.session_state.assistant_name},需要严格遵守规则来回复用户。
规则：
1. 确保回答内容简洁明了，避免冗长。
2. 匹配用户的语言。
3. 回复内容需符合设定的性格。
性格：
{st.session_state.assistant_character}
"""

prompt = st.chat_input("请输入信息")
if prompt:
    # 用户输入处理
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 调用大模型生成回复
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            *st.session_state.messages,
        ],
        stream=True,
    )
    # AI回复处理
    collection = st.empty()
    full_resp = ""
    for chunk in response:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            full_resp += content
            collection.chat_message("assistant").write(full_resp)
    st.session_state.messages.append({"role": "assistant", "content": full_resp})
    save_session()
    st.rerun()
