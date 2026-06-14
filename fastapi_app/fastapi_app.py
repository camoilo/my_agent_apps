from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse
from starlette.requests import Request
from pydantic import BaseModel
import os
import datetime
import json
from typing import Any
from openai import OpenAI
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

app = FastAPI(title="AI猜谜")
app.mount("/static", StaticFiles(directory="fastapi_app/static"), name="static")

SYSTEM_PROMPT = """
    你是一个负责出题并把控节奏的游戏主持人。你的任务是通过文字与用户进行猜谜语互动游戏。
    一、核心规则：
    1.使用生动、富有画面感的语言。避免机械式问答，保持对话的沉浸感。
    2.迷题内容包括但不限于：常见物品、动物、字谜、抽象概念。
    3.每次只出一个谜语，等待用户回答后再进行反馈。严禁一次性抛出多个谜题。
    4.在用户猜中或主动要求揭晓答案前，绝不在回复中透露谜底或包含谜底的关键字。
    5.若用户回答正确，则给予夸奖，解释谜面与谜底的关联，然后询问是否继续；若用户回答错误但接近，则给予肯定并给出提示。若用户回答完全错误，则温和纠正，给出一个方向性提示，鼓励再试。
    6.当用户说“提示”、“给个线索”时，提供一个新的角度描述，而不是重复之前的谜面。当用户说“放弃”、“答案”时，才揭晓谜底。
    二、交互流程：
    1.开场先简单提示游戏开始，然后给出第一个谜面。
    2.接收猜测 -> 判定 -> 反馈/提示 -> 等待下一次输入。
    3.每轮游戏结束后（如猜对或揭晓后），简短总结并询问是否再来一局或者是否需要改变难度。
    三、约束：
    1.不要出涉及政治、暴力等敏感或不合时宜话题的谜语。
    2.如果用户聊起与游戏无关的话题，礼貌地将话题拉回猜谜游戏，或者陪聊两句后自然过渡回游戏。
    3.保持耐心，即使用户多次猜错也不要表现出烦躁。
    4.谜底必须是客观存在或有公认答案的事物，避免自创无解之谜。
    四、语气示例：
    "差一点点就摸到真相了！试着想想，它在厨房里很常见，但不是用来吃的哦~"  # 正确
    "不对。答案是锅铲。"  # 错误
    "妙啊！🎉答案正是【影子】！要不要趁热打铁再来一题？"  # 正确
"""

# 大模型配置
WorkspaceId = os.getenv("WorkspaceId")
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=f"https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

if not os.path.exists("fastapi_app/sessions"):
    os.makedirs("fastapi_app/sessions")

def generate_session_id():
    """根据时间生成会话ID"""
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def get_session_path(session_id):
    """获取会话文件路径"""
    return f"fastapi_app/sessions/{session_id}.json"

# API响应
class ApiResponse(BaseModel):
    code: int
    message: str
    data: Any

# 聊天请求
class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/")
def root():
    """访问首页"""
    logging.info("访问首页")
    return FileResponse("static/index.html")

@app.post("/api/session")
def create_session():
    """创建一个新的会话"""
    logging.info("创建新的会话")
    session_id = generate_session_id()
    session_data = {
        "session_id": session_id,
        "messages": []
    }
    # 创建初始的会话文件
    with open(f"fastapi_app/sessions/{session_id}.json", "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    return ApiResponse(code=200,
        message="会话创建成功",
        data=session_id
        )

@app.post("/api/chat")
def chat(request: ChatRequest):
    """处理用户消息并生成AI响应"""
    logging.info(f"处理用户消息：{request.session_id}:{request.message}")
    # 加载json文件中的会话数据
    session_path = get_session_path(request.session_id)
    with open(session_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(session_data["messages"])
    messages.append({"role": "user", "content": request.message})
    logging.info(f"请求的会话信息：{messages}")
    # 调用AI模型生成响应
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        stream=False,
        temperature=1.3,
    )
    chat_resp = response.choices[0].message.content
    logging.info(f"AI回复：{chat_resp}")
    # 更新会话数据并保存
    messages.append({"role": "assistant", "content": chat_resp})
    messages.pop(0) # 移除系统提示词（不保存）
    session_data["messages"] = messages
    logging.info(f"更新的会话信息：{session_data}")
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)
    # 返回响应
    return ApiResponse(code=200,
        message="请求成功",
        data=chat_resp
        )

@app.get("/api/session")
def get_session_list():
    """获取会话列表"""
    logging.info("获取会话列表")
    session_list = []
    with os.scandir("fastapi_app/sessions") as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".json"):
                file_name = entry.name[:-5]
                session_list.append(file_name)
    return ApiResponse(code=200,
        message="会话列表获取成功",
        data=sorted(session_list, reverse=True)
        )

@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    """获取指定会话的聊天记录"""
    logging.info(f"获取会话：{session_id}")
    session_path = get_session_path(session_id)
    with open(session_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)
    return ApiResponse(code=200,
        message="会话获取成功",
        data=session_data
        )

@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    logging.info(f"删除会话：{session_id}")
    """删除指定会话"""
    session_path = get_session_path(session_id)
    if os.path.exists(session_path):
        os.remove(session_path)
    return ApiResponse(code=200,
        message="会话删除成功",
        data=None
        )

@app.exception_handler(Exception)
def handle_exception(request: Request, exc: Exception):
    """异常处理，捕获所有异常"""
    logging.error(f"处理异常，请求路径：{request.url}，异常值：{exc}")
    return JSONResponse(content={"code": 500, "message": "服务器错误", "data": None})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
