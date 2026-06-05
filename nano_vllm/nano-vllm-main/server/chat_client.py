import requests
import json

# 你的本地服务器地址
API_URL = "http://127.0.0.1:8000/v1/chat/completions"


def main():
    print("🚀 本地大模型终端已启动！(输入 'exit' 或 'quit' 退出)\n")

    # messages 列表用于保存上下文，让模型拥有“记忆”
    messages = []

    while True:
        # 1. 获取你的输入
        user_input = input("🧑 你: ")

        if user_input.lower() in ['exit', 'quit']:
            print("👋 再见！")
            break

        if not user_input.strip():
            continue

        # 将你的新消息追加到历史记录中
        messages.append({"role": "user", "content": user_input})

        # 2. 构造请求体，关闭流式 (stream: False)
        payload = {
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 512,
            "stream": False  # 🌟 关键：强迫服务器算完所有字再返回
        }

        # 3. 发送请求并进入“死等”状态
        try:
            # 这里的 request.post 会一直阻塞当前 Python 线程，直到收到完整响应
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # 检查 HTTP 状态码是否为 200

            # 4. 解析完整的 JSON 返回值
            result = response.json()

            # 从 OpenAI 格式的 JSON 中提取助手的最终回复
            assistant_reply = result["choices"][0]["message"]["content"]

            print(f"🤖 助手: {assistant_reply}\n")

            # 5. 将助手的回复也追加到历史记录中，形成完整的对话链
            messages.append({"role": "assistant", "content": assistant_reply})

        except requests.exceptions.RequestException as e:
            print(f"❌ 网络请求失败，请检查服务器是否启动: {e}\n")
            # 如果请求失败，把刚才加进去的废消息撤销掉
            messages.pop()


if __name__ == "__main__":
    main()