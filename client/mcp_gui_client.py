#!/usr/bin/env python3
"""
Gradio Web GUI for LLM統合MCPクライアント
"""

import os
import gradio as gr
from dotenv import load_dotenv
from datetime import datetime
from mcp_llm_client import LLMClient

load_dotenv()


class GUISession:
    """ブラウザセッションごとの状態管理"""

    def __init__(self):
        self.client: LLMClient | None = None
        self.initialized: bool = False
        self.status_messages: list[str] = []

    def _capture_status(self, msg: str):
        self.status_messages.append(msg)

    async def initialize(self):
        self.client = LLMClient(
            config_file="mcp_servers.json",
            status_callback=self._capture_status,
        )
        await self.client.initialize()
        self.initialized = True

    async def cleanup(self):
        if self.client:
            await self.client.cleanup()

    def get_tools_markdown(self) -> str:
        if not self.client:
            return "初期化中..."
        lines = []
        for server_name, tools in self.client.collector.tools_schema.items():
            lines.append(f"### {server_name}")
            for tool in tools:
                desc = tool["description"][:60]
                if len(tool["description"]) > 60:
                    desc += "..."
                lines.append(f"- **{tool['name']}**: {desc}")
            lines.append("")
        return "\n".join(lines) if lines else "ツールが見つかりません"

    def get_status_markdown(self) -> str:
        if not self.client:
            return "未接続"
        ctx = self.client.context
        duration = datetime.now() - ctx["session_start"]
        total_tools = sum(
            len(t) for t in self.client.collector.tools_schema.values()
        )
        return (
            f"**経過時間:** {str(duration).split('.')[0]}\n\n"
            f"**ツール実行:** {ctx['tool_calls']}回\n\n"
            f"**エラー:** {ctx['errors']}回\n\n"
            f"**サーバー:** {len(self.client.clients)}個\n\n"
            f"**ツール数:** {total_tools}個"
        )


async def on_app_load(session):
    """ブラウザセッション開始時に LLMClient を初期化"""
    new_session = GUISession()
    await new_session.initialize()
    return (
        new_session,
        new_session.get_tools_markdown(),
        new_session.get_status_markdown(),
        gr.update(interactive=True, placeholder="メッセージを入力（/help でコマンド一覧）"),
    )


async def respond(message, history, session):
    """チャットメッセージを処理してレスポンスを返す"""
    if not session.initialized:
        history.append({"role": "user", "content": message})
        history.append(
            {"role": "assistant", "content": "初期化中です。しばらくお待ちください..."}
        )
        return "", history, session, session.get_tools_markdown(), session.get_status_markdown()

    # スラッシュコマンド
    if message.startswith("/"):
        is_command, result = session.client._handle_command(message)
        if is_command:
            history.append({"role": "user", "content": message})
            if result == "__QUIT__":
                result = "セッションを終了します。ブラウザタブを閉じてください。"
            history.append({"role": "assistant", "content": result})
            return "", history, session, session.get_tools_markdown(), session.get_status_markdown()

    # ステータスバッファをクリア
    session.status_messages.clear()

    # LLMClient でクエリ処理
    response = await session.client.process_query(message)

    # ステータスメッセージがあれば折りたたみブロックで表示
    if session.status_messages:
        status_text = "\n".join(f"- {msg}" for msg in session.status_messages)
        full_response = (
            f"<details><summary>実行ログ</summary>\n\n{status_text}\n\n</details>\n\n{response}"
        )
    else:
        full_response = response

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_response})

    return "", history, session, session.get_tools_markdown(), session.get_status_markdown()


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="MCP LLMクライアント",
    ) as demo:
        session_state = gr.State(GUISession())

        gr.Markdown("# MCP LLM クライアント\n自然言語でMCPツールを操作するWebインターフェース")

        with gr.Row():
            # メインチャットエリア
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="チャット",
                    height=500,
                    buttons=["copy"],
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="初期化中...",
                        container=False,
                        scale=7,
                        interactive=False,
                    )
                    submit_btn = gr.Button("送信", scale=1, variant="primary")

                with gr.Row():
                    gr.Examples(
                        examples=[
                            "1+2を計算して",
                            "今日の天気を教えて",
                            "/tools",
                            "/status",
                        ],
                        inputs=msg_input,
                    )

            # サイドバー
            with gr.Column(scale=1):
                with gr.Accordion("利用可能なツール", open=True):
                    tools_display = gr.Markdown(value="初期化中...")

                with gr.Accordion("セッション情報", open=True):
                    status_display = gr.Markdown(value="未接続")

                gr.Markdown(
                    "---\n"
                    "**コマンド:**\n"
                    "- `/help` - ヘルプ\n"
                    "- `/tools` - ツール一覧\n"
                    "- `/status` - 状態表示\n"
                    "- `/history` - 履歴\n"
                    "- `/clear` - 履歴クリア\n"
                )

        # イベントハンドラ
        outputs = [msg_input, chatbot, session_state, tools_display, status_display]

        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, session_state],
            outputs=outputs,
        )
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, session_state],
            outputs=outputs,
        )

        # ライフサイクル
        demo.load(
            fn=on_app_load,
            inputs=[session_state],
            outputs=[session_state, tools_display, status_display, msg_input],
        )

    return demo


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] 環境変数 OPENAI_API_KEY を設定してください")
        exit(1)

    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
