#!/usr/bin/env python3
"""
MCP Agent - Interactive Dialogue Engine
対話的なMCPエージェント（簡略化版）

主な機能:
- MCPサーバーへの接続とツール実行
- LLMによるタスク分解と実行判断
- 対話的なREPLインターフェース
"""

import asyncio
import os
from typing import List, Dict, Any, Optional

from config_manager import ConfigManager, Config
from connection_manager import ConnectionManager
from llm_interface import LLMInterface
from utils import Logger, safe_str


class MCPAgent:
    """対話型MCPエージェント"""

    def __init__(self, config_path: str = "config.yaml", mcp_servers_path: str = "mcp_servers.json"):
        # 設定読み込み
        self.config = ConfigManager.load(config_path)
        self.verbose = self.config.development.verbose
        self.logger = Logger(verbose=self.verbose)

        # マネージャー初期化
        self.connection_manager = ConnectionManager(
            config_file=mcp_servers_path,
            verbose=self.verbose
        )

        self.llm_interface = LLMInterface(
            config=self.config,
            logger=self.logger
        )

        # 会話履歴
        self.conversation_history: List[Dict] = []

    async def initialize(self):
        """エージェントの初期化"""
        await self.connection_manager.initialize()

    async def process_request(self, user_query: str) -> str:
        """
        ユーザーリクエストを処理

        Args:
            user_query: ユーザーの入力

        Returns:
            エージェントの応答
        """
        # 会話履歴に追加
        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })

        # コンテキスト生成
        context = self._get_recent_context()
        tools_info = self.connection_manager.format_tools_for_llm()

        # 実行方式判定
        execution_type = await self.llm_interface.determine_execution_type(
            user_query, context, tools_info
        )

        exec_type = execution_type.get("type", "TOOL")

        # NO_TOOLの場合は直接応答
        if exec_type == "NO_TOOL":
            response = execution_type.get("response", "")
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            return response

        # CLARIFICATIONの場合
        if exec_type == "CLARIFICATION":
            clarification = execution_type.get("clarification", {})
            question = clarification.get("question", execution_type.get("response", "追加情報が必要です"))
            self.conversation_history.append({
                "role": "assistant",
                "content": question
            })
            return question

        # ツール実行が必要な場合
        task_list = await self.llm_interface.generate_task_list(
            user_query, context, tools_info
        )

        if not task_list:
            response = "タスクを生成できませんでした。別の方法でお手伝いしましょうか？"
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            return response

        # タスクを実行
        results = await self._execute_tasks(task_list)

        # 結果を解釈
        response = await self.llm_interface.interpret_results(
            user_query, results, context
        )

        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    async def _execute_tasks(self, task_list: List[Dict]) -> List[Dict]:
        """タスクリストを実行"""
        results = []

        for i, task in enumerate(task_list):
            tool = task.get("tool", "")
            params = task.get("params", {})
            description = task.get("description", tool)

            self.logger.ulog(f"\n[ステップ {i+1}/{len(task_list)}] {description}", "info")
            self.logger.ulog(f"  -> {tool} を実行中...", "info")

            try:
                result = await self.connection_manager.call_tool(tool, params)

                # 結果を文字列に変換
                if hasattr(result, 'content'):
                    result_text = ""
                    for content_item in result.content:
                        if hasattr(content_item, 'text'):
                            result_text += content_item.text
                    result = result_text

                results.append({
                    "tool": tool,
                    "description": description,
                    "result": safe_str(result),
                    "success": True
                })

                self.logger.ulog(f"[完了] {description}", "info:completed")

            except Exception as e:
                error_msg = safe_str(str(e))
                results.append({
                    "tool": tool,
                    "description": description,
                    "result": f"エラー: {error_msg}",
                    "success": False
                })
                self.logger.ulog(f"[失敗] {description}: {error_msg}", "error:failed")

        return results

    def _get_recent_context(self, max_entries: int = 10) -> str:
        """最近の会話履歴を取得"""
        recent = self.conversation_history[-max_entries:]

        lines = []
        for entry in recent:
            role = "User" if entry["role"] == "user" else "Assistant"
            content = entry["content"][:150] + "..." if len(entry["content"]) > 150 else entry["content"]
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    async def close(self):
        """エージェントを終了"""
        await self.connection_manager.close()


async def main():
    """メイン実行関数"""
    logger = Logger()
    logger.ulog("MCP Agent を起動しています...", "info:startup", always_print=True)

    # 設定ファイルのパスを解決
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    mcp_servers_path = os.path.join(script_dir, "mcp_servers.json")

    agent = MCPAgent(config_path=config_path, mcp_servers_path=mcp_servers_path)
    await agent.initialize()

    try:
        # ウェルカムメッセージ
        print("=" * 50)
        print("         MCP Agent - 準備完了")
        print("=" * 50)
        print(f"  接続サーバー: {len(agent.connection_manager.clients)}個")
        print(f"  利用可能ツール: {len(agent.connection_manager.tools_info)}個")
        print("=" * 50)
        print("終了するには 'quit' または 'exit' を入力してください。")
        print("-" * 60)

        while True:
            try:
                user_input = input("\nAgent> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.lower() in ['quit', 'exit', '終了']:
                break

            if not user_input:
                continue

            response = await agent.process_request(user_input)
            print(f"\n{response}")

    except Exception as e:
        logger.ulog(f"\n予期しないエラー: {e}", "error", always_print=True)
    finally:
        await agent.close()
        logger.ulog("\nMCP Agent を終了しました。", "info", always_print=True)


if __name__ == "__main__":
    asyncio.run(main())
