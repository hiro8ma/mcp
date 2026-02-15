#!/usr/bin/env python3
"""
LLM統合MCPクライアント
自然言語でMCPツールを操作する対話型クライアント
"""

import asyncio
import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

load_dotenv()


class ToolCollector:
    """MCPサーバーのツール情報を収集"""

    def __init__(self, config_file: str = "mcp_servers.json", status_callback=None):
        self._log = status_callback or (lambda msg: print(msg, flush=True))
        self.servers = {}
        self.clients = {}
        self.tools_schema = {}
        self.load_config(config_file)

    def load_config(self, config_file: str):
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"[WARNING] 設定ファイル {config_file} が見つかりません")
            self.servers = {}
            return

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if "mcpServers" in config:
            for server_name, server_config in config["mcpServers"].items():
                path = [server_config["command"]] + server_config["args"]
                self.servers[server_name] = {
                    "name": server_name,
                    "path": path
                }

    async def collect_all_tools(self, timeout_per_server: float = 30):
        self._log("[収集] ツール情報を収集中...")

        for server_name, server_info in self.servers.items():
            try:
                async with asyncio.timeout(timeout_per_server):
                    command = server_info["path"][0]
                    args = server_info["path"][1:]
                    transport = StdioTransport(command=command, args=args)
                    client = Client(transport)
                    await client.__aenter__()
                    await client.ping()
                    self.clients[server_name] = client

                    tools = await client.list_tools()
                    self.tools_schema[server_name] = []

                    for tool in tools:
                        tool_info = {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                        }
                        self.tools_schema[server_name].append(tool_info)

                    self._log(f"  [成功] {server_name}: {len(tools)}個のツール")
                    await client.__aexit__(None, None, None)

            except TimeoutError:
                self._log(f"  [タイムアウト] {server_name}: {timeout_per_server}秒で応答なし")
            except Exception as e:
                self._log(f"  [エラー] {server_name}: {e}")


class LLMIntegrationPrep:
    """LLM統合の準備"""

    def prepare_tools_for_llm(self, tools_schema: Dict[str, List[Any]]) -> str:
        tools_description = []

        for server_name, tools in tools_schema.items():
            for tool in tools:
                params_desc = self._format_parameters(tool.get('parameters', {}))
                tool_desc = f"サーバー: {server_name}, ツール名: {tool['name']}\n  説明: {tool['description']}\n  {params_desc}"
                tools_description.append(tool_desc)

        return "\n\n".join(tools_description)

    def _format_parameters(self, params_schema: Dict) -> str:
        if not params_schema or 'properties' not in params_schema:
            return "パラメータ: なし"

        param_lines = ["パラメータ:"]
        properties = params_schema.get('properties', {})
        required = params_schema.get('required', [])

        for key, value in properties.items():
            param_type = value.get('type', 'any')
            param_desc = value.get('description', '')
            is_required = key in required
            req_text = "必須" if is_required else "オプション"
            param_lines.append(f"    - {key} ({param_type}, {req_text}): {param_desc}")

        return "\n".join(param_lines)

    def validate_llm_response(self, response: str) -> Dict:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"LLMの応答をパースできません")


class LLMClient:
    """LLM統合MCPクライアント"""

    def __init__(self, config_file: str = "mcp_servers.json", status_callback=None):
        self._log = status_callback or (lambda msg: print(msg, flush=True))
        self.collector = ToolCollector(config_file, status_callback=self._log)
        self.prep = LLMIntegrationPrep()
        self.llm = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.clients = {}
        self.conversation_history = []
        self.context = {
            "session_start": datetime.now(),
            "tool_calls": 0,
            "errors": 0
        }
        self._init_commands()

    def _init_commands(self):
        """スラッシュコマンドを初期化"""
        self.commands = {
            "help": {
                "handler": self._cmd_help,
                "description": "コマンド一覧を表示"
            },
            "tools": {
                "handler": self._cmd_tools,
                "description": "利用可能なツール一覧"
            },
            "status": {
                "handler": self._cmd_status,
                "description": "セッション状態を表示"
            },
            "history": {
                "handler": self._cmd_history,
                "description": "会話履歴を表示"
            },
            "clear": {
                "handler": self._cmd_clear,
                "description": "会話履歴をクリア"
            },
            "quit": {
                "handler": self._cmd_quit,
                "description": "終了"
            },
        }

    def _cmd_help(self, args: str) -> str:
        """ヘルプを表示"""
        lines = ["\n利用可能なコマンド:"]
        for cmd, info in self.commands.items():
            lines.append(f"  /{cmd:<10} - {info['description']}")
        lines.append("\nそれ以外の入力は自然言語として処理されます。\n")
        return "\n".join(lines)

    def _cmd_tools(self, args: str) -> str:
        """ツール一覧を表示"""
        lines = ["\n利用可能なツール:"]
        for server_name, tools in self.collector.tools_schema.items():
            lines.append(f"\n  [{server_name}]")
            for tool in tools:
                desc = tool['description'][:40] + "..." if len(tool['description']) > 40 else tool['description']
                lines.append(f"    - {tool['name']}: {desc}")
        lines.append("")
        return "\n".join(lines)

    def _cmd_status(self, args: str) -> str:
        """ステータスを表示"""
        duration = datetime.now() - self.context["session_start"]
        total_tools = sum(len(tools) for tools in self.collector.tools_schema.values())
        lines = [
            "\nセッション情報:",
            f"  経過時間:       {str(duration).split('.')[0]}",
            f"  ツール実行回数: {self.context['tool_calls']}",
            f"  エラー回数:     {self.context['errors']}",
            f"  会話履歴:       {len(self.conversation_history)}件",
            f"  接続サーバー:   {len(self.clients)}個",
            f"  利用可能ツール: {total_tools}個\n"
        ]
        return "\n".join(lines)

    def _cmd_history(self, args: str) -> str:
        """会話履歴を表示"""
        if not self.conversation_history:
            return "\n会話履歴はありません。\n"

        lines = ["\n会話履歴:"]
        for i, msg in enumerate(self.conversation_history, 1):
            role = "あなた" if msg["role"] == "user" else "AI"
            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            lines.append(f"  {i}. [{role}] {content}")
        lines.append("")
        return "\n".join(lines)

    def _cmd_clear(self, args: str) -> str:
        """会話履歴をクリア"""
        self.conversation_history = []
        return "\n会話履歴をクリアしました。\n"

    def _cmd_quit(self, args: str) -> str:
        """終了フラグを返す"""
        return "__QUIT__"

    def _handle_command(self, input_text: str) -> tuple[bool, str]:
        """コマンドを処理。(is_command, result)を返す"""
        if not input_text.startswith("/"):
            return False, ""

        parts = input_text[1:].split(maxsplit=1)
        cmd_name = parts[0].lower()
        cmd_args = parts[1] if len(parts) > 1 else ""

        if cmd_name in self.commands:
            result = self.commands[cmd_name]["handler"](cmd_args)
            return True, result
        else:
            return True, f"\n不明なコマンド: /{cmd_name}\n/help でコマンド一覧を確認してください。\n"

    async def initialize(self):
        self._log("[起動] LLM統合MCPクライアントを起動中...")
        await self.collector.collect_all_tools()

        for server_name, server_info in self.collector.servers.items():
            try:
                async with asyncio.timeout(30):
                    command = server_info["path"][0]
                    args = server_info["path"][1:]
                    transport = StdioTransport(command=command, args=args)
                    client = Client(transport)
                    await client.__aenter__()
                    self.clients[server_name] = client
            except TimeoutError:
                self._log(f"  [WARNING] {server_name}への接続タイムアウト")
            except Exception as e:
                self._log(f"  [WARNING] {server_name}への接続失敗: {e}")

        self._log("[完了] 初期化完了")
        self._show_available_tools()

    def _show_available_tools(self):
        total_tools = sum(len(tools) for tools in self.collector.tools_schema.values())
        self._log(f"[ツール] 利用可能なツール: {total_tools}個")
        for server_name, tools in self.collector.tools_schema.items():
            self._log(f"  - {server_name}: {len(tools)}個")

    async def _analyze_query(self, query: str) -> Dict:
        tools_desc = self.prep.prepare_tools_for_llm(self.collector.tools_schema)

        recent_history = ""
        if self.conversation_history:
            recent_messages = self.conversation_history[-5:]
            history_lines = []
            for msg in recent_messages:
                role = "ユーザー" if msg["role"] == "user" else "アシスタント"
                history_lines.append(f"{role}: {msg['content']}")
            recent_history = "\n".join(history_lines)

        prompt = f"""
あなたは優秀なアシスタントです。ユーザーの質問を分析し、適切な対応を決定してください。

## これまでの会話
{recent_history if recent_history else "（新しい会話）"}

## 現在のユーザーの質問
{query}

## 利用可能なツール
{tools_desc}

## 応答形式
以下のJSON形式で必ず応答してください（JSONのみ）：

needs_tool=trueの場合:
{{"needs_tool": true, "server": "サーバー名", "tool": "ツール名", "arguments": {{}}, "reasoning": "理由"}}

needs_tool=falseの場合:
{{"needs_tool": false, "reasoning": "理由", "response": "回答"}}
"""

        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        raw_response = response.choices[0].message.content
        return self.prep.validate_llm_response(raw_response)

    async def process_query(self, query: str) -> str:
        try:
            self._log("  [分析] クエリを分析中...")
            decision = await self._analyze_query(query)

            self.conversation_history.append({"role": "user", "content": query})

            if decision.get("reasoning"):
                self._log(f"  [判断] {decision['reasoning']}")

            if decision.get("needs_tool", False):
                self._log(f"  [選択] ツール: {decision['server']}.{decision['tool']}")
                self._log(f"  [実行] 処理中...")

                result = await self._execute_tool(
                    decision['server'],
                    decision['tool'],
                    decision['arguments']
                )
                self._log(f"  [完了] 実行完了")

                return await self._interpret_result(query, decision, result)
            else:
                response = decision.get("response", "回答を生成できませんでした。")
                self.conversation_history.append({"role": "assistant", "content": response})
                return response

        except Exception as e:
            self.context["errors"] += 1
            return f"エラーが発生しました: {str(e)}"

    async def _execute_tool(self, server: str, tool: str, arguments: Dict) -> Any:
        if server not in self.clients:
            raise ValueError(f"サーバー '{server}' が見つかりません")

        self.context["tool_calls"] += 1
        client = self.clients[server]
        result = await client.call_tool(tool, arguments)

        if hasattr(result, 'content'):
            if isinstance(result.content, list) and result.content:
                first = result.content[0]
                if hasattr(first, 'text'):
                    return first.text
        return str(result)

    async def _interpret_result(self, query: str, decision: Dict, result: Any) -> str:
        prompt = f"""
ユーザーの質問: {query}
実行したツール: {decision['server']}.{decision['tool']}
実行結果: {result}

結果をユーザーが理解しやすいように日本語で説明してください。
"""

        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        interpreted = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": interpreted})
        return interpreted

    async def interactive_mode(self):
        print("\n" + "=" * 60)
        print("LLM統合MCPクライアント - 対話モード")
        print("=" * 60)
        print("自然言語でMCPツールを操作できます。")
        print("/help でコマンド一覧を表示")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("あなた: ").strip()

                if not user_input:
                    continue

                # スラッシュコマンドの処理
                is_command, result = self._handle_command(user_input)
                if is_command:
                    if result == "__QUIT__":
                        print("\nお疲れさまでした！")
                        break
                    print(result)
                    continue

                print("\n処理中...")
                response = await self.process_query(user_input)
                print(f"\nアシスタント: {response}\n")

            except KeyboardInterrupt:
                print("\n\n中断されました")
                break

    async def cleanup(self):
        for client in self.clients.values():
            try:
                await client.__aexit__(None, None, None)
            except:
                pass


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] 環境変数 OPENAI_API_KEY を設定してください")
        return

    client = LLMClient()

    try:
        await client.initialize()
        await client.interactive_mode()
    except KeyboardInterrupt:
        print("\n中断されました")
    finally:
        await client.cleanup()
        print("終了します")


if __name__ == "__main__":
    asyncio.run(main())
