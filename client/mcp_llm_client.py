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

    def __init__(self, config_file: str = "mcp_servers.json"):
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

    async def collect_all_tools(self):
        print("[収集] ツール情報を収集中...", flush=True)

        for server_name, server_info in self.servers.items():
            try:
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

                print(f"  [成功] {server_name}: {len(tools)}個のツール", flush=True)
                await client.__aexit__(None, None, None)

            except Exception as e:
                print(f"  [エラー] {server_name}: {e}", flush=True)


class LLMIntegrationPrep:
    """LLM統合の準備"""

    def prepare_tools_for_llm(self, tools_schema: Dict[str, List[Any]]) -> str:
        tools_description = []

        for server_name, tools in tools_schema.items():
            for tool in tools:
                params_desc = self._format_parameters(tool.get('parameters', {}))
                tool_desc = f"{server_name}.{tool['name']}:\n  説明: {tool['description']}\n  {params_desc}"
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

    def __init__(self, config_file: str = "mcp_servers.json"):
        self.collector = ToolCollector(config_file)
        self.prep = LLMIntegrationPrep()
        self.llm = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.clients = {}
        self.conversation_history = []
        self.context = {
            "session_start": datetime.now(),
            "tool_calls": 0,
            "errors": 0
        }

    async def initialize(self):
        print("[起動] LLM統合MCPクライアントを起動中...", flush=True)
        await self.collector.collect_all_tools()

        for server_name, server_info in self.collector.servers.items():
            try:
                command = server_info["path"][0]
                args = server_info["path"][1:]
                transport = StdioTransport(command=command, args=args)
                client = Client(transport)
                await client.__aenter__()
                self.clients[server_name] = client
            except Exception as e:
                print(f"  [WARNING] {server_name}への接続失敗: {e}")

        print("[完了] 初期化完了\n", flush=True)
        self._show_available_tools()

    def _show_available_tools(self):
        total_tools = sum(len(tools) for tools in self.collector.tools_schema.values())
        print(f"[ツール] 利用可能なツール: {total_tools}個")
        for server_name, tools in self.collector.tools_schema.items():
            print(f"  - {server_name}: {len(tools)}個")
        print()

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
            print("  [分析] クエリを分析中...", flush=True)
            decision = await self._analyze_query(query)

            self.conversation_history.append({"role": "user", "content": query})

            if decision.get("reasoning"):
                print(f"  [判断] {decision['reasoning']}", flush=True)

            if decision.get("needs_tool", False):
                print(f"  [選択] ツール: {decision['server']}.{decision['tool']}", flush=True)
                print(f"  [実行] 処理中...", flush=True)

                result = await self._execute_tool(
                    decision['server'],
                    decision['tool'],
                    decision['arguments']
                )
                print(f"  [完了] 実行完了", flush=True)

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
        print("コマンド: help, status, quit")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("あなた: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nお疲れさまでした！")
                    break
                elif user_input.lower() in ['help', '?']:
                    self._show_available_tools()
                    continue
                elif user_input.lower() == 'status':
                    self._show_status()
                    continue

                print("\n処理中...")
                response = await self.process_query(user_input)
                print(f"\nアシスタント: {response}\n")

            except KeyboardInterrupt:
                print("\n\n中断されました")
                break

    def _show_status(self):
        duration = datetime.now() - self.context["session_start"]
        print(f"\nセッション情報:")
        print(f"  経過時間: {str(duration).split('.')[0]}")
        print(f"  ツール実行回数: {self.context['tool_calls']}")
        print(f"  エラー回数: {self.context['errors']}\n")

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
