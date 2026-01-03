#!/usr/bin/env python3
"""
Simplified MCP Connection Manager
MCPサーバーへの接続とツール情報管理
"""

import json
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from utils import setup_windows_encoding, safe_str, Logger

setup_windows_encoding()
load_dotenv()


class ConnectionManager:
    """MCPサーバーへの接続を管理するマネージャー"""

    def __init__(self, config_file: str = "mcp_servers.json", verbose: bool = True):
        self.config_file = config_file
        self.verbose = verbose
        self.logger = Logger(verbose=verbose)

        self.servers: Dict[str, Dict] = {}
        self.clients: Dict[str, Client] = {}
        self.tools_info: Dict[str, Dict] = {}

        self._initialized = False
        self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_file}")

        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if "mcpServers" in config:
            for server_name, server_config in config["mcpServers"].items():
                path = [server_config["command"]] + server_config["args"]
                self.servers[server_name] = {
                    "name": server_name,
                    "path": path
                }
        else:
            for server_info in config.get("servers", []):
                self.servers[server_info["name"]] = server_info

    async def initialize(self):
        if self._initialized:
            self.logger.ulog("既に初期化済みです", "info:connection")
            return

        self.logger.ulog("=" * 50, "info")
        self.logger.ulog(" MCP Connection Manager", "info")
        self.logger.ulog("=" * 50, "info")
        self.logger.ulog(f"{len(self.servers)}個のサーバーを検出", "info:config")

        await self._connect_all_servers()
        await self._collect_tools_info()

        self._initialized = True

        self.logger.ulog(f"\n初期化完了", "info:init")
        self.logger.ulog(f"  接続サーバー: {len(self.clients)}個", "info")
        self.logger.ulog(f"  利用可能ツール: {len(self.tools_info)}個", "info")
        self.logger.ulog("=" * 50, "info")

    async def _connect_all_servers(self):
        self.logger.ulog("\nMCPサーバーに接続中...", "info:connection")

        for server_name, server_info in self.servers.items():
            try:
                self.logger.ulog(f"  {server_name}に接続中...", "info:connection")

                command = server_info["path"][0]
                args = server_info["path"][1:]
                transport = StdioTransport(command=command, args=args)
                client = Client(transport)
                await client.__aenter__()
                self.clients[server_name] = client

                self.logger.ulog(" OK", "info:connection")

            except Exception as e:
                self.logger.ulog(f" 失敗: {e}", "error:error")
                continue

    async def _collect_tools_info(self):
        self.logger.ulog("\nツール情報を収集中...", "info:collection")

        for server_name, client in self.clients.items():
            try:
                tools = await client.list_tools()
                tool_count = 0

                for tool in tools:
                    tool_name = tool.name

                    self.tools_info[tool_name] = {
                        "server": server_name,
                        "schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                        "description": tool.description if hasattr(tool, 'description') else ""
                    }
                    tool_count += 1

                self.logger.ulog(f"[{server_name}] {tool_count}個のツールを発見", "info:collection")

            except Exception as e:
                self.logger.ulog(f"[{server_name}] ツール情報取得失敗: {e}", "error:error")
                continue

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if tool_name not in self.tools_info:
            raise ValueError(f"未知のツール: {tool_name}")

        server_name = self.tools_info[tool_name]["server"]

        if server_name not in self.clients:
            raise ValueError(f"サーバー '{server_name}' に接続されていません")

        client = self.clients[server_name]

        try:
            result = await client.call_tool(tool_name, arguments)

            if hasattr(result, 'content') and hasattr(result.content, '__iter__'):
                for content_item in result.content:
                    if hasattr(content_item, 'text') and isinstance(content_item.text, str):
                        content_item.text = safe_str(content_item.text)

            return result

        except Exception as e:
            error_msg = safe_str(str(e))
            raise RuntimeError(f"ツール実行エラー ({tool_name}): {error_msg}")

    def get_available_tools(self) -> List[str]:
        return list(self.tools_info.keys())

    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        return self.tools_info.get(tool_name)

    def format_tools_for_llm(self) -> str:
        formatted = []

        for tool_name, info in self.tools_info.items():
            server = info["server"]
            description = info["description"]

            schema = info.get("schema", {})
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            params_info = []
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                is_required = param_name in required
                req_text = "必須" if is_required else "オプション"
                param_desc = param_info.get("description", "")
                params_info.append(f"  - {param_name} ({param_type}, {req_text}): {param_desc}")

            params_text = "\n".join(params_info) if params_info else "  パラメータなし"

            formatted.append(f"""
{tool_name} (サーバー: {server}):
  説明: {description}
  パラメータ:
{params_text}
""".strip())

        return "\n\n".join(formatted)

    async def close(self):
        for server_name, client in self.clients.items():
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                if self.verbose:
                    self.logger.ulog(f"{server_name}の切断エラー: {e}", "warning:warning")

        self.clients.clear()
        self._initialized = False
