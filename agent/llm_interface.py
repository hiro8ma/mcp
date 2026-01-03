#!/usr/bin/env python3
"""
LLM通信統一インターフェース
全クラスのLLM通信を統一管理
"""

import json
import re
from typing import Dict, List, Any
from openai import AsyncOpenAI

from config_manager import Config
from utils import Logger, safe_str


class LLMInterface:
    """全LLM通信の統一インターフェース"""

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.client = AsyncOpenAI()

    def _get_llm_params(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        params = {
            "model": self.config.llm.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.llm.temperature),
        }

        if hasattr(self.config.llm, "max_completion_tokens"):
            params["max_completion_tokens"] = self.config.llm.max_completion_tokens

        if kwargs.get("response_format"):
            params["response_format"] = kwargs["response_format"]

        return params

    async def _call_llm(self, messages: List[Dict], **kwargs) -> str:
        params = self._get_llm_params(messages, **kwargs)
        response = await self.client.chat.completions.create(**params)
        return safe_str(response.choices[0].message.content)

    async def determine_execution_type(self, user_query: str, recent_context: str, tools_info: str) -> Dict:
        prompt = f"""あなたはユーザーからの要求を分析し、次のどの実行方式が最適かを判定するAIです。

利用可能ツール一覧:
{tools_info}

最近の会話履歴:
{recent_context}

現在のユーザー要求:
「{user_query}」

判定ルール:
1. 計算、データベース検索、API呼び出し、ファイル操作などが必要な場合 → TOOL
2. ユーザーの要求が曖昧で詳細確認が必要 → CLARIFICATION
3. 単純な質問で既存の知識だけで十分回答可能 → NO_TOOL

結果をJSON形式で返してください:
{{"type": "NO_TOOL|CLARIFICATION|TOOL", "reason": "判定理由", "response": "ユーザーへの応答"}}"""

        try:
            content = await self._call_llm(
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            result = json.loads(content)

            if result.get('type') not in ['NO_TOOL', 'CLARIFICATION']:
                result['type'] = 'TOOL'

            self.logger.ulog(f"判定: {result.get('type', 'UNKNOWN')} - {result.get('reason', '')}", "info:classification", show_level=True)

            return result

        except Exception as e:
            self.logger.ulog(f"実行方式判定失敗: {e}", "error:error")
            return {"type": "TOOL", "reason": "判定エラーによりデフォルト選択"}

    async def generate_task_list(self, user_query: str, context: str, tools_info: str, custom_instructions: str = "") -> List[Dict]:
        prompt = f"""あなたは以下のタスクを遂行するAIアシスタントです：

ユーザーリクエスト: {user_query}
{custom_instructions}

利用可能ツール:
{tools_info}

会話履歴とコンテキスト:
{context}

上記のユーザーリクエストを実行するために必要なタスクを順序立ててリストアップしてください。
各タスクはJSON形式で、以下の要素を含める必要があります：
- tool: 使用するツール名
- params: ツールに渡すパラメータ（辞書形式）
- description: タスクの説明

応答は純粋なJSONリスト形式でお願いします：
[
  {{"tool": "ツール名", "params": {{"param1": "value1"}}, "description": "タスクの説明"}},
  ...
]"""

        try:
            content = await self._call_llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            try:
                tasks = json.loads(content)
                if isinstance(tasks, list):
                    return tasks
            except json.JSONDecodeError:
                pass

            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                tasks = json.loads(json_match.group(1))
                if isinstance(tasks, list):
                    return tasks

            self.logger.ulog(f"タスクリスト解析失敗: {content[:100]}...", "warning:task")
            return []

        except Exception as e:
            self.logger.ulog(f"タスクリスト生成失敗: {e}", "error:task")
            return []

    async def interpret_results(self, user_query: str, results: List[Dict], context: str, custom_instructions: str = "") -> str:
        prompt = f"""以下のユーザーリクエストに対するツール実行結果を解釈して、自然な日本語で回答してください。

ユーザーリクエスト: {user_query}
{custom_instructions}

ツール実行結果:
{json.dumps(results, ensure_ascii=False, indent=2)}

会話コンテキスト:
{context}

実行結果を基に、ユーザーにとって理解しやすい形で回答を作成してください。"""

        try:
            return await self._call_llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
        except Exception as e:
            self.logger.ulog(f"結果解釈失敗: {e}", "error:interpretation")
            return f"実行は完了しましたが、結果の解釈中にエラーが発生しました: {str(e)}"
