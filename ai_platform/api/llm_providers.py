"""
LLM Providers

複数のLLMプロバイダーに対応
- OpenAI API
- Google Vertex AI (Model Garden)
- Anthropic (Claude)
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from enum import Enum


class LLMProvider(str, Enum):
    """サポートするLLMプロバイダー"""
    OPENAI = "openai"
    VERTEX_AI = "vertex_ai"
    ANTHROPIC = "anthropic"


class BaseLLMClient(ABC):
    """LLMクライアントの基底クラス"""

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """チャット補完"""
        pass

    @abstractmethod
    def get_langchain_llm(self):
        """LangChain用のLLMインスタンスを取得"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API クライアント"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        response = await client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature)
        )
        return response.choices[0].message.content

    def get_langchain_llm(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key
        )


class VertexAIClient(BaseLLMClient):
    """
    Google Vertex AI (Model Garden) クライアント

    使用可能なモデル（2026-08 時点）:
    - gemini-3.1-pro （Pro の最上位。3.5 Pro は出ていない）
    - gemini-3.5-flash / gemini-3.6-flash / gemini-3.7-flash
    - gemini-3.1-flash-lite （最も費用効率がよい）
    - claude 系 （Model Garden 経由）

    gemini-2.x 系は 2026-10-16 に提供終了する。
    Flash は 4 か月で 3.1 から 3.7 まで進む一方、Pro は 3.1 で止まっている。
    系列によって更新の速さが違うため、まとめて追わない。
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        model: str = "gemini-3.5-flash"
    ):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.model = model

        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required")

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=self.project_id, location=self.location)

        model = GenerativeModel(kwargs.get("model", self.model))

        # メッセージを変換
        prompt = self._convert_messages(messages)

        response = await model.generate_content_async(prompt)
        return response.text

    def _convert_messages(self, messages: List[Dict[str, str]]) -> str:
        """OpenAI形式のメッセージをVertex AI形式に変換"""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"システム指示: {content}\n")
            elif role == "user":
                parts.append(f"ユーザー: {content}\n")
            elif role == "assistant":
                parts.append(f"アシスタント: {content}\n")

        return "\n".join(parts)

    def get_langchain_llm(self):
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model=self.model,
            project=self.project_id,
            location=self.location
        )


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude クライアント"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.temperature = temperature

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # システムメッセージを抽出
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        response = await client.messages.create(
            model=kwargs.get("model", self.model),
            max_tokens=4096,
            system=system_msg,
            messages=chat_messages
        )
        return response.content[0].text

    def get_langchain_llm(self):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key
        )


class LLMFactory:
    """LLMクライアントのファクトリー"""

    @staticmethod
    def create(
        provider: LLMProvider,
        **kwargs
    ) -> BaseLLMClient:
        """プロバイダーに応じたLLMクライアントを作成"""

        if provider == LLMProvider.OPENAI:
            return OpenAIClient(**kwargs)
        elif provider == LLMProvider.VERTEX_AI:
            return VertexAIClient(**kwargs)
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def create_from_env() -> BaseLLMClient:
        """環境変数から自動的にLLMクライアントを作成"""

        # 優先順位: OpenAI > Anthropic > Vertex AI
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIClient()
        elif os.getenv("ANTHROPIC_API_KEY"):
            return AnthropicClient()
        elif os.getenv("GOOGLE_CLOUD_PROJECT"):
            return VertexAIClient()
        else:
            raise ValueError(
                "No LLM provider configured. "
                "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_CLOUD_PROJECT"
            )
