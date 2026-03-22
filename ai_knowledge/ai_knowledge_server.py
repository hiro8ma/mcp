#!/usr/bin/env python3
"""
FT済みローカルLLMによるAIエンジニアリング知識Q&A MCPサーバー

Gemma 3 4B をLoRAでファインチューニングしたモデルを使い、
AIエンジニアリングに関する質問に回答する。
"""

from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("AI Knowledge Assistant")

# モデルのパス設定
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
ADAPTER_PATH = str(Path(__file__).parent.parent.parent / "ft" / "adapters")

# 遅延ロード（初回呼び出し時にロード）
_model = None
_tokenizer = None


def _load_model():
    """モデルを遅延ロードする。"""
    global _model, _tokenizer
    if _model is None:
        from mlx_lm import load

        _model, _tokenizer = load(MODEL_ID, adapter_path=ADAPTER_PATH)
    return _model, _tokenizer


def _generate(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
    """FT済みモデルで推論を実行する。"""
    from mlx_lm import generate

    model, tokenizer = _load_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


@mcp.tool()
def ask_ai_engineering(question: str) -> str:
    """AIエンジニアリングに関する質問に回答します。

    RAG、評価パイプライン、プロンプトエンジニアリング、エージェント設計、
    ファインチューニングなどのトピックについて、
    FT済みローカルモデルが実務に即した回答を生成します。

    Args:
        question: AIエンジニアリングに関する質問
    """
    return _generate(
        system_prompt="あなたはAIエンジニアリングの専門家です。技術的に正確で、実務に即した回答をしてください。",
        user_prompt=question,
    )


@mcp.tool()
def quiz_ai_engineering(topic: str) -> str:
    """AIエンジニアリングのトピックについてクイズ形式で出題します。

    指定されたトピックについて、理解度を確認するための質問を生成します。

    Args:
        topic: クイズのトピック（例: RAG, 評価パイプライン, エージェント, LoRA）
    """
    return _generate(
        system_prompt="あなたはAIエンジニアリングの講師です。指定されたトピックについて、理解度を確認するための質問を1つ出題してください。質問の後に、模範回答も提供してください。",
        user_prompt=f"トピック: {topic}",
    )


@mcp.tool()
def explain_concept(concept: str, depth: str = "intermediate") -> str:
    """AIエンジニアリングの概念を指定された深さで説明します。

    Args:
        concept: 説明する概念（例: LoRA, ReAct, ハイブリッド検索, 量子化）
        depth: 説明の深さ（beginner, intermediate, advanced）
    """
    depth_map = {
        "beginner": "初心者にもわかるように、具体例を交えて簡潔に",
        "intermediate": "実務経験のあるエンジニア向けに、技術的な詳細を含めて",
        "advanced": "AIエンジニアリングの専門家向けに、論文レベルの詳細とトレードオフを含めて",
    }
    instruction = depth_map.get(depth, depth_map["intermediate"])
    return _generate(
        system_prompt=f"あなたはAIエンジニアリングの専門家です。{instruction}説明してください。",
        user_prompt=f"「{concept}」について説明してください。",
    )


if __name__ == "__main__":
    mcp.run()
