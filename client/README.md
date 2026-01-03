# MCP Client

LLM-integrated MCP client for natural language tool interaction.

## Setup

```bash
cp .env.example .env
# Set OPENAI_API_KEY
```

## Usage

```bash
make run
```

## Example

```
あなた: 1+2を計算して
[分析] クエリを分析中...
[判断] 1と2を加算するためにcalc.addツールを使用します。
[選択] ツール: calc.add
[実行] 処理中...
[完了] 実行完了

アシスタント: 1+2を計算した結果は3です。
```

## Commands

- `help` - Show available tools
- `status` - Show session info
- `quit` - Exit
