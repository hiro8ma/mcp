# MCP Agent

Interactive MCP Agent with LLM task orchestration.

## Setup

```bash
cp .env.example .env
# Set OPENAI_API_KEY
```

## Usage

```bash
make run
```

## Features

- Connects to multiple MCP servers (calc, db, external_api, universal_tools)
- LLM-based task decomposition and execution
- Interactive REPL interface

## Configuration

Edit `config.yaml` to customize:
- LLM model and temperature
- Retry settings
- Display options
