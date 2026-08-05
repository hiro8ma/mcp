# MCP Apps — サーバーが「対話の中で動く UI」を返す公式拡張

MCP の公式拡張 MCP Apps の整理（2026-08 時点）。
このリポジトリのサーバー群（能力レイヤー）に UI という能力を足せるか、の判断材料。

## 何を解決するか

テキスト応答では足りないケース（データ探索・多項目フォーム・リッチメディア・リアルタイム監視・多段ワークフロー）で、サーバーが HTML の UI を返し、ホストがチャット内のサンドボックス iframe に描画する。
スタンドアロン Web アプリと違い、会話のコンテキストに留まり、アプリからサーバーのツールを呼び直せて、認証・状態管理をホスト側の仕組みに乗せられる。

## 仕組み（プロトコル要素）

| 項目 | 値 |
|---|---|
| 拡張識別子 | `io.modelcontextprotocol/ui` |
| MIME type | `text/html;profile=mcp-app` |
| URI スキーム | `ui://` 接頭辞必須 |
| tool 側リンク | `_meta.ui.resourceUri`（フラット形式 `_meta["ui/resourceUri"]` は GA 前に削除予定。新規はネスト形式のみ） |
| resource 側 | `_meta.ui.csp.connectDomains` / `resourceDomains`、`_meta.ui.permissions`、`_meta.ui.prefersBorder` |
| capability | `capabilities.extensions["io.modelcontextprotocol/ui"].mimeTypes` |

実行フローは 3 段階。

1. Discovery — tool description の `_meta.ui.resourceUri` をホストが見てプリロード可能
2. Sandboxed rendering — `ui://` リソース（HTML 一式）をサンドボックス iframe に描画。親 DOM・cookie にはアクセス不可
3. 双方向通信 — postMessage 上の JSON-RPC（MCP の方言）。View → Host は `ui/initialize` `ui/message` `ui/update-model-context` 等、Host → View は `ui/notifications/tool-input` `tool-result` 等

## 経緯と現在地

- SEP-1865（Status: Final、Extensions Track）。MCP-UI（Ido Salomon / Liad Yosef）と OpenAI Apps SDK のアプローチを単一標準に統合したもの
- 仕様バージョン 2026-01-26 で Stable。MCP 初の公式拡張
- 同日 Claude の app directory で Amplitude / Asana / Box / Canva / Clay / Figma / Hex / monday.com / Slack が day-one 統合
- 対応ホストは 11 種（Claude web / Claude Desktop / VS Code Copilot / Microsoft 365 Copilot / Goose / Postman / MCPJam / ChatGPT / Cursor / Archestra.AI / PostHog Code）。**Claude Code は client matrix に載っていない**（ターミナル UI のため iframe 前提と噛み合わない）
- ChatGPT は `_meta.ui.resourceUri` を primary として受け付け、独自拡張は `window.openai.*` に分離。OpenAI 自身が「製品名で分岐せず feature detection せよ」と推奨しており、OpenAI Apps SDK と MCP Apps は対立ではなく収斂の関係

## SDK 対応と Go での実装可否

公式 SDK は TypeScript のみ（`@modelcontextprotocol/ext-apps`。core / react / app-bridge / server の 4 エントリポイント）。
ただし公式ドキュメントが「App クラスは convenience wrapper であり必須ではない。postMessage プロトコルを直接実装してよい」と明記している。

**Go サーバーでの MCP Apps 提供は成立する**。必要なのは 3 点。

1. tool に `_meta.ui.resourceUri` を載せる（go-sdk の `Meta` で表現可能）
2. `ui://` リソースを `text/html;profile=mcp-app` で返す
3. capability に `io.modelcontextprotocol/ui` を宣言する（go-sdk v1.7.0 の `AddExtension`。PR #794 で 2026-02 に追加済み）

MCP Apps 専用ヘルパーは go-sdk にないため、`_meta` のキー名は手書きになる。
View 側（iframe 内の HTML / JS）は結局 TypeScript エコシステムに寄る。

## 注意点

- MIME type を `text/html+mcp` と書いている記事があるが誤り。仕様は `text/html;profile=mcp-app`
- Claude / Claude Desktop でのレンダリング不具合の報告あり（ext-apps issue #615 / #671。ネゴシエーションと resources/read は成功するのに iframe が描画されない）。実装しても描画検証は MCPJam Inspector 等を併用するのが安全
- サンドボックスの CSP により、iframe 内から外部オリジンへのアクセスは `_meta.ui.csp` で宣言した先に限られる

## このリポジトリへの適用候補

weather_go の `get_weekly_forecast` に `ui://weather/forecast` リソース（週間予報チャート）を足すのが最小の実験。
tool は既存のまま `_meta.ui.resourceUri` を足すだけで、非対応ホストにはテキストのフォールバックが出るため後方互換。
着手する場合はホスト描画の検証環境（Claude Desktop + MCPJam Inspector）を先に決めてから。

## 出典

- https://modelcontextprotocol.io/extensions/apps/overview
- https://modelcontextprotocol.io/seps/1865-mcp-apps-interactive-user-interfaces-for-mcp
- https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/
- https://github.com/modelcontextprotocol/ext-apps
- https://github.com/modelcontextprotocol/go-sdk/pull/794 （Extensions capability）
- https://developers.openai.com/apps-sdk/mcp-apps-in-chatgpt
- https://modelcontextprotocol.io/extensions/client-matrix
