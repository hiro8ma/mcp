-- マルチテナント構成のベクトル検索スキーマ。
--
-- テナント分離は Row Level Security で担保する。ただし RLS は接続ユーザーによって
-- 効いたり効かなかったりする。実測した挙動は次のとおり。
--
--   superuser で接続       → RLS はバイパスされる。FORCE を付けても無視される
--   テーブル所有者で接続   → FORCE ROW LEVEL SECURITY が無いとバイパスされる
--   非特権ロールで接続     → 正しく遮断される
--
-- したがって、アプリは必ず app_user（非特権）で接続する必要がある。
-- postgres で接続している限り、分離を担っているのは WHERE 句だけになる。
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS items (
    tenant_id   text        NOT NULL,
    item_id     text        NOT NULL,
    title       text        NOT NULL,
    description text        NOT NULL DEFAULT '',
    category    text,
    tags        text[]      NOT NULL DEFAULT '{}',
    embedding   vector(384) NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, item_id)
);

ALTER TABLE items ENABLE ROW LEVEL SECURITY;
ALTER TABLE items FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON items;
CREATE POLICY tenant_isolation ON items
    USING (tenant_id = current_setting('app.tenant_id', true));

-- app.tenant_id が未設定なら current_setting は空文字を返し、どの行にも一致しない。
-- 設定漏れが「全件見える」ではなく「何も見えない」に倒れる。

CREATE INDEX IF NOT EXISTS items_tenant_idx ON items (tenant_id);
