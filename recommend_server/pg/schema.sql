-- マルチテナント構成のベクトル検索スキーマ。
-- テナント間の分離は WHERE 句ではなく Row Level Security で担保する。
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

DROP POLICY IF EXISTS tenant_isolation ON items;
CREATE POLICY tenant_isolation ON items
    USING (tenant_id = current_setting('app.tenant_id', true));

-- 検索は必ずテナントで絞ってから距離計算に入る。
-- 単一列の HNSW だけだと全テナント横断で近傍を探してからフィルタするため、
-- テナント数が増えたときに再現率が落ちる。
CREATE INDEX IF NOT EXISTS items_tenant_idx ON items (tenant_id);
