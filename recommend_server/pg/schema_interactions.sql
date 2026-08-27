-- 行動履歴。協調フィルタリングはこのテーブルだけで動く。
--
-- 内容ベースフィルタリングが items.embedding（アイテムの中身）を見るのに対し、
-- 協調フィルタリングは中身を一切見ない。誰が何に触れたかの共起だけを使う。
-- そのため「中身では説明できない併買」を拾える一方、行動ログが無いアイテムは
-- 一切推薦できない（コールドスタート）。
CREATE TABLE IF NOT EXISTS interactions (
    tenant_id text        NOT NULL,
    user_id   text        NOT NULL,
    item_id   text        NOT NULL,
    -- rating は明示評価。閲覧・購入のような暗黙のフィードバックは 1.0 で入れる。
    rating    real        NOT NULL DEFAULT 1.0,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id, item_id)
);

-- items と同じくテナント分離は RLS で担保する。
-- 行動履歴はアイテム以上に他テナントへ漏らせない。
ALTER TABLE interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE interactions FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation_interactions ON interactions;
CREATE POLICY tenant_isolation_interactions ON interactions
    USING (tenant_id = current_setting('app.tenant_id', true));

-- ユーザー間型はユーザー単位で履歴を引く。アイテム間型はアイテム単位で引く。
-- 方向が逆なので索引も両方要る。
CREATE INDEX IF NOT EXISTS interactions_user_idx ON interactions (tenant_id, user_id);
CREATE INDEX IF NOT EXISTS interactions_item_idx ON interactions (tenant_id, item_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON interactions TO app_user;
