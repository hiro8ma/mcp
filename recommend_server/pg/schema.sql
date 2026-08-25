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
    embedding   vector({{DIM}}) NOT NULL,
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

-- テナントで絞るための索引。
--
-- これは検索の速度を助けるだけで、フィルタ付きベクトル検索の再現率問題は解決しない。
-- pgvector でテナント条件を付けて近傍検索すると、プランナは 2 択を迫られる。
--
--   embedding の索引を使う  → 全テナント横断で近傍を取ってから絞る。取りこぼしが出る
--   tenant_id の索引を使う  → 絞ってから距離計算。厳密だが件数に比例して遅くなる
--
-- 再現率を保ったまま速くしたいなら、次のいずれかが要る。
--   テナントごとの部分索引 / tenant_id によるパーティション / pgvector 0.8 の iterative_scan
--
-- 現状の bench_index.py は単一テナントでしか測らないため、この劣化は観測できない。
-- 複数テナントを投入した状態での計測は未実装。
CREATE INDEX IF NOT EXISTS items_tenant_idx ON items (tenant_id);
