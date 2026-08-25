"""テナント分離が実際に効いていることを確かめる。

RLS は接続ユーザーによってバイパスされる。実装した時点では効いているつもりでも、
接続先を変えただけで無言で無効になる種類の設定なので、テストで固定しておく。
"""

from __future__ import annotations

import psycopg
import pytest
from pgvector.psycopg import register_vector

from . import embed, store

DIM = embed.dimension()


def _vec(seed: float) -> list[float]:
    return [seed] + [0.0] * (DIM - 1)


@pytest.fixture(scope="module", autouse=True)
def seeded():
    store.init_schema()
    for t, i, v in (("t-a", "a1", 1.0), ("t-b", "b1", 2.0)):
        store.upsert(t, [store.Item(item_id=i, title=f"{t} の商品")], [_vec(v)])
    yield
    with store.connect(owner=True) as conn:
        conn.execute("DELETE FROM items WHERE tenant_id IN ('t-a','t-b')")
        conn.commit()


def test_search_returns_only_own_tenant():
    hits = store.search("t-a", _vec(1.0), top_k=50)
    assert hits, "自テナントの行が取れていない"
    assert all(h.item_id.startswith("a") for h in hits), f"他テナントが混ざった: {hits}"


def test_count_is_per_tenant():
    assert store.count("t-a") == 1
    assert store.count("t-b") == 1


def test_rls_blocks_without_tenant_setting():
    """app.tenant_id を立てずにアプリロールで読むと 0 件になる。

    設定漏れが「全件見える」ではなく「何も見えない」に倒れることを確かめる。
    """
    with psycopg.connect(store.APP_DSN) as conn:
        register_vector(conn)
        n = conn.execute("SELECT count(*) FROM items").fetchone()[0]
    assert n == 0, f"RLS が効いていない。{n} 件見えている"


def test_dimension_matches_schema():
    """スキーマの次元とモデルの次元が一致していることを確かめる。

    モデルを差し替えたときに、投入時の行ごとのエラーではなく
    ここで落ちるようにしておく。
    """
    store.assert_dimension_matches()


def test_app_role_is_not_superuser():
    """アプリロールが superuser だと RLS がバイパスされる。設定ミスの検出。"""
    with psycopg.connect(store.APP_DSN) as conn:
        is_super = conn.execute(
            "SELECT usesuper FROM pg_user WHERE usename = current_user"
        ).fetchone()[0]
    assert not is_super, "アプリロールが superuser になっている。RLS が無効化される"
