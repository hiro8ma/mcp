"""合成カタログを投入する。

索引の再現率を測るには、実データに近い分布と件数が要る。
カテゴリ・素材・色・シーズンの組み合わせで機械的に作る。
"""

from __future__ import annotations

import argparse
import itertools
import random

from . import embed, store

CATEGORIES = {
    "アウター": ["コート", "ジャケット", "ブルゾン", "ダウン"],
    "トップス": ["シャツ", "ニット", "カットソー", "パーカー"],
    "ボトムス": ["デニム", "スラックス", "スカート", "ショーツ"],
    "シューズ": ["スニーカー", "ブーツ", "ローファー", "サンダル"],
    "バッグ": ["トート", "ショルダー", "バックパック", "クラッチ"],
    "アクセサリー": ["ネックレス", "リング", "ベルト", "スカーフ"],
}
MATERIALS = ["コットン", "ウール", "リネン", "レザー", "ナイロン", "デニム", "カシミヤ"]
COLORS = ["ブラック", "ホワイト", "ネイビー", "ベージュ", "グレー", "カーキ", "ブラウン"]
SEASONS = ["春夏", "秋冬", "オールシーズン"]
FITS = ["オーバーサイズ", "スリム", "レギュラー", "クロップド"]


def build_catalog(limit: int, seed: int = 42) -> list[store.Item]:
    rng = random.Random(seed)
    combos = [
        (cat, sub, mat, col)
        for cat, subs in CATEGORIES.items()
        for sub in subs
        for mat in MATERIALS
        for col in COLORS
    ]
    rng.shuffle(combos)

    items: list[store.Item] = []
    for i, (cat, sub, mat, col) in enumerate(itertools.islice(combos, limit)):
        season = rng.choice(SEASONS)
        fit = rng.choice(FITS)
        title = f"{col}{mat}{sub}"
        description = (
            f"{season}向けの{cat}。{mat}素材の{sub}で、{fit}シルエット。"
            f"カラーは{col}。日常使いから休日まで合わせやすい定番。"
        )
        items.append(
            store.Item(
                item_id=f"item-{i:05d}",
                title=title,
                description=description,
                category=cat,
                tags=(sub, mat, col, season, fit),
            )
        )
    return items


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default="tenant-a")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()

    store.init_schema()
    items = build_catalog(args.limit, args.seed)

    total = 0
    for start in range(0, len(items), args.batch):
        chunk = items[start : start + args.batch]
        vectors = embed.encode([f"{it.title}. {it.description}" for it in chunk])
        total += store.upsert(args.tenant, chunk, vectors)
        print(f"  {total}/{len(items)} 件投入")

    print(f"完了: tenant={args.tenant} 件数={store.count(args.tenant)}")


if __name__ == "__main__":
    main()
