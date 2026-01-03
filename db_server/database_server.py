#!/usr/bin/env python3
"""
Database MCP Server
"""

import sqlite3
import re
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from fastmcp import FastMCP

mcp = FastMCP("Database Server")

DB_PATH = os.path.join(os.path.dirname(__file__), 'intelligent_shop.db')


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def validate_sql_safety(sql: str) -> bool:
    sql_upper = sql.upper().strip()

    if not sql_upper.startswith('SELECT'):
        return False

    dangerous_keywords = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER',
        'CREATE', 'TRUNCATE', 'REPLACE', 'PRAGMA',
        'ATTACH', 'DETACH', 'VACUUM'
    ]

    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False

    dangerous_patterns = [
        r';\s*(DROP|DELETE|INSERT|UPDATE)',
        r'--',
        r'/\*.*\*/',
        r'UNION.*SELECT',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, sql_upper):
            return False

    return True


def create_sample_database():
    """サンプルデータベースを作成"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        price INTEGER NOT NULL CHECK(price > 0),
        stock INTEGER NOT NULL CHECK(stock >= 0),
        category TEXT NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL CHECK(quantity > 0),
        unit_price INTEGER NOT NULL CHECK(unit_price > 0),
        total_amount INTEGER NOT NULL CHECK(total_amount > 0),
        sale_date DATE NOT NULL,
        customer_id INTEGER NOT NULL,
        sales_person TEXT,
        notes TEXT,
        FOREIGN KEY (product_id) REFERENCES products (id),
        FOREIGN KEY (customer_id) REFERENCES customers (id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        phone TEXT,
        address TEXT,
        customer_type TEXT CHECK(customer_type IN ('individual', 'business')),
        registration_date DATE DEFAULT (date('now')),
        total_purchases INTEGER DEFAULT 0,
        last_purchase_date DATE
    )
    ''')

    products = [
        ('iPhone 15 Pro', 159800, 15, 'スマートフォン', 'A17 Proチップ搭載の最新iPhone'),
        ('MacBook Air M3', 134800, 8, 'ノートPC', '13インチ、8GB RAM、256GB SSD'),
        ('iPad Pro 12.9', 128800, 12, 'タブレット', 'M2チップ搭載、12.9インチLiquid Retina XDRディスプレイ'),
        ('AirPods Pro 第3世代', 39800, 2, 'オーディオ', 'アクティブノイズキャンセリング搭載'),
        ('Apple Watch Series 9', 59800, 5, 'ウェアラブル', 'GPSモデル、45mm'),
        ('Magic Keyboard', 19800, 8, 'アクセサリ', 'iPad Pro用、バックライト付き'),
        ('iPhone 15', 124800, 25, 'スマートフォン', 'A16 Bionicチップ搭載'),
        ('iPad Air', 98800, 18, 'タブレット', 'M1チップ搭載、10.9インチ'),
        ('MacBook Pro 14インチ', 248800, 3, 'ノートPC', 'M3 Proチップ、16GB RAM、512GB SSD'),
        ('AirPods 第3世代', 19800, 30, 'オーディオ', '空間オーディオ対応')
    ]

    cursor.executemany('''
    INSERT OR IGNORE INTO products (name, price, stock, category, description)
    VALUES (?, ?, ?, ?, ?)
    ''', products)

    customers = [
        ('田中太郎', 'tanaka@example.com', '090-1234-5678', '東京都渋谷区', 'individual'),
        ('佐藤商事株式会社', 'sato@business.com', '03-1234-5678', '大阪府大阪市', 'business'),
        ('山田花子', 'yamada@example.com', '080-9876-5432', '愛知県名古屋市', 'individual'),
        ('鈴木システム', 'suzuki@tech.com', '045-111-2222', '神奈川県横浜市', 'business'),
        ('高橋一郎', 'takahashi@gmail.com', '070-5555-6666', '福岡県福岡市', 'individual')
    ]

    cursor.executemany('''
    INSERT OR IGNORE INTO customers (name, email, phone, address, customer_type)
    VALUES (?, ?, ?, ?, ?)
    ''', customers)

    sales_data = []

    for i in range(100):
        product_id = random.randint(1, 10)
        quantity = random.randint(1, 5)

        cursor.execute('SELECT price FROM products WHERE id = ?', (product_id,))
        unit_price = cursor.fetchone()[0]
        total_amount = unit_price * quantity

        days_ago = random.randint(0, 90)
        sale_date = (datetime.now() - timedelta(days=days_ago)).date()

        customer_id = random.randint(1, 5)
        sales_person = random.choice(['田中', '佐藤', '山田', '鈴木'])

        sales_data.append((
            product_id, customer_id, quantity, unit_price, total_amount,
            sale_date, sales_person, None
        ))

    cursor.executemany('''
    INSERT INTO sales
    (product_id, customer_id, quantity, unit_price, total_amount, sale_date, sales_person, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', sales_data)

    conn.commit()
    conn.close()
    print("Database created: intelligent_shop.db")


@mcp.tool()
def list_tables() -> List[Dict[str, Any]]:
    """データベース内のすべてのテーブルとスキーマ情報を一覧表示。"""
    conn = get_db_connection()
    cursor = conn.execute('''
    SELECT name, sql
    FROM sqlite_master
    WHERE type='table' AND name NOT LIKE 'sqlite_%'
    ORDER BY name
    ''')

    tables = []
    for row in cursor.fetchall():
        tables.append({
            "table_name": row["name"],
            "creation_sql": row["sql"]
        })

    conn.close()
    return tables


@mcp.tool()
def execute_safe_query(sql: str) -> Dict[str, Any]:
    """SELECTクエリのみを安全に実行。"""
    if not validate_sql_safety(sql):
        raise ValueError("安全でないSQL文です。SELECT文のみ実行可能です。")

    conn = get_db_connection()

    try:
        cursor = conn.execute(sql)
        results = [dict(row) for row in cursor.fetchall()]
        column_names = [description[0] for description in cursor.description] if cursor.description else []

        query_result = {
            "sql": sql,
            "results": results,
            "column_names": column_names,
            "row_count": len(results),
            "executed_at": datetime.now().isoformat()
        }

        conn.close()
        return query_result

    except sqlite3.Error as e:
        conn.close()
        raise ValueError(f"SQLエラー: {str(e)}")


if __name__ == "__main__":
    import sys
    if "--init" in sys.argv:
        create_sample_database()
    elif "--http" in sys.argv:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
    else:
        mcp.run()
