"""
入力/出力ガードレール — AIエンジニアリング Ch10 Step 2 の実装。

3層防御:
  Layer 1: 正規表現による高速フィルタ（<1ms）
  Layer 2: ルールベースのPII検出
  Layer 3: 出力品質チェック
"""

import re


# Layer 1: プロンプトインジェクション検出（正規表現）
INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+instructions",
    r"disregard\s+(previous|above|all)\s+instructions",
    r"(system|admin)\s+prompt",
    r"you\s+are\s+now\s+.*(evil|bad|harmful)",
    r"前の指示を(無視|忘れ)",
    r"システムプロンプトを(表示|教えて|見せて)",
]

# Layer 2: PII検出パターン
PII_PATTERNS = {
    "phone": r"\d{2,4}-\d{2,4}-\d{4}",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "mynumber": r"\d{4}\s?\d{4}\s?\d{4}",
}


def check_injection(text: str) -> tuple[bool, str]:
    """Layer 1: プロンプトインジェクション検出。"""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True, f"プロンプトインジェクションの可能性を検出: {pattern}"
    return False, ""


def mask_pii(text: str) -> tuple[str, dict]:
    """Layer 2: PII情報をマスクし、逆引きマップを返す。"""
    reverse_map = {}
    masked_text = text

    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, masked_text)
        for i, match in enumerate(matches):
            placeholder = f"[{pii_type.upper()}_{i}]"
            reverse_map[placeholder] = match
            masked_text = masked_text.replace(match, placeholder, 1)

    return masked_text, reverse_map


def unmask_pii(text: str, reverse_map: dict) -> str:
    """出力からPIIマスクを元に戻す。"""
    for placeholder, original in reverse_map.items():
        text = text.replace(placeholder, original)
    return text


def check_output_quality(text: str) -> tuple[bool, str]:
    """Layer 3: 出力品質チェック。"""
    if not text or not text.strip():
        return False, "空の応答"
    if len(text.strip()) < 10:
        return False, "応答が短すぎます"
    return True, ""


def apply_input_guardrails(text: str) -> tuple[str, dict, list[str]]:
    """入力ガードレールを適用。マスク済みテキスト、逆引きマップ、警告を返す。"""
    warnings = []

    # Layer 1: インジェクション検出
    is_injection, msg = check_injection(text)
    if is_injection:
        warnings.append(msg)

    # Layer 2: PIIマスク
    masked_text, reverse_map = mask_pii(text)
    if reverse_map:
        warnings.append(f"PII検出・マスク済み: {len(reverse_map)}件")

    return masked_text, reverse_map, warnings


def apply_output_guardrails(
    text: str, reverse_map: dict
) -> tuple[str, list[str]]:
    """出力ガードレールを適用。処理済みテキストと警告を返す。"""
    warnings = []

    # Layer 3: 品質チェック
    is_ok, msg = check_output_quality(text)
    if not is_ok:
        warnings.append(msg)

    # PIIマスク解除
    unmasked = unmask_pii(text, reverse_map)

    return unmasked, warnings
