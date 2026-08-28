# pgvector と PGroonga を同居させた PostgreSQL。
#
# 公式イメージはどちらか一方しか入っていない。
#   pgvector/pgvector       pgvector のみ
#   groonga/pgroonga        PGroonga のみ
#
# ハイブリッド検索（ベクトル + 全文）を 1 台で試すには両方が要るため、
# pgvector のイメージに PGroonga を足す。
#
# groonga-tokenizer-mecab も入れる。無くても PGroonga は動くが、
# 既定の bigram（2 文字ずつ切る）方式になる。語の区切りを見ないため、
# 意図しない部分一致を拾う。形態素解析を使うには別途インストールが要る。
FROM pgvector/pgvector:pg17

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      wget lsb-release ca-certificates \
 && wget -q https://packages.groonga.org/debian/groonga-apt-source-latest-$(lsb_release --codename --short).deb \
 && apt-get install -y -V ./groonga-apt-source-latest-$(lsb_release --codename --short).deb \
 && rm groonga-apt-source-latest-$(lsb_release --codename --short).deb \
 && apt-get update \
 && apt-get install -y -V --no-install-recommends \
      postgresql-17-pgdg-pgroonga \
      groonga-tokenizer-mecab \
 && apt-get purge -y wget lsb-release \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*
