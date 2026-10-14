# COPY initial loading, partitioned indexes, and model evolution

`bulk_load.py` loads a stream of `(Item, embedding)` pairs using psycopg 3 binary
COPY, explicit `set_types`, and pgvector type registration. It is an administrative
initial loader for an **empty, unpartitioned, items-shaped table without ANN
indexes**. It does not replace the application's RLS-protected UPSERT.

The loader checks whether RLS applies to the actual role. COPY FROM is unsupported
in that case. A table owner is still subject to FORCE RLS unless it has a bypass;
policies remain enabled. Use an authorized administrative load, or a separate
staging table followed by INSERT under the application's RLS policy. COPY has
no ON CONFLICT/UPSERT behavior.

Within a transaction, it serializes initial writers, checks emptiness, copies
rows, optionally builds HNSW, and runs ANALYZE. A failure rolls back the whole
operation. It holds a write lock through that process, so use it before serving
traffic. If the caller owns an outer transaction, the caller must commit; a
rollback also removes loaded rows and the new index.

## Lab example

Prepare a separate table in a disposable database with the vector extension.
This statement intentionally fails if the table already exists.

```sql
CREATE TABLE bulk_items_demo (
    tenant_id text NOT NULL, item_id text NOT NULL, title text NOT NULL,
    description text NOT NULL DEFAULT '', category text,
    tags text[] NOT NULL DEFAULT '{}', embedding vector(3) NOT NULL,
    PRIMARY KEY (tenant_id,item_id)
);
```

From `recommend_server`, using its environment:

```sh
PG_OWNER_DSN="$TEST_DATABASE_URL" python -m pg.bulk_load \
  --synthetic --tenant lab --table bulk_items_demo --rows 10000 \
  --dimension 3 --build-index
```

`--synthetic` acknowledges that random vectors do not encode item text. Real
ingestion should pass precomputed model output to `load_initial_items`. Verify
model/version/preprocessing separately: the current items schema has no
embedding-model field, and a matching dimension cannot detect the wrong model.

The result separates COPY, index-build, and ANALYZE times. COPY time includes
iteration/conversion/validation; synthetic generation is included in this CLI.
It is not a pure transport benchmark and promises no speed multiple over INSERT.
For a fair comparison, prepare the same vectors before timing and compare equal
transaction boundaries and constraints. Memory is bounded by streaming input
rather than an all-rows list; the caller's iterator may itself hold more data.

`--maintenance-work-mem` defaults to 64MB and `--parallel-workers` to 0. Budget
CPU, memory, I/O and concurrent builds before increasing these. Worker pool limits
and container shared memory also constrain parallel builds. Settings apply to
the build transaction; a caller-owned outer transaction defines that lifetime.
The vector HNSW path supports at most 2,000 dimensions. A larger model needs a
different representation/indexing plan, such as halfvec, before opting in.

## Partition and model SQL experiments

Run on a lab database, **without wrapping the file in one transaction**:

```sh
psql -v ON_ERROR_STOP=1 "$TEST_DATABASE_URL" -f pg/examples/partitioned-hnsw.sql
psql -v ON_ERROR_STOP=1 "$TEST_DATABASE_URL" -f pg/examples/model-specific-indexes.sql
```

Each example first creates its own schema, fails if that name already exists,
and removes only that schema after successful assertions. A failed run may leave
its schema or an invalid index for inspection; review it before cleanup/retry.

The partition example asserts invalid parent before/after attaching the first
child, valid parent after all existing children, exclusive upper time bounds,
and automatic child-index creation for a new empty month. CONCURRENTLY permits
DML during construction but still has locks/waits and can leave invalid indexes
after failure. Every concurrent statement must be executed separately outside
a transaction, including when a driver sends SQL. Analyze the parent explicitly.
A scheduler for future months is an operational choice, not installed here.

The current `items` primary key and UPSERT target are `(tenant_id,item_id)`.
Range partitioning requires the partition key in parent uniqueness constraints;
partitioning on a changing `updated_at` can also move rows. These examples are
independent schemas, not a migration of the live item store.

The model example uses an untyped `vector` column with model/dimension CHECKs
and model-specific partial expression HNSW indexes. Queries match both the model
predicate and the cast. The CHECK explicitly lists two models; adding another
model also requires a matching constraint/index plan. A generic prepared plan
may not prove a parameterized partial-index predicate; inspect the execution
plan. Models with the same dimension still need their own semantic-space version.
Model-specific tables are another option for parallel evaluation and rollback.

## Validation

```sh
PG_BINARY_TEST_DSN="$TEST_DATABASE_URL" python -m pytest \
  pg/test_bulk_load.py pg/test_bench_binary.py pg/test_recommend.py -q
```

Tests cover binary float32 round trips, input/duplicate/build failures, caller
rollback, existing-row/index preservation, and rejection of FORCE-RLS owners.
Database tests skip explicitly if the lab DSN is absent. Role tests require
CREATEROLE in the disposable database; production application roles need no
such permission.

Input: provided material on partitioned HNSW construction, binary COPY and
embedding-model evolution. References: [pgvector loading](https://github.com/pgvector/pgvector/blob/v0.8.7/README.md#loading),
[psycopg binary COPY](https://www.psycopg.org/psycopg3/docs/basic/copy.html#binary-copy),
[PostgreSQL COPY/RLS](https://www.postgresql.org/docs/17/sql-copy.html),
[partitioned indexes](https://www.postgresql.org/docs/17/ddl-partitioning.html).
