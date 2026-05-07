# SmartShop AI — Pipeline Deep Dive

A complete walkthrough of every stage in the SmartShop ML pipeline, from raw
Amazon review data to live inference endpoints. Written for someone seeing this
project for the first time.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Stage 0 — Raw Data Ingestion](#2-stage-0--raw-data-ingestion)
3. [Stage 1 — Feature Engineering (Feast Materialization)](#3-stage-1--feature-engineering-feast-materialization)
4. [Stage 2 — Model Training (Kubeflow TrainJob)](#4-stage-2--model-training-kubeflow-trainjob)
5. [Stage 3 — Embedding Generation (Milvus)](#5-stage-3--embedding-generation-milvus)
6. [Stage 4 — Model Serving (KServe)](#6-stage-4--model-serving-kserve)
7. [Infrastructure Layer](#7-infrastructure-layer)
8. [Data Lineage Summary](#8-data-lineage-summary)

---

## 1. The Big Picture

SmartShop AI is an e-commerce ML platform that does three things:

1. **Recommends products** — a Two-Tower neural model with category-aware
   embeddings (4M users, 1.8M items, 5 super-categories), served in real-time
   with sub-ms feature lookups from Redis.
2. **Summarizes reviews** — Mistral-7B fine-tuned with LoRA + FSDP on 1.4M
   Amazon reviews to generate concise review summaries.
3. **Answers product questions** — RAG pipeline that searches review
   embeddings in Milvus, then generates answers with the fine-tuned LLM.

```
                          ┌─────────────────────────────────────────────┐
                          │           SmartShop AI Pipeline             │
                          └─────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ Stage 0  │    │   Stage 1    │    │   Stage 2    │    │   Stage 3    │    │   Stage 4    │
  │          │    │              │    │              │    │              │    │              │
  │ Raw Data │───▶│   Feature    │───▶│   Model      │    │  Embedding   │    │   Model      │
  │ Ingest   │    │ Engineering  │    │  Training    │    │ Generation   │    │  Serving     │
  │          │    │              │    │              │    │              │    │              │
  │ HF → S3  │    │ Feast + Spark│    │ Kubeflow DDP │    │ Feast + ST   │    │ KServe       │
  └──────────┘    │ → Redis + S3 │    │ → S3 models  │    │ → Milvus     │    │ 3 endpoints  │
                  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       NB: 00_setup         NB: 01_data_pipeline    NB: 02_training     NB: 03_embeddings    NB: 04_serving
```

**Every stage maps to one demo notebook.** Run them in order.

---

## 2. Stage 0 — Raw Data Ingestion

**What:** Download the Amazon Reviews 2023 dataset from HuggingFace and store it
as Parquet files in MinIO (S3-compatible object storage).

**Who runs it:** Cluster admin, before demo users touch anything.

**How:** A Kubernetes Job streams data from HuggingFace Hub directly into MinIO.

### Input

| Source | Size | Records |
|---|---|---|
| [McAuley Lab Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | ~80 GB | ~233M reviews (33 categories) |

33 product categories are ingested and grouped into 5 super-categories for training:

| Super-Category | Example Categories |
|---|---|
| Tech & Computing | Electronics, Computers, Cell Phones |
| Books & Media | Books, Kindle, Movies |
| Home & Living | Home, Kitchen, Garden |
| Photography & Audio | Cameras, Headphones, Car Audio |
| DIY & Outdoors | Tools, Sports, Automotive |

**Scale:** ~233M reviews, ~4M unique users, ~1.8M items

### Output — S3 Bucket Layout

```
s3://smartshop-raw/
├── raw/
│   ├── reviews/
│   │   ├── Electronics/          # Parquet partitions
│   │   │   ├── part-00000.parquet
│   │   │   └── ...
│   │   ├── Books/
│   │   └── Home_and_Kitchen/
│   └── metadata/
│       ├── Electronics_meta/     # Product catalog
│       ├── Books_meta.parquet
│       └── Home_and_Kitchen_meta.parquet
```

### Raw Review Schema

Each review record looks like this:

| Column | Type | Example | Description |
|---|---|---|---|
| `user_id` | string | `AE22NQXDRA7N7` | Amazon reviewer ID |
| `parent_asin` | string | `B07XJ8C8F5` | Product ASIN (parent grouping) |
| `rating` | float | `5.0` | Star rating (1.0–5.0) |
| `title` | string | `Great product!` | Review headline |
| `text` | string | `Works perfectly...` | Full review body |
| `helpful_vote` | int | `3` | Upvotes on the review |
| `timestamp` | long | `1609459200000` | Review time (epoch milliseconds) |
| `verified_purchase` | bool | `true` | Amazon verified purchase flag |

### Raw Metadata Schema

| Column | Type | Example |
|---|---|---|
| `parent_asin` | string | `B07XJ8C8F5` |
| `title` | string | `Wireless Bluetooth Speaker` |
| `main_category` | string | `Electronics` |
| `price` | float | `29.99` |
| `store` | string | `JBL` |

### Notebook

`00_setup.ipynb` — validates that raw data exists in S3, checks infra connectivity.

---

## 3. Stage 1 — Feature Engineering (Feast Materialization)

**What:** Transform raw reviews into structured features and write them to:
- **Redis** — for real-time serving lookups (< 1ms)
- **S3 Parquet** — for training data reads (offline store)

**How:** Feast `@batch_feature_view` decorators define PySpark transformations.
Feast's `SparkComputeEngine` runs these on Kubernetes with Spark executors.

**Notebook:** `01_data_pipeline.ipynb`

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                  store.materialize(start, end)                      │
│                                                                     │
│  For each feature view:                                             │
│                                                                     │
│  1. READ ──── SparkSource.query (SQL over S3 parquet)               │
│       │                                                             │
│  2. TRANSFORM ── @batch_feature_view UDF (PySpark code)             │
│       │                                                             │
│  3. REPARTITION ── .repartition(200)                                │
│       │                                                             │
│  4. WRITE                                                           │
│       ├── online=True?  → foreachPartition → Redis (islice chunks)  │
│       └── offline=True? → spark_df.write.parquet → S3               │
└─────────────────────────────────────────────────────────────────────┘
```

The Spark driver runs inside the Feast pod. It spawns executor pods on
Kubernetes via `spark.master: k8s://`. Executors handle parquet reads, joins,
and aggregations.

### Feature Views — What Each One Does

#### 3.1 `interactions` — Training Labels

**Purpose:** User-item pairs with binary labels for the recommendation model.

**Routing:** `online=False, offline=True` → **S3 only**

Interaction rows don't fit in Redis (would consume too much memory). Training reads directly
from S3 parquet.

**Transformation logic:**

```
Raw review:
  user_id=AE22NQ  parent_asin=B07XJ8  rating=5.0

                    ↓ rating >= 4 → label = 1.0
                    ↓ rating <  4 → label = 0.0

Output:
  user_id=AE22NQ  item_id=B07XJ8  label=1.0  event_timestamp=...
```

No aggregation — one output row per review. The label is a binary positive
signal: did this user like this product?

**Sample data — BEFORE (raw reviews):**

| user_id | parent_asin | rating | title | text | helpful_vote | timestamp |
|---|---|---|---|---|---|---|
| AE22NQ | B07XJ8C8F5 | 5.0 | Great product! | Works perfectly, love it | 3 | 1609459200000 |
| AE22NQ | B09K3M2P1Q | 4.0 | Good value | Solid build quality | 1 | 1612137600000 |
| AE22NQ | B08N5WRWNW | 2.0 | Disappointing | Broke after 2 weeks | 0 | 1614556800000 |
| BKRMX7 | B07XJ8C8F5 | 3.0 | It's okay | Nothing special | 0 | 1611014400000 |
| BKRMX7 | B09K3M2P1Q | 5.0 | Amazing! | Best purchase this year | 7 | 1613692800000 |

**Sample data — AFTER (interactions parquet on S3):**

| user_id | item_id | label | event_timestamp |
|---|---|---|---|
| AE22NQ | B07XJ8C8F5 | 1.0 | 2021-01-01 00:00:00 |
| AE22NQ | B09K3M2P1Q | 1.0 | 2021-02-01 00:00:00 |
| AE22NQ | B08N5WRWNW | **0.0** | 2021-03-01 00:00:00 |
| BKRMX7 | B07XJ8C8F5 | **0.0** | 2021-01-19 00:00:00 |
| BKRMX7 | B09K3M2P1Q | 1.0 | 2021-02-19 00:00:00 |

> Notice: rating 5.0 and 4.0 → label 1.0 (liked). Rating 2.0 and 3.0 → label 0.0 (didn't like).
> Most columns are dropped — only the entity keys, label, and timestamp survive.

**Output schema:**

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Who reviewed |
| `item_id` | string | What was reviewed |
| `label` | float64 | 1.0 = liked (rating ≥ 4), 0.0 = didn't |
| `event_timestamp` | timestamp | When the review was written |

**Destination:** `s3a://smartshop-features/offline/interactions/`
**Row count:** ~5M (capped via `max_rows` training param)

**Use cases:**
- Primary training signal for the Two-Tower recommendation model
- Joined with `user_features` and `item_features` during training to form the
  complete training dataset: `interactions × user_features × item_features`
- The label split (~60% positive, ~40% negative) provides a natural class
  balance for binary cross-entropy training
- Not served online — training-only data

---

#### 3.2 `user_features` — Per-User Behavior Profile

**Purpose:** Aggregated statistics about each user's reviewing behavior.
Used by the recommendation model and serving layer.

**Routing:** `online=True, offline=True` → **Redis + S3**

**Transformation logic:**

```
Raw reviews (3 reviews by AE22NQ):
  AE22NQ  B07XJ8  5.0  "Great product!"        Electronics
  AE22NQ  B09K3M  4.0  "Good value"            Electronics
  AE22NQ  B08N5W  2.0  "Disappointing"         Home_and_Kitchen

                    ↓ groupBy("user_id")
                    ↓ avg(rating), count(*), countDistinct(items), ...
                    ↓ find most-reviewed category

Output (1 row):
  user_id              = AE22NQ
  user_avg_rating      = 3.67        ← average of [5, 4, 2]
  user_review_count    = 3           ← total reviews
  user_unique_items    = 3           ← distinct products
  user_avg_review_length = 18.3     ← mean chars of review text
  user_category_count  = 2           ← Electronics + Home_and_Kitchen
  user_tenure_days     = 59          ← days between first and last review
  user_primary_category = Electronics ← category with most reviews
```

**Sample data — BEFORE (5 raw reviews for 2 users):**

| user_id | parent_asin | rating | text | category (derived) |
|---|---|---|---|---|
| AE22NQ | B07XJ8C8F5 | 5.0 | Works perfectly, love it (24 chars) | Electronics |
| AE22NQ | B09K3M2P1Q | 4.0 | Solid build quality (19 chars) | Electronics |
| AE22NQ | B08N5WRWNW | 2.0 | Broke after 2 weeks (19 chars) | Home_and_Kitchen |
| BKRMX7 | B07XJ8C8F5 | 3.0 | Nothing special (15 chars) | Electronics |
| BKRMX7 | B09K3M2P1Q | 5.0 | Best purchase this year (23 chars) | Electronics |

**Sample data — AFTER (2 aggregated rows — one per user):**

| user_id | user_avg_rating | user_review_count | user_unique_items | user_avg_review_length | user_category_count | user_tenure_days | user_primary_category |
|---|---|---|---|---|---|---|---|
| AE22NQ | 3.67 | 3 | 3 | 20.7 | 2 | 59 | Electronics |
| BKRMX7 | 4.00 | 2 | 2 | 19.0 | 1 | 31 | Electronics |

> Notice: 5 raw review rows became 2 aggregated rows. All per-review detail
> (title, text, rating per item) is collapsed into summary statistics per user.
> The `category` column is derived from the file path, not a raw column.

**Output schema:**

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Entity key |
| `user_avg_rating` | float64 | Mean star rating |
| `user_review_count` | int64 | Total reviews |
| `user_unique_items` | int64 | Distinct products reviewed |
| `user_avg_review_length` | float64 | Mean review text length |
| `user_category_count` | int64 | Categories reviewed |
| `user_tenure_days` | int64 | Days between first and last review |
| `user_primary_category` | string | Most frequently reviewed category |

**Destinations:**
- Redis → serving reads `user_features` when scoring recommendations
- `s3a://smartshop-features/offline/user_features/` → training reads this

**Row count:** ~4M (one per unique user)

**Use cases:**
- **Training:** The User Tower of the Two-Tower model takes these 6 numeric
  features (plus a learned user embedding) as input. They help the model
  distinguish power reviewers from casual ones, identify category preferences,
  and learn rating tendencies
- **Serving:** When a user visits the app, the rec server fetches their
  `user_features` from Redis in < 1ms. Combined with candidate item features,
  the model scores relevance instantly
- **Personalization signals:** `user_primary_category` enables category-aware
  recommendations. `user_avg_rating` helps calibrate whether a user is
  generally positive or critical

---

#### 3.3 `item_features` — Per-Item Review Stats

**Purpose:** Product quality signals derived from aggregating all reviews for
each product.

**Routing:** `online=True, offline=True` → **Redis + S3**

**Transformation logic:**

```
Raw reviews (2 reviews for B07XJ8):
  B07XJ8  rating=5.0  helpful_vote=3  text="Works perfectly, love it"
  B07XJ8  rating=3.0  helpful_vote=0  text="It's okay"

                    ↓ groupBy("parent_asin")
                    ↓ avg(rating), stddev(rating), count(*), sum(helpful_vote), ...

Output (1 row):
  item_id                  = B07XJ8
  item_avg_rating          = 4.0        ← avg of [5, 3]
  item_rating_stddev       = 1.41       ← stddev of [5, 3]
  item_review_count        = 2          ← total reviews
  item_total_helpful_votes = 3          ← sum of upvotes
  item_avg_review_length   = 17.0       ← mean text length
```

**Sample data — BEFORE (4 raw reviews for 2 products):**

| parent_asin | rating | helpful_vote | text | timestamp |
|---|---|---|---|---|
| B07XJ8C8F5 | 5.0 | 3 | Works perfectly, love it (24 chars) | 1609459200000 |
| B07XJ8C8F5 | 3.0 | 0 | It's okay (9 chars) | 1611014400000 |
| B09K3M2P1Q | 4.0 | 1 | Solid build quality (19 chars) | 1612137600000 |
| B09K3M2P1Q | 5.0 | 7 | Best purchase this year (23 chars) | 1613692800000 |

**Sample data — AFTER (2 aggregated rows — one per product):**

| item_id | item_avg_rating | item_rating_stddev | item_review_count | item_total_helpful_votes | item_avg_review_length |
|---|---|---|---|---|---|
| B07XJ8C8F5 | 4.00 | 1.41 | 2 | 3 | 16.5 |
| B09K3M2P1Q | 4.50 | 0.71 | 2 | 8 | 21.0 |

> Notice: Individual review rows collapse into one row per product.
> `stddev` captures rating consistency — low stddev = consensus, high = polarizing.
> `helpful_votes` are summed, not averaged, to reward well-reviewed products.

**Output schema:**

| Column | Type | Description |
|---|---|---|
| `item_id` | string | Entity key (parent ASIN) |
| `item_avg_rating` | float64 | Mean star rating |
| `item_rating_stddev` | float64 | Rating variance |
| `item_review_count` | int64 | Total reviews |
| `item_total_helpful_votes` | int64 | Sum of helpful upvotes |
| `item_avg_review_length` | float64 | Mean review text length |

**Destinations:**
- Redis → serving uses these features in recommendation scoring
- `s3a://smartshop-features/offline/item_features/` → training reads this

**Row count:** ~2M (one per unique product)

**Use cases:**
- **Training:** The Item Tower of the Two-Tower model uses these 5 numeric
  features (plus a learned item embedding). They let the model learn that
  highly-rated products with many helpful votes are generally higher quality
- **Serving:** When scoring candidate products, `item_features` are fetched
  from Redis alongside `user_features` for real-time inference
- **Quality filtering:** `item_rating_stddev` is particularly useful — a
  product with avg 4.0 and stddev 0.2 is consistently liked, while avg 4.0
  and stddev 2.0 is polarizing. The model learns these quality signals
- **Popularity signal:** `item_review_count` and `item_total_helpful_votes`
  act as popularity proxies without needing explicit click/impression data

---

#### 3.4 `item_metadata` — Product Catalog

**Purpose:** Static product attributes for display and filtering. Not derived
from reviews — comes from the metadata parquet files.

**Routing:** `online=True, offline=True` → **Redis + S3**

**Transformation logic:**

```
Raw metadata:
  parent_asin=B07XJ8  title="Wireless Bluetooth Speaker"
  main_category=Electronics  price=29.99  store=JBL

                    ↓ rename columns, deduplicate by item_id

Output (1 row):
  item_id       = B07XJ8
  item_title    = Wireless Bluetooth Speaker
  item_category = Electronics
  item_price    = 29.99
  item_brand    = JBL
```

**Sample data — BEFORE (raw metadata parquet):**

| parent_asin | title | main_category | price | store | description | features | images |
|---|---|---|---|---|---|---|---|
| B07XJ8C8F5 | Wireless Bluetooth Speaker | Electronics | 29.99 | JBL | Portable speaker with... | [Waterproof, 12hr battery] | [{large: url, ...}] |
| B09K3M2P1Q | USB-C Hub Adapter | Electronics | 24.99 | Anker | 7-in-1 multiport... | [4K HDMI, PD charging] | [{large: url, ...}] |
| B07XJ8C8F5 | Wireless Bluetooth Speaker | Electronics | 29.99 | JBL | *(duplicate row)* | ... | ... |

**Sample data — AFTER (item_metadata — cleaned, deduplicated):**

| item_id | item_title | item_category | item_price | item_brand | event_timestamp |
|---|---|---|---|---|---|
| B07XJ8C8F5 | Wireless Bluetooth Speaker | Electronics | 29.99 | JBL | 2024-01-01 00:00:00 |
| B09K3M2P1Q | USB-C Hub Adapter | Electronics | 24.99 | Anker | 2024-01-01 00:00:00 |

> Notice: Duplicate B07XJ8C8F5 is removed. Heavy columns (description, features,
> images) are dropped — only columns needed for display and filtering remain.
> `parent_asin` → `item_id`, `store` → `item_brand`, `main_category` → `item_category`.

**Output schema:**

| Column | Type | Description |
|---|---|---|
| `item_id` | string | Entity key |
| `item_title` | string | Product name |
| `item_category` | string | Top-level category |
| `item_price` | float32 | Price (null if unavailable) |
| `item_brand` | string | Brand / store name |

**Destination:** Redis + `s3a://smartshop-features/offline/item_metadata/`
**Row count:** ~1.8M (one per unique product, deduplicated)

**Use cases:**
- **Serving (display):** When the rec model returns a list of item_ids,
  `item_metadata` is fetched from Redis to render product cards in the UI
  with title, price, brand, and category — without a separate catalog API call
- **Serving (RAG):** The RAG endpoint enriches LLM responses with product
  attributes from `item_metadata` for grounded answers
- **Filtering:** `item_category` enables category-scoped recommendations
  (e.g., "show me electronics only")
- **Training (optional):** Item category can be used as a feature during
  training for category-aware embeddings

---

### Summary: Before vs After Materialization

| Feature View | Raw Input | Rows In | Transform | Rows Out | Output Columns | Redis | S3 | Used By |
|---|---|---|---|---|---|---|---|---|
| `interactions` | Reviews (all categories) | 233M | Binary label from rating | 233M (1:1) | 4 cols | No | Yes | Training (labels) |
| `user_features` | Reviews (all categories) | 233M | groupBy user → aggregate | ~4M (1 per user) | 8 cols | Yes | Yes | Training + Serving |
| `item_features` | Reviews (all categories) | 233M | groupBy item → aggregate | ~1.8M (1 per item) | 6 cols | Yes | Yes | Training + Serving |
| `item_metadata` | Metadata parquet | ~1.8M+ | Rename + dedupe | ~1.8M (1 per item) | 5 cols | Yes | Yes | Serving (display + RAG) |

**Key transformations:**
- `interactions`: No aggregation — 1 review = 1 row. Drops all text/metadata, keeps only user+item+label
- `user_features`: Many-to-one aggregation — all reviews by a user collapse into a single profile row
- `item_features`: Many-to-one aggregation — all reviews for a product collapse into quality stats
- `item_metadata`: Passthrough with cleanup — rename columns, drop heavy nested fields, deduplicate

---

### What Happens to Redis After Materialization

After Stage 1, Redis contains:

| Feature View | Keys | ~Size | Key format |
|---|---|---|---|
| `user_features` | ~4M | ~1.5 GB | `HASH user_id=AE22NQ smartshop {feature_values}` |
| `item_features` | ~2M | ~300 MB | `HASH item_id=B07XJ8 smartshop {feature_values}` |
| `item_metadata` | ~1.8M | ~700 MB | `HASH item_id=B07XJ8 smartshop {feature_values}` |
| **Total** | ~37.5M | **~6 GB** | |

`interactions` is NOT in Redis — only in S3.

### What Happens to S3 After Materialization

```
s3://smartshop-features/
└── offline/
    ├── interactions/          # training labels
    │   ├── part-00000.parquet
    │   └── ...                # ~200 partitions
    ├── user_features/         # ~4M rows — user profiles
    ├── item_features/         # 2M rows — item stats
    └── item_metadata/         # ~1.8M rows — product catalog
```

### Key Technical Detail: The `islice` Fix

The Feast fork includes a critical fix to prevent executor OOM during Redis
writes. The original code loaded entire Spark partitions into Python memory
with `list(rows)`. With 200 partitions over 4M users, each partition is ~20K
rows — still enough to cause `MemoryError` at scale.

The fix uses `itertools.islice` to process rows in 5,000-row chunks:

```python
while True:
    chunk = list(islice(rows, 5_000))
    if not chunk:
        break
    # convert to Arrow → write to Redis
```

This keeps Python memory bounded regardless of partition size.

---

## 4. Stage 2 — Model Training (Kubeflow TrainJob)

**What:** Train two models using distributed GPU compute on Kubernetes.

**Notebook:** `02_training.ipynb`

### 4.1 Recommendation Model — Two-Tower DDP

**Architecture:** Two-Tower Neural Collaborative Filtering

```
                    ┌───────────────┐     ┌───────────────┐
                    │  User Tower   │     │  Item Tower   │
                    │               │     │               │
                    │  Embedding    │     │  Embedding    │
                    │      +        │     │      +        │
                    │  user_features│     │  item_features│
                    │      ↓        │     │      ↓        │
                    │  Linear(128)  │     │  Linear(128)  │
                    │  ReLU         │     │  ReLU         │
                    │  Dropout(0.2) │     │  Dropout(0.2) │
                    │  Linear(64)   │     │  Linear(64)   │
                    │  L2 Normalize │     │  L2 Normalize │
                    └──────┬────────┘     └──────┬────────┘
                           │                      │
                           └──────┬───────────────┘
                                  │ dot product
                                  ↓
                           similarity score
                                  ↓
                        Binary Cross-Entropy Loss
```

**What each tower takes as input:**

User Tower:
- `user_id` → learned embedding (64-dim)
- `user_avg_rating`, `user_review_count`, `user_unique_items`,
  `user_avg_review_length`, `user_category_count`, `user_tenure_days` → 6 floats

Item Tower:
- `item_id` → learned embedding (64-dim)
- `item_avg_rating`, `item_rating_stddev`, `item_review_count`,
  `item_total_helpful_votes`, `item_avg_review_length` → 5 floats + padding

**Training data:** Reads from Feast offline S3 parquet:

```python
interactions = spark.read.parquet("s3://smartshop-features/offline/interactions/")
user_feats   = spark.read.parquet("s3://smartshop-features/offline/user_features/")
item_feats   = spark.read.parquet("s3://smartshop-features/offline/item_features/")

# Join: interactions × user_features × item_features
training_df = interactions.join(user_feats, "user_id").join(item_feats, "item_id")
```

**Infrastructure:**

| Setting | Value |
|---|---|
| Framework | PyTorch DDP (`torchrun --nproc_per_node`) |
| Orchestration | Kubeflow TrainJob |
| GPUs | 4 × A100 (1 node) |
| Batch size | 2048 per GPU |
| Epochs | 10 |
| Optimizer | Adam, lr=0.0003 |
| Embed dim | 64 + 16 (category) |

**Output:** `s3://smartshop-models/recommendation/best_model.pt` (~1.6 GB)

---

### 4.2 LLM Fine-Tuning — Mistral-7B LoRA + FSDP

**Purpose:** Fine-tune Mistral-7B to generate concise review summaries.

**Why LoRA + FSDP:**
- Full fine-tune of 7B params needs ~112 GB GPU RAM (impossible on one A100)
- LoRA trains low-rank adapter matrices on top of the frozen base model
- FSDP shards model parameters across GPUs for efficient multi-GPU training
- Adapter checkpoint is ~168 MB vs ~28 GB for full fine-tune

**Infrastructure:**

| Setting | Value |
|---|---|
| Framework | PyTorch FSDP + LoRA (PEFT) |
| Orchestration | Kubeflow TrainJob |
| GPUs | 4 × A100 (1 node) |
| LoRA rank | r=16 (~70M trainable params, ~1% of model) |
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` |

**Output:** `s3://smartshop-models/llm-adapter/` (LoRA adapter, ~168 MB)

---

### Zero Training/Serving Skew

This is a critical design point. The exact same Feast feature definitions
(`features.py`) are used for:

- **Training:** reads `offline/user_features/` and `offline/item_features/`
  from S3 parquet — written by `store.materialize()`
- **Serving:** reads from Redis via `store.get_online_features()` — also
  written by `store.materialize()`

Same transformation logic, same schema, same data. No separate ETL for
training vs serving.

---

## 5. Stage 3 — Embedding Generation (Milvus)

**What:** Generate sentence embeddings from review text and store them in Milvus
vector database for semantic search.

**Notebook:** `03_embeddings.ipynb`

### How It Works

```
Raw reviews (S3)
      │
      ▼
SparkComputeEngine (k8s:// + GPU executors)
      │
      ▼
sentence-transformers/all-MiniLM-L6-v2
   (runs on GPU executor pods via pandas_udf)
      │
      ▼
384-dimensional embedding per review
      │
      ▼
Milvus vector store (IVF_FLAT, COSINE index)
```

### Feature View: `review_embeddings`

**Defined in:** `features_milvus.py` (separate from `features.py`)

**Transformation:**

```
Raw review:
  user_id=AE22NQ  parent_asin=B07XJ8
  title="Great Bluetooth speaker"
  text="Works perfectly, love the build quality and sound"

                    ↓ concat title + text
                    ↓ filter: length >= 20 chars
                    ↓ generate review_id = SHA256(user_id + asin + timestamp)

embed_text = "Great Bluetooth speaker Works perfectly, love the build quality..."

                    ↓ sentence-transformers encode (on GPU)

embedding = [0.023, -0.112, 0.045, ...] (384 floats)

Output:
  review_id  = a3f2b...
  item_id    = B07XJ8
  user_id    = AE22NQ
  rating     = 5.0
  review_title = "Great Bluetooth speaker"
  embed_text = "Great Bluetooth speaker Works perfectly..."
  embedding  = [0.023, -0.112, ...]  (384-dim)
```

**Output schema:**

| Column | Type | Description |
|---|---|---|
| `review_id` | string | SHA256 hash — entity key |
| `item_id` | string | Product ASIN |
| `user_id` | string | Reviewer |
| `rating` | float64 | Star rating |
| `review_title` | string | Review headline |
| `embed_text` | string | Concatenated title + body (first 511 chars) |
| `embedding` | float32[384] | MiniLM-L6-v2 sentence embedding |

**Destination:** Milvus vector store (IVF_FLAT, COSINE similarity)

This uses a separate `feature_store_milvus.yaml` config that points to the
Milvus online store instead of Redis.

---

## 6. Stage 4 — Model Serving (KServe)

**What:** Deploy three inference endpoints and wire them to the feature stores.

**Notebook:** `04_serving.ipynb`

### 6.1 Recommendation Server (`smartshop-rec`)

**What it does:** Given a user and a list of candidate items, scores each
user-item pair and returns ranked recommendations.

```
Client request:
  { "user_id": "AE22NQ", "candidate_items": ["B07XJ8", "B09K3M", "B08N5W"] }

                    ↓ Feast get_online_features()

Redis lookup (< 1ms):
  user_features for AE22NQ: avg_rating=3.67, review_count=3, ...
  item_features for B07XJ8: avg_rating=4.0, review_count=2, ...
  item_features for B09K3M: avg_rating=4.5, review_count=2, ...
  item_metadata for B07XJ8: title="Wireless Speaker", category="Electronics"
  ...

                    ↓ Two-Tower model forward pass

Score each pair:
  (AE22NQ, B07XJ8) → 0.92
  (AE22NQ, B09K3M) → 0.87
  (AE22NQ, B08N5W) → 0.31

                    ↓ Sort by score descending

Response:
  [
    { "item_id": "B07XJ8", "score": 0.92, "title": "Wireless Speaker", "category": "Electronics" },
    { "item_id": "B09K3M", "score": 0.87, "title": "USB-C Cable", "category": "Electronics" },
    ...
  ]
```

**Data sources at serving time:**
- Redis → `user_features` + `item_features` + `item_metadata`
- S3 → `best_model.pt` (loaded at startup)

---

### 6.2 LLM Server (`smartshop-llm`)

**What it does:** Generates review summaries using the fine-tuned Mistral-7B
with the LoRA adapter.

```
Client request:
  { "prompt": "Summarize reviews for this Bluetooth speaker: ..." }

                    ↓ vLLM inference engine
                    ↓ Base model: Mistral-7B (4-bit quantized)
                    ↓ + LoRA adapter (fine-tuned on Amazon reviews)

Response:
  "This Bluetooth speaker receives mostly positive reviews. Users praise its
   sound quality and build, though some note limited battery life..."
```

**Data sources at serving time:**
- S3 → Mistral-7B base model weights + LoRA adapter

---

### 6.3 RAG Server (`smartshop-rag`)

**What it does:** Answers product questions by searching relevant reviews in
Milvus, then generating an answer with the LLM.

```
Client request:
  { "question": "Is this speaker waterproof?", "item_id": "B07XJ8" }

                    ↓ Step 1: Encode question with MiniLM-L6-v2

query_embedding = [0.045, -0.023, ...]  (384-dim)

                    ↓ Step 2: Feast vector search in Milvus

Top-5 most similar reviews for B07XJ8:
  "Used it in the shower, works fine but not fully waterproof..."
  "Splashed water on it, no issues..."
  "Don't submerge it, it's only IPX5..."

                    ↓ Step 3: Build prompt with retrieved context

Prompt:
  "Based on these customer reviews, answer: Is this speaker waterproof?
   Context: [review 1] [review 2] [review 3]..."

                    ↓ Step 4: Send to LLM server for generation

Response:
  {
    "answer": "Based on customer reviews, the speaker is splash-resistant
              (IPX5) but not fully waterproof. Users report it works fine
              with minor water exposure but advise against submerging it.",
    "sources": [
      { "review_id": "a3f2b...", "text": "Used it in the shower...", "rating": 4.0 },
      ...
    ]
  }
```

**Data sources at serving time:**
- Milvus → vector similarity search over review embeddings
- Redis → item metadata for context enrichment
- LLM Server → text generation

---

## 7. Infrastructure Layer

### Storage

| Component | Role | Capacity |
|---|---|---|
| **MinIO** | S3-compatible object store | 200 Gi NFS |
| **Redis** | Online feature store (key-value) | 16 Gi |
| **Milvus** | Vector database for embeddings | 50 Gi NFS |
| **PostgreSQL** | MLflow backend + Feast registry | 10 Gi |

All pipeline stages read/write to MinIO S3. There are **zero data copies**
between stages — MinIO is the single source of truth.

### Compute

| Component | Role | Resources |
|---|---|---|
| **Spark Executors** | Feature engineering, embedding generation | 2 pods × (4 CPU, 20 Gi, 1 GPU) |
| **Feast Pod** | Spark driver + feature registry + online serving | 1 pod (8 CPU, 16 Gi) |
| **TrainJob Workers** | Distributed training | 1 pod × 4 A100 GPUs |
| **KServe Pods** | Model inference | Autoscale 0→5 replicas |

### Spark Configuration (k8s mode)

The Feast pod acts as the Spark driver. It spawns executor pods on Kubernetes
for distributed data processing.

| Setting | Value | Why |
|---|---|---|
| `spark.master` | `k8s://...` | Driver on Feast pod, executors as K8s pods |
| `spark.executor.instances` | 2–4 | Executor pods for parallel processing |
| `spark.executor.memory` | 6g | JVM heap |
| `spark.executor.memoryOverhead` | 14g | Python off-heap |
| `partitions` | 200 | Balances parallelism vs connection load |
| `spark.executor.maxNumFailures` | 16 | Resilience to transient failures |

### Networking (Spark Driver ↔ Executors)

```
Executor Pod ──────► feast-spark-driver Service ──────► Feast Pod (driver)
                     port 7078 (driver)                 container: registry
                     port 7079 (blockManager)
```

The `feast-spark-driver` Kubernetes Service provides a stable hostname for
executor → driver communication. Without it, executors can't send results
back to the driver.

---

## 8. Data Lineage Summary

Every piece of data in the pipeline can be traced back to the raw Amazon
reviews. Here's the complete lineage:

```
HuggingFace Hub
    │
    ▼
s3://smartshop-raw/raw/reviews/          ← Raw reviews (parquet)
s3://smartshop-raw/raw/metadata/         ← Product catalog (parquet)
    │
    ├──► interactions BFV ──► s3://smartshop-features/offline/interactions/
    │                              │
    │                              └──► Training (joins with user_features + item_features)
    │                                        │
    │                                        └──► s3://smartshop-models/recommendation/best_model.pt
    │                                                  │
    │                                                  └──► KServe: smartshop-rec
    │
    ├──► user_features BFV ──► Redis (35M keys) + s3://smartshop-features/offline/user_features/
    │                              │                    │
    │                              │                    └──► Training (user features)
    │                              └──► KServe: smartshop-rec (online lookup)
    │
    ├──► item_features BFV ──► Redis (2M keys) + s3://smartshop-features/offline/item_features/
    │                              │                    │
    │                              │                    └──► Training (item features)
    │                              └──► KServe: smartshop-rec (online lookup)
    │
    ├──► item_metadata BFV ──► Redis (1.8M keys) + s3://smartshop-features/offline/item_metadata/
    │                              │
    │                              └──► KServe: smartshop-rec + smartshop-rag (display)
    │
    ├──► review_embeddings BFV ──► Milvus (384-dim vectors, IVF_FLAT COSINE)
    │                              │
    │                              └──► KServe: smartshop-rag (vector similarity search)
    │
    └──► LLM training data ──► s3://smartshop-features/llm_data/
                                    │
                                    └──► Mistral-7B LoRA + FSDP fine-tune
                                              │
                                              └──► s3://smartshop-models/llm-adapter/
                                                        │
                                                        └──► KServe: smartshop-llm + smartshop-rag
```

### Key Design Principles

1. **Single storage layer** — MinIO is the only data store. No ETL copies.
2. **Zero train/serve skew** — Feast writes to both Redis and S3 from the same
   transformation. Training reads S3, serving reads Redis. Same data.
3. **GPU executors are optional** — Same PySpark code works with or without GPU
   executors. Drop-in replacement, zero code change.
4. **Chunked writes prevent OOM** — The `islice` fix bounds Python memory
   during Redis writes regardless of partition size.
5. **Offline-only views** — `interactions` skips Redis entirely (too large),
   writes only to S3 for training. The Feast fork allows `online=False,
   offline=True` views in `store.materialize()`.
