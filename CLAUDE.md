# CLAUDE.md — PAAC-VN

> Vietnamese Google Play App Clustering Benchmark  
> **Vai trò:** research protocol + trạng thái dự án + kế hoạch thực thi hiện tại.  
> **Cập nhật:** 2026-10-01  
> **Trạng thái:** dataset/config đã freeze; người dùng xác nhận toàn bộ embedding đã hoàn tất; bước tiếp theo là audit artifact rồi chuyển sang clustering/evaluation.

---

# 0. Quy tắc vận hành

## 0.1. Thứ tự nguồn sự thật

Khi có mâu thuẫn, ưu tiên:

1. **Execution truth mới nhất**: output/result/meta từ lần chạy thực tế mới nhất.
2. **Code truth**: code đang dùng trong `pipeline/`.
3. **`pipeline/models.yaml`**: cấu hình model/tokenizer/pooling/prefix/revision.
4. **`data_v1/FROZEN.md`**: dataset/config đã đóng băng và SHA-256.
5. **`CLAUDE.md`**: research scope, RQ, protocol, kế hoạch và trạng thái tổng hợp.
6. Tài liệu lịch sử cũ chỉ để tham khảo.

Nếu code/output mới hơn khác mô tả lịch sử, **không tự sửa code để khớp tài liệu cũ**.

## 0.2. Không được tự thay đổi

- Chỉ có **2 thí nghiệm chính: TN1 và TN2**.
- Không tự thêm model, clustering algorithm, metric, branch, preprocessing hay RQ mới.
- Không thay `N_CHARS = 950`.
- Không thay TN2 common-token = 512.
- Không thay model revision, pooling, prefix, word segmentation, max length.
- Không thay `k = 49`.
- Không thay KMeans/Agglomerative protocol.
- Không thay paired bootstrap theo app.
- Không rerun embedding trừ khi audit chỉ ra artifact thiếu/invalid.
- Không dùng output lịch sử trong `archive/` làm kết quả chính thức.
- Không suy đoán khi thiếu dữ liệu: kiểm tra bằng script hoặc ghi `NOT VERIFIED`.

## 0.3. Reproducibility

- `SEED = 42`.
- Embedding lưu `float32`, L2-normalized.
- Model Hugging Face dùng revision hash đã freeze.
- Số liệu đưa vào paper phải truy ngược được về script/output chính thức.
- Dataset/config v1 không được silently overwrite; thay đổi phải tạo version mới.

---

# 1. Mục tiêu nghiên cứu

PAAC-VN nghiên cứu khả năng của các biểu diễn văn bản trong việc tổ chức **mô tả ứng dụng Google Play tiếng Việt** thành các cụm phù hợp với `genreId`, đồng thời khảo sát độ bền của các embedding đa ngôn ngữ khi cùng một app được biểu diễn bằng:

- listing tiếng Anh dùng làm nguồn cho bản dịch của Google;
- bản mô tả tiếng Việt do Google dịch.

Chỉ sử dụng trường **`description`**.

Không dùng `title`, `summary`, BERTopic, UMAP+HDBSCAN, GMM, LLM clustering, dataset cân bằng, TN1 human-only, provenance matching hoặc các nhánh ngoài 2 thí nghiệm đã freeze.

Nhãn tham chiếu: **`genreId` — 49 nhãn phẳng, đơn nhãn**.

Kết quả phải được diễn đạt là **mức phù hợp với danh mục Google Play**, không phải “độ chính xác phân loại”.

---

# 2. Câu hỏi nghiên cứu

## RQ1 — Representation + context budget

> Trên corpus mô tả ứng dụng tiếng Việt hỗn hợp nguồn, các biểu diễn văn bản tổ chức không gian ngữ nghĩa phù hợp với `genreId` đến mức nào, và việc chuyển từ common-content sang native context làm thay đổi kết quả ra sao?

Thí nghiệm: **TN1**.

## RQ2 — Robustness across English-source vs Vietnamese-MT

> Khi cùng một app được mã hóa từ listing tiếng Anh nguồn và từ bản tiếng Việt do Google dịch, vector embedding và cấu trúc phân cụm thay đổi đến mức nào?

Thí nghiệm: **TN2**.

Cách diễn đạt đúng:

> **robustness under English-source vs Vietnamese-MT text**

Không viết TN2 như một thiết kế nhân quả “đo tác động của dịch máy”.

---

# 3. Dataset canonical

## 3.1. TN1

- File: `data_v1/benchmark_v1.parquet`
- Số app canonical: **23,883**
- Nhãn: 49 `genreId`
- Text: `description_clean`
- Corpus gồm cả Google-translated và not-Google-translated.
- Không tách provenance thành thí nghiệm riêng.

SHA-256:

`0cf5e72c00f1392e9f7e83ef2fc282e8deb9e81bc6e5d332ce5d761a41f7c52b`

## 3.2. TN2

- File: `data_v1/paired_en.parquet`
- Số cặp canonical: **13,657**
- 49 `genreId`
- EN: `description_original_clean`
- VI: `description_clean`
- Hai nhánh cùng `appId`, cùng `genreId`.

SHA-256:

`9208bfc42fc66b45c19496af34088cb742d1ea2e7cca25998e139b16c7e77493`

TN2 chỉ giữ listing nguồn được detector xác nhận là English theo pipeline đã chốt.

## 3.3. Near-duplicate

TN1:

- MinHash-LSH
- word shingle = 3
- Jaccard threshold = 0.8
- `num_perm = 128`
- `seed = 42`
- 202 nhóm duplicate
- 575 app thuộc duplicate groups
- 373 app bị loại
- đại diện ưu tiên `maxInstalls`, sau đó `installs`, sau đó `appId`

TN2:

- dedup lại trên English-source text
- 54 nhóm
- 120 app
- 66 app bị loại

## 3.4. Provenance wording

Field lịch sử có thể dùng `human/machine/unknown`, nhưng:

> **`human` không được diễn giải là human-authored.**

Trong paper nên dùng nghĩa:

- `google_translated`
- `not_google_translated`
- `unknown`

hoặc giải thích rõ đây là tên field legacy.

---

# 4. Freeze state

Dataset/config v1 được freeze tại:

`2026-09-21T08:50:22.884000+00:00`

TN1:

**`N_CHARS = 950`**

| Artifact | SHA-256 |
|---|---|
| `data_v1/benchmark_v1.parquet` | `0cf5e72c00f1392e9f7e83ef2fc282e8deb9e81bc6e5d332ce5d761a41f7c52b` |
| `data_v1/paired_en.parquet` | `9208bfc42fc66b45c19496af34088cb742d1ea2e7cca25998e139b16c7e77493` |
| `pipeline/models.yaml` | `bdc8a5f2f72d699ff76dbedf5ceca015d5f2c2d2ff531c6289d900fd4b3e01d6` |
| `data_v1/report/tokenizer_audit.csv` | `92070f79888d24cc05a85b4d7e774be27f16864d1a557a8ff346fe44cf991877` |
| `data_v1/report/tokenizer_audit_summary.json` | `048311598641c301a4ebed8da90e1552bf092306e11280ee7c0ccf8d2694c895` |
| `requirements-lock.txt` | `4282f3e2dce62000cb3c923619226f11e2587a8546f3a946af4294cfa91b0a0f` |

Không thay đổi các artifact này trong main experiment.

---

# 5. Thiết kế thí nghiệm

## 5.1. TN1 — Vietnamese representation benchmark

Input:

`benchmark_v1.parquet → description_clean`

### Round 1 — `char_budget`

```text
description_clean
→ cắt tối đa 950 raw characters tại ranh giới từ gần nhất
→ PyVi nếu model yêu cầu
→ clustering prefix nếu model yêu cầu
→ tokenizer/analyzer
→ safety truncation tại native max_len
→ embedding
```

Mục tiêu: kiểm soát lượng raw textual content giữa representation.

`N_CHARS = 950` được chọn từ tokenizer audit full dataset theo tiêu chí:

> chọn N lớn nhất trong grid audit sao cho P95 token length của PhoBERT-family limiter không vượt 256.

### Round 2 — `native`

```text
description_clean
→ preprocessing riêng của model
→ tokenizer
→ truncation theo max_len trong models.yaml
→ embedding
```

Mục tiêu: đo representation với native context capacity của chính model.

Không mô tả native round là fair input budget.

## 5.2. TN2 — English-source vs Vietnamese-MT robustness

Input:

```text
EN = description_original_clean
VI = description_clean
```

Cùng app/cùng nhãn/cùng row order.

Chỉ dùng 4 representation:

- TF-IDF → SVD300
- multilingual-e5-large
- BGE-M3
- Qwen3-Embedding-0.6B

### Round 1 — `token_budget`

```text
max input units = 512
```

- neural model: 512 tokenizer units của chính model;
- TF-IDF: 512 token đầu do analyzer sinh ra;
- prefix nằm trong token budget.

### Round 2 — `native`

Dùng `max_len` frozen của mỗi model.

TN2 **không dùng `N_CHARS`**.

---

# 6. Representation frozen

`pipeline/models.yaml` là nguồn sự thật tuyệt đối cho implementation.

| ID | Model | TN1 | TN2 | max_len | pooling | PyVi | dim | clustering prefix |
|---|---|:---:|:---:|---:|---|:---:|---:|---|
| `tfidf_lsa` | TF-IDF → SVD300 | ✓ | ✓ | unbounded | SVD300 → L2 | no | 300 | none |
| `phobert_base_v2` | PhoBERT-base-v2 | ✓ | — | 256 | mean attention-mask | yes | 768 | none |
| `videberta_base` | ViDeBERTa-base | ✓ | — | 512 | mean attention-mask | yes | 768 | none |
| `sup_simcse_phobert` | Sup-SimCSE-PhoBERT | ✓ | — | 256 | `pooler_output` | yes | 768 | none |
| `bkai_vietnamese_biencoder` | BKAI Vietnamese Bi-Encoder | ✓ | — | 256 | mean attention-mask | yes | 768 | none |
| `multilingual_e5_large` | multilingual-e5-large | ✓ | ✓ | 512 | mean attention-mask | no | 1024 | `query: ` |
| `bge_m3` | BGE-M3 | ✓ | ✓ | 8192 | CLS, dense-only | no | 1024 | none |
| `qwen3_embedding_0_6b` | Qwen3-Embedding-0.6B | ✓ | ✓ | 32768 | last token | no | 1024 | none |

Correction bắt buộc:

1. **ViDeBERTa dùng PyVi preprocessing.**
2. **multilingual-E5-large dùng `query: ` cho clustering.**
3. **Qwen3-Embedding-0.6B native max_len = 32768.**
4. **Qwen document embedding không thêm query instruction.**
5. **BGE-M3 chỉ dùng dense representation.**

Không lấy mô tả model từ tài liệu cũ nếu khác `models.yaml`.

---

# 7. TF-IDF frozen recipe

```text
analyzer = word
ngram_range = (1, 2)
min_df = 5
max_df = 0.5
sublinear_tf = True
norm = l2
TruncatedSVD(n_components=300, random_state=42)
L2 normalize sau SVD
```

TN2:

> vectorizer và SVD fit **jointly trên EN ∪ VI**.

Không fit TF-IDF/SVD riêng theo language hoặc shard.

---

# 8. Embedding implementation

Code hiện tại: `pipeline/embed.py`.

Pipeline:

- load dataset canonical;
- pair-safe guard cho TN2;
- preprocessing theo model;
- truncation theo round;
- model inference;
- pooling frozen;
- L2 normalization;
- save embedding `float32`;
- save row sidecar;
- save metadata;
- log truncation rate.

Output:

```text
embeddings/{exp}__{repr}__{lang}__{trunc}.npy
embeddings/{exp}__{repr}__{lang}__{trunc}.rows.parquet
embeddings/{exp}__{repr}__{lang}__{trunc}.json
```

- `.npy` = embedding matrix
- `.rows.parquet` = `row_idx ↔ appId ↔ genreId`
- `.json` = model/config/hash/runtime/truncation metadata

### Implementation note

Phiên bản `embed.py` hiện trong project **không có checkpoint/resume**. Neural embeddings được gom theo batch trong RAM rồi `np.concatenate()` và save khi job hoàn thành.

Embedding phase đã hoàn tất, nên không sửa pipeline chỉ để thêm checkpoint/resume trừ khi cần rerun một artifact invalid.

---

# 9. Trạng thái embedding

Expected:

- TN1: `8 × 2 = 16`
- TN2: `4 × 2 languages × 2 rounds = 16`
- Tổng: **32 embedding sets**

**Người dùng xác nhận toàn bộ embedding đã chạy hoàn tất.**

Trước clustering phải audit:

- đủ `.npy`;
- đủ `.rows.parquet`;
- đủ `.json`;
- `n_rows` đúng;
- `embedding_dim` khớp `models.yaml`;
- revision khớp;
- `dataset_sha256` khớp `FROZEN.md`;
- `models_yaml_sha256` khớp;
- TN2 EN/VI cùng config có cùng `app_ids_sha256`;
- dtype = `float32`;
- `l2_normalized = true`.

Chỉ sau audit pass mới ghi:

**VERIFIED COMPLETE**

Không rerun toàn bộ embedding để “chắc chắn”.

---

# 10. Clustering protocol

## 10.1. General

- embedding L2-normalized
- `k = 49`

## 10.2. KMeans chính

```text
KMeans(
    n_clusters=49,
    n_init=10,
    random_state=s
)
s ∈ {42,...,51}
```

Mỗi embedding configuration:

- 10 outer seeds;
- `n_init=10` mỗi run.

`±std` là biến thiên của **best-of-10 KMeans**, không phải raw single-init.

## 10.3. KMeans diagnostic

```text
n_init = 1
same 10 outer seeds
```

Chỉ dùng đánh giá initialization stability.

## 10.4. Agglomerative

Frozen:

```text
average linkage
cosine distance
k = 49
deterministic
```

Implementation bắt buộc:

1. X đã L2-normalized;
2. cosine distance tính blockwise bằng BLAS;
3. ghi vào SciPy condensed buffer `float64`;
4. không tạo square `(n,n)` matrix;
5. không dùng sklearn precomputed;
6. không dùng `pdist(X, "cosine")` cho full pipeline;
7. `scipy.cluster.hierarchy.linkage(..., method="average")`;
8. `fcluster(..., 49, criterion="maxclust")`.

Giữ linkage matrix `Z` cho k-sweep.

Agglomerative deterministic → không báo ±std theo seed.

---

# 11. Evaluation protocol

## 11.1. TN1

External:

- AMI
- ARI
- NMI
- V-measure
- macro-F1 sau Hungarian mapping
- ACC sau Hungarian mapping

Internal:

- Silhouette
- Davies–Bouldin
- Calinski–Harabasz

Báo thêm cluster-size distribution.

Internal metrics **không dùng để rank representation giữa các embedding spaces khác nhau**.

## 11.2. TN2-A — vector robustness

```text
cos(E_EN_i, E_VI_i)
1 - cosine
```

Negative-pair baseline:

```text
cos(E_EN_i, E_VI_j), j != i
```

và:

```text
Δcos = matched cosine - median negative-pair baseline
```

Không gọi negative-pair distribution là null hypothesis.

## 11.3. TN2-B — clustering robustness

KMeans:

- cross-language all-pairs: 100
- within-VI all-pairs: 45
- within-EN all-pairs: 45
- matched-seed: 10, chỉ phụ

Metrics:

- AMI
- ARI
- NMI

Within-language distribution gọi là:

> empirical seed-stability baseline

không gọi là ceiling.

## 11.4. TN2-C — genre alignment

Cho EN và VI riêng:

- AMI
- ARI
- NMI
- V-measure
- macro-F1
- ACC

Sau đó tính hiệu theo convention thống nhất.

## 11.5. TN2-D — geometry

- Silhouette
- Davies–Bouldin
- Calinski–Harabasz

Chỉ diễn giải trong cùng embedding space/configuration.

---

# 12. Statistical inference

## 12.1. Primary — paired bootstrap trên app

```text
B = 1000
unit = app
same sampled indices for compared representations
do NOT recluster
```

Mỗi bootstrap:

- resample app indices with replacement;
- giữ nguyên cluster labels;
- tính lại metric;
- lấy paired difference.

Báo 95% percentile CI và distribution của Δ.

Phạm vi:

> bất định của metric trên một partition đã cho.

## 12.2. Secondary — Wilcoxon TN2-C

Có thể dùng paired Wilcoxon trên 10 KMeans outer seeds để hỏi:

> hiệu EN−VI có nhất quán qua initialization hay không?

Không xem 10 seed là 10 dataset độc lập.

## 12.3. Không dùng

- Friedman
- Nemenyi

---

# 13. k-sweep phụ

Chỉ trên cấu hình tốt nhất sau kết quả chính:

```text
k ∈ {30, 40, 49, 60, 80}
```

Mục tiêu:

> kiểm tra internal structure có nghiêng mạnh về một k khác hay không, trong khi k chính vẫn = 49 theo label space.

Chỉ một hình phụ.

---

# 14. Trạng thái dự án

## Phase A — Data
**COMPLETE / FROZEN**

## Phase B — Model recipe
**COMPLETE / FROZEN**

## Phase C — Embedding
**USER-CONFIRMED COMPLETE**

Expected 32 sets. Không rerun embedding ở bước hiện tại.

## Phase D — Embedding artifact audit
**NEXT GATE**

Audit đủ 32 configurations.

Output nên có:

`results/audit/embedding_manifest.csv`

với:

```text
experiment
representation
language
truncation
npy_exists
rows_exists
json_exists
n_rows
dim
dataset_sha256
models_yaml_sha256
app_ids_sha256
revision
dtype
l2_normalized
status
```

PASS → khóa embedding phase.

## Phase E — Clustering
**NOT STARTED / NEXT AFTER AUDIT**

1. KMeans `n_init=10`, 10 seeds
2. KMeans diagnostic `n_init=1`, 10 seeds
3. Agglomerative average-cosine
4. giữ `Z`
5. k-sweep phụ sau khi biết best configuration

## Phase F — Evaluation
**NOT STARTED**

1. TN1 metrics
2. TN2-A
3. TN2-B
4. TN2-C
5. TN2-D
6. cluster-size analysis
7. confusion/error analysis

## Phase G — Statistics
**NOT STARTED**

1. paired bootstrap B=1000
2. Wilcoxon TN2-C phụ
3. uncertainty tables

## Phase H — Paper
**NOT STARTED**

1. novelty/literature verification
2. methodology
3. results
4. discussion
5. limitations
6. tables/figures
7. venue/template

---

# 15. Plan từ trạng thái hiện tại

## Gate 1 — Audit embedding

Chứng minh 32 embedding sets hoàn chỉnh, đúng dataset/config và đúng row pairing.

PASS → không quay lại embedding.  
FAIL → chỉ rerun configuration lỗi.

## Gate 2 — Verify clustering pipeline

Trước full run:

- smoke-test 1 configuration;
- kiểm tra row order;
- labels length;
- KMeans output;
- condensed cosine implementation;
- RAM Agglomerative.

Không đổi protocol vì implementation convenience.

## Gate 3 — Full clustering grid

Sinh labels + metadata toàn lưới.

Không nhìn kết quả giữa chừng để đổi model/parameter.

## Gate 4 — Evaluation

Sinh một long-format metrics table duy nhất làm source cho tables/figures.

## Gate 5 — Statistical analysis

Bootstrap/Wilcoxon trên outputs đã cố định.

## Gate 6 — Interpretation

Sau khi result hoàn tất mới:

- chọn best configuration;
- error analysis;
- k-sweep;
- discussion.

## Gate 7 — Paper

Mọi claim phải truy ngược được về result/table/figure chính thức.

---

# 16. Cây thư mục mục tiêu

```text
PAAC-VN/
├── CLAUDE.md
├── requirements-lock.txt
├── README.md
├── MODEL_LOCK_NOTES.md
│
├── pipeline/
│   ├── models.yaml
│   ├── validate_models.py
│   ├── tokenizer_audit.py
│   ├── clean_dataset_v2.py
│   ├── make_paired.py
│   ├── embed.py
│   ├── cluster.py
│   ├── evaluate.py
│   ├── stats_tests.py
│   └── make_tables.py
│
├── data_v1/
│   ├── benchmark_v1.parquet
│   ├── paired_en.parquet
│   ├── FROZEN.md
│   └── report/
│       ├── tokenizer_audit.csv
│       ├── tokenizer_audit_summary.json
│       ├── truncation_rates.csv
│       └── ...
│
├── embeddings/
│   ├── *.npy
│   ├── *.rows.parquet
│   └── *.json
│
├── results/
│   ├── audit/
│   ├── labels/
│   ├── metrics.csv
│   ├── stats/
│   └── figures/
│
└── paper/
    ├── tables/
    └── figures/
```

---

# 17. Environment frozen

`requirements-lock.txt` là reference environment v1.

```text
numpy==1.26.4
pandas==3.0.6
pyarrow==25.0.1
scikit-learn==1.9.1
scipy==1.12.0
torch==2.14.0
transformers==5.17.0
sentence-transformers==6.1.0
pyvi==0.1.1
datasketch==2.0.0
fasttext-wheel==0.9.2
py3langid==0.2.2
```

Không upgrade dependency giữa main experiments nếu không có lý do bắt buộc.

---

# 18. Known constraints / interpretation rules

1. TN1 là corpus hỗn hợp nguồn; không tách riêng ảnh hưởng translation provenance.
2. TN2 chỉ áp dụng cho 4 representation trong design.
3. TN1 common-content dùng **characters**, TN2 common-token dùng **tokens** → không so trực tiếp budget.
4. Vietnamese MT thường dài hơn English source; TN2 token-budget có asymmetric content truncation.
5. English source listing không đảm bảo là “human-original”.
6. `genreId` có label noise.
7. Dataset phản ánh collection design, không đại diện toàn bộ Google Play market.
8. Internal metrics không rank model giữa spaces khác geometry/dimension.
9. KMeans std là variability của best-of-10.
10. Agglomerative average-linkage có thể tạo một cụm lớn + nhiều cụm nhỏ; phải báo cluster-size distribution.

---

# 19. Error-analysis pairs

- `SPORTS ↔ GAME_SPORTS`
- `EDUCATION ↔ GAME_EDUCATIONAL`
- `MUSIC_AND_AUDIO ↔ GAME_MUSIC`
- `TOOLS ↔ PRODUCTIVITY`
- `ENTERTAINMENT ↔ VIDEO_PLAYERS`
- `BUSINESS ↔ FINANCE`

Chỉ phân tích sau khi metrics chính hoàn tất.

---

# 20. Quy tắc cho AI/CLI session mới

Khi bắt đầu:

1. đọc `CLAUDE.md`;
2. đọc `data_v1/FROZEN.md`;
3. đọc `pipeline/models.yaml`;
4. nếu làm embedding/audit → đọc `pipeline/embed.py`;
5. nếu làm clustering → đọc `pipeline/cluster.py`;
6. nếu làm evaluation → đọc `pipeline/evaluate.py` và `stats_tests.py`.

Không load toàn bộ `.npy`/`.parquet` chỉ để hiểu project.

Dataset lớn:

- ưu tiên metadata;
- dùng `pyarrow.parquet.ParquetFile`;
- không materialize dataframe nếu không cần.

Embedding:

- ưu tiên `.json`;
- `.rows.parquet` metadata/schema;
- chỉ `np.load(..., mmap_mode="r")` khi cần shape/norm/sample verification.

---

# 21. Current next action

> **AUDIT 32 embedding artifacts → nếu PASS, khóa embedding phase → bắt đầu clustering.**

Không mở research branch mới trước khi:

1. embedding audit pass;
2. clustering hoàn thành;
3. evaluation hoàn thành;
4. statistical analysis hoàn thành.

Mục tiêu từ đây là **hoàn tất benchmark đã thiết kế**, không mở rộng scope.

---

# 22. Project Context nên giữ

Tối thiểu:

```text
CLAUDE.md
data_v1/FROZEN.md
pipeline/models.yaml
pipeline/embed.py
requirements-lock.txt
```

Nên thêm:

```text
MODEL_LOCK_NOTES.md
data_v1/report/tokenizer_audit_summary.json
HANDOFF-PAAC-VN.md
```

`CLAUDE.md` = protocol + project plan.  
`HANDOFF-PAAC-VN.md` = trạng thái thực thi ngắn hạn, nếu cần.

---

# Gate 5 Protocol Amendment — PAAC-GATE5-AMENDMENT-2026-10-04

**Approved/amended at:** `2026-10-04T14:35:33+07:00` (Asia/Ho_Chi_Minh, BEFORE any Gate 5 full bootstrap run).
**Status:** approved Gate 5 G1–G3, fixed after seeing Gate 4 point estimates but before any Gate 5 uncertainty estimates. This appendix supersedes conflicting Gate5 statistical/k-sweep selection wording in §§12–13; frozen dataset/model/embedding/clustering/Gate4 artifacts remain unchanged.

## Gate 5: estimands, comparisons, and multiplicity

- B=1000, RNG master seed 42; paired *app* bootstrap, one common persisted index matrix for TN1 (23,883 rows) and one for TN2 (13,657 rows), shared over all representations, languages, seeds, and comparison arms per experiment. Labels already frozen; no reclustering. 10 main best-of-10 KMeans seeds 42…51 are **fixed partitions**, averaged within each replicate; `kmeans_diag` excluded. Agglomerative retained descriptive-only.
- Primary endpoint AMI: TN1-A native − char (8); TN2-B mean cross EN–VI 100 AMI pairs − average of EN 45 and VI 45 *within* AMI pairs (8); TN2-C matched EN − VI, averaged over ten seeds (8). TN2-A endpoint is matched cosine − all-negative *per-app* median, mean over bootstrapped apps (8). TN2-A uses the saved app-pair table, and is a catalogue-relative contrast, NOT a null distribution.
- Supplementary/exploratory: TN1-B 7 non-TFIDF representations versus TF-IDF within each round (14); six predetermined E5/BGE/Qwen pairs, three per round. Model-triplet designated after viewing Gate4 point estimates, thus exploratory. Do not bootstrap TN1-C KMeans vs Agglomerative; report collapsed partitions/cluster sizes descriptively.
- Point estimates **must exactly reconcile** with Gate 4 saved seed-level TN1/TN2-C/TN2-B metrics and TN2-A mean (float tolerance 1e-8) during a mandatory **pilot**. A source mismatch BLOCKS full inference (record as SOURCE_DISCREPANCY); no score optimization. `--mode full` MUST verify pilot PASS and matching script, sources and this appendix hash.
- Primary 95% percentile CI; 90% CI for equivalence. Operational SESOI ±0.01 **AMI** for TN1-A and TN2-C. TOST nominal when 90% percentile CI is entirely inside the equivalence limits. For six exploratory top-three pairs, report *unadjusted* TOST and CI only, no confirmatory equivalence. Illustrative ±0.005 AMI sensitivity (not a replacement for primary).
- Percentile-bootstrap p-values must be obtained by **inverting the empirical percentile confidence limits**: two-sided p = 2·min(P(Δ*≤0),P(Δ*≥0)), positive-direction TN2-A p = P(Δ*≤0), one-sided equivalence p_L=P(Δ*≤−0.01), p_U=P(Δ*≥+0.01), p_TOST=max(p_L,p_U). Empirical tails have B=1000 discrete resolution, are NOT guaranteed to be null-calibrated; do unit tests (no independent calibration study). Holm stepdown FWER within each 8-member primary *difference* family (TN1-A, TN2-A, TN2-B, TN2-C), and separately within TN1-A and TN2-C 8-member TOST families. No global FWER claim across families. Preserve raw/bootstrap distributions.
- Sensitivity: reuse app × seed × replicate AMI computed for primary. Resample the *10 seed slots* with replacement, reweight seedwise AMI without refit. TN2-B uses cross weights w_i·w_j, and distinct original within-seed pair weights w_i·w_j (i<j), without treating duplicated seed slots as new partitions; record any undefined sensitivity replicate. Seed sensitivity is heuristic, not evidence of independent seed datasets. Never conceal disagreement between primary and sensitivity equivalence evidence.
- Outputs: save ALL AMI by experiment/configuration/language/seed/replicate, all paired differences, app bootstrap index matrices, seed weights, config × bootstrap block checkpoints (mandatory for TN2-B), script/inputs/hash/version manifest, and explicit VALID / UNDEFINED_REPLICATE / SOURCE_DISCREPANCY audit states. Full outputs are immutable, restart only missing blocks with identical manifests. No bootstrap/wilcoxon results allowed to revise Gate4 data.

## Gate 6: k-sweep scope and fixed selection

- Fixed supplementary k values `{30,40,49,60,80}`, one figure. **Use Agglomerative saved linkage Z for TN1 TF-IDF LSA char_budget**, not E5 KMeans best. This is highest **TN1 Agglomerative AMI at k=49** in the current Gate4 long table (TF-IDF char_budget 0.389706); E5 char_budget agglomerative solution collapses into a dominant cluster (reported approx. 99%), making it poor for studying varying k. Choosing by existing Gate4 AMI is a descriptive post-result choice, not a confirmatory statistical discovery. No refitting embeddings, KMeans or linkage. Report observed number of clusters, cluster sizes and internal metrics for **the same TF-IDF embedding space only**; do not use k-sweep to retune main k=49 or to compare internal indices between embedding spaces.

## Runtime and forensic parallelism

- Gate4 computation COMPLETE; additional A1-C/A2/A3 forensic assurance independent of starting Gate5; any material discrepancy must be documented before final claims. Do not imply the statistical amendment existed before Gate4 point estimates. Full bootstrap launch must occur strictly *after* this timestamp; date/version and file SHA recorded in Gate5 manifest.
