# PAAC-VN — Vietnamese Google Play App Clustering Benchmark
## Hướng dẫn triển khai và tái tạo kết quả

**Paper:** From Pairwise Alignment to Cluster Stability: A Paired Cross-Lingual Study of English and Vietnamese App Descriptions  
**Tác giả:** Lâm Trần Tấn Phát; Bùi Phúc Bình; Đinh Quốc Khánh — HUTECH, Vietnam  
**Email:** lamphat.344961@gmail.com

---

## Cấu trúc thư mục

```
Project3/
├── README.md                         ← File này
├── Project_3_final.pdf               ← Bài báo (file nộp)
├── CLAUDE.md                         ← Research protocol + cấu hình thí nghiệm
├── FROZEN.md                         ← SHA-256 dataset/config đã freeze
├── requirements-lock.txt             ← Phiên bản thư viện chính xác (pip freeze)
├── requirements.txt                  ← Phiên bản tối thiểu
│
├── pipeline/                         ← Source code
│   ├── models.yaml                   ← Cấu hình model (nguồn sự thật duy nhất)
│   ├── embed.py                      ← Tạo embedding (Phase C, cần GPU)
│   ├── cluster.py                    ← Phân cụm 1 configuration
│   ├── run_clustering_grid.py        ← Phân cụm toàn bộ 32 configurations
│   ├── evaluate.py                   ← Tính metrics (TN1/TN2-A/B/C/D)
│   ├── gate5_statistics.py           ← Bootstrap 1000 replicates
│   ├── k_sweep_saved_z.py            ← K-sweep phụ trên linkage Z đã lưu
│   ├── gate6_error_analysis.py       ← Co-assignment error analysis
│   ├── c4_paper_ready_export.py      ← Xuất bảng paper-ready từ kết quả
│   ├── audit_embeddings.py           ← Kiểm tra embedding artifacts
│   ├── audit_clustering.py           ← Kiểm tra clustering artifacts
│   ├── validate_models.py            ← Kiểm tra cấu hình models.yaml
│   ├── clean_dataset_v2.py           ← Làm sạch dữ liệu thô
│   ├── make_paired.py                ← Tạo paired dataset TN2
│   ├── tokenizer_audit.py            ← Audit tokenizer để chọn N_CHARS=950
│   ├── final_forensic_closure.py     ← Forensic closure audit
│   └── ...                           ← Các script phụ khác
│
├── data_v1/                          ← Dataset đã freeze
│   ├── benchmark_v1.parquet          ← TN1: 23,883 apps (95 MB)
│   ├── paired_en.parquet             ← TN2: 13,657 cặp EN/VI (46 MB)
│   ├── FROZEN.md                     ← SHA-256 xác nhận
│   └── report/                       ← Báo cáo dữ liệu
│
├── embeddings/                       ← Embedding đã tính sẵn (~1.9 GB)
│   ├── tn1/                          ← 16 configs × {.npy, .rows.parquet, .json}
│   └── tn2/                          ← 16 configs × {.npy, .rows.parquet, .json}
│
├── results/
│   ├── audit/                        ← Manifest files (cần cho clustering/evaluation)
│   │   ├── embedding_manifest.csv    ← 32/32 COMPLETE sau audit embeddings
│   │   └── clustering_manifest.csv  ← 32/32 COMPLETE sau audit clustering
│   ├── labels/                       ← Cluster labels đã tính (160 files)
│   ├── evaluation/                   ← Kết quả đánh giá
│   │   ├── metrics_long.csv          ← Bảng metrics đầy đủ (30,128 dòng)
│   │   ├── tn2_a_app_pairs.parquet   ← Delta-cos per-app evidence
│   │   └── evaluation_meta.json      ← Metadata + SHA256 integrity
│   ├── statistics/gate5/             ← Kết quả bootstrap
│   │   ├── bootstrap_summary.csv     ← Bảng CI và p-values (KẾT QUẢ CHÍNH)
│   │   └── bootstrap_seedwise_ami.parquet ← Seedwise AMI distributions
│   ├── figures/                      ← Figure k-sweep
│   ├── gate6/                        ← Error analysis
│   └── final_closure/C4_paper_ready/ ← Bảng sẵn sàng cho paper
│
└── tests/                            ← Unit tests
```

---

## Tài liệu nền và củng cố paper

Xem `docs/INDEX.md` để có mục lục đầy đủ. Tóm tắt nhanh:

| Thư mục | Nội dung |
|---------|---------|
| `docs/paper/` | Bản revised paper (`PAAC_VN_REVISED.pdf`), supplement (`SUPPLEMENT.pdf`), Vietnamese abstract |
| `docs/literature/` | 23 references đã verify (`VERIFIED_LITERATURE_MATRIX.md`, `REFERENCE_VERIFICATION.csv`), audit novelty |
| `docs/evidence/` | Bằng chứng cho từng luận điểm, truy ngược số liệu về file nguồn (`SECTION_EVIDENCE_BLUEPRINT.md`) |
| `docs/review/` | 10 phản biện dự kiến + trả lời (`REVIEWER_STRESS_TEST.md`), scope policy |
| `docs/decisions/` | Quyết định thiết kế thí nghiệm (N_CHARS=950, k=49, seed=42, changed-text control) |

---

## Cài đặt môi trường

**Python 3.10+** được khuyến nghị.

```bash
pip install -r requirements-lock.txt
```

Hoặc cài tối thiểu:
```bash
pip install numpy==1.26.4 scipy==1.12.0 scikit-learn==1.9.1 pandas==3.0.6 pyarrow==25.0.1
pip install torch==2.14.0 transformers==5.17.0 sentence-transformers==6.1.0
pip install pyvi==0.1.1 datasketch==2.0.0 fasttext-wheel==0.9.2 py3langid==0.2.2 pyyaml
```

**GPU** chỉ cần thiết nếu muốn chạy lại toàn bộ embedding từ đầu.

---

## QUAN TRỌNG: Về cơ chế write-once của các script

Tất cả script trong pipeline đều có **write-once guard** — từ chối overwrite kết quả đã tồn tại để đảm bảo tính toàn vẹn khoa học:

- `evaluate.py`: từ chối nếu `results/evaluation/metrics_long.csv` đã tồn tại
- `gate5_statistics.py --mode finalize`: từ chối nếu `bootstrap_summary.csv` đã tồn tại
- `c4_paper_ready_export.py`: từ chối nếu `C4_paper_ready/` không trống
- `gate5_statistics.py --mode pilot/prepare`: dùng exclusive-create, không overwrite

**Ý nghĩa thực tế:** Kết quả trong `results/` là final và không bị ghi đè ngẫu nhiên. Để chạy lại một phase, cần xóa output cũ trước — hướng dẫn cụ thể ở từng bước bên dưới.

---

## Kiểm tra kết quả ngay (không cần chạy gì)

Kết quả đã được tính sẵn và có thể đọc trực tiếp:

| File | Nội dung trong paper |
|------|---------------------|
| `results/final_closure/C4_paper_ready/tn2_vector_alignment.csv` | Table 3a — Delta-cos |
| `results/final_closure/C4_paper_ready/tn2_partition_shift.csv` | Table 3b — S deficit |
| `results/final_closure/C4_paper_ready/tn2_genre_alignment_point_summary.csv` | Table 4 — EN/VI AMI |
| `results/final_closure/C4_paper_ready/tn1_main_results.csv` | TN1 kết quả chính |
| `results/statistics/gate5/bootstrap_summary.csv` | Bootstrap CIs |
| `results/figures/k_sweep_tfidf_char_budget.png` | Figure k-sweep |
| `results/evaluation/metrics_long.csv` | Toàn bộ metrics raw |

---

## Tái tạo từng bước (từ embedding có sẵn — không cần GPU)

### Bước 1: Verify embeddings

Kiểm tra 32 embedding artifacts (SHA-256, n_rows, dtype, l2_normalized):

```bash
python pipeline/audit_embeddings.py --repo .
```

Output: `results/audit/embedding_manifest.csv` (ghi đè an toàn — audit_embeddings không có write-once guard).  
Expected: 32/32 COMPLETE.

### Bước 2: Clustering

Labels đã được tính và có trong `results/labels/`. Script resume-safe — **tự động bỏ qua** configurations đã hoàn thành:

```bash
python pipeline/run_clustering_grid.py --repo .
```

Script đọc `results/audit/embedding_manifest.csv` (cần tồn tại từ Bước 1).  
Expected: 32/32 skipped (đã hoàn thành).

**Để chạy lại từ đầu** (xóa labels cũ trước):
```bash
# Cảnh báo: xóa toàn bộ labels đã tính
Remove-Item results\labels\* -Recurse -Force   # PowerShell
# rm -rf results/labels/*                       # bash
python pipeline/run_clustering_grid.py --repo .
```

### Bước 3: Verify clustering

```bash
python pipeline/audit_clustering.py --repo .
```

Output: `results/audit/clustering_manifest.csv` (ghi đè).  
Expected: 32/32 COMPLETE.

### Bước 4: Evaluation (tính metrics)

Kiểm tra metadata mà không tính lại (nhanh, không ghi file):
```bash
python pipeline/evaluate.py --repo . --preflight
```

**Để tính lại metrics** (xóa outputs cũ trước):
```bash
# Cảnh báo: sau khi xóa, gate5 --mode pilot sẽ cần chạy lại
Remove-Item results\evaluation\metrics_long.csv
Remove-Item results\evaluation\tn2_a_app_pairs.parquet
Remove-Item results\evaluation\evaluation_meta.json
python pipeline/evaluate.py --repo .
```

### Bước 5: Bootstrap statistics (Gate 5)

Gate 5 có thể chạy lại hoàn toàn (các intermediate files đã được xóa khỏi thư mục):

```bash
# Bước 5a: Pilot — verify Gate4 inputs và point estimates
python pipeline/gate5_statistics.py --repo . --mode pilot

# Bước 5b: Prepare — tạo shared RNG indices (seed=42, tái tạo được)
python pipeline/gate5_statistics.py --repo . --mode prepare

# Bước 5c: Full — chạy B=1000 bootstrap (mất ~30-60 phút)
python pipeline/gate5_statistics.py --repo . --mode full

# Bước 5d: Finalize — tổng hợp kết quả
# (Cần xóa pre-computed outputs trước)
Remove-Item results\statistics\gate5\bootstrap_summary.csv
Remove-Item results\statistics\gate5\bootstrap_seedwise_ami.parquet
Remove-Item results\statistics\gate5\statistical_audit.json
python pipeline/gate5_statistics.py --repo . --mode finalize
```

**Lưu ý quan trọng cho gate5:**
- `--mode pilot` yêu cầu `results/evaluation/metrics_long.csv` và `results/audit/clustering_manifest.csv` phải nguyên vẹn (không xóa trước khi chạy pilot)
- `--mode finalize` yêu cầu **tất cả** block files trong `results/statistics/gate5/blocks/` phải tồn tại (được tạo bởi `--mode full`)
- `bootstrap_summary.csv` pre-computed đã có sẵn nên thầy không cần chạy lại

### Bước 6: K-sweep phụ

```bash
python pipeline/k_sweep_saved_z.py --repo . --repr tfidf_lsa --trunc char_budget
```

Output: `results/figures/k_sweep_tfidf_char_budget.{csv,png}`  
Dùng linkage Z từ TF-IDF char_budget đã lưu trong `results/labels/`.

### Bước 7: Error analysis

```bash
python pipeline/gate6_error_analysis.py --repo .
```

Output: `results/gate6/error_analysis/`

### Bước 8: Xuất bảng paper-ready

**Để chạy lại** (xóa outputs cũ trước):
```bash
Remove-Item results\final_closure\C4_paper_ready\* -Force
python pipeline/c4_paper_ready_export.py --repo .
```

---

## Tái tạo từ đầu bao gồm embedding (cần GPU)

**Ước tính thời gian:** 4–8 giờ với GPU NVIDIA (A100/V100 class).

Models sẽ tự động download từ HuggingFace theo revision hash trong `pipeline/models.yaml`.

```bash
# TN1 — TF-IDF (không cần GPU, ~5 phút)
python pipeline/embed.py --exp tn1 --repr tfidf_lsa --trunc char_budget
python pipeline/embed.py --exp tn1 --repr tfidf_lsa --trunc native

# TN1 — Neural (cần GPU, ~30-60 phút mỗi model)
python pipeline/embed.py --exp tn1 --repr multilingual_e5_large --trunc char_budget
python pipeline/embed.py --exp tn1 --repr multilingual_e5_large --trunc native
python pipeline/embed.py --exp tn1 --repr bge_m3 --trunc char_budget
python pipeline/embed.py --exp tn1 --repr bge_m3 --trunc native
python pipeline/embed.py --exp tn1 --repr qwen3_embedding_0_6b --trunc char_budget
python pipeline/embed.py --exp tn1 --repr qwen3_embedding_0_6b --trunc native
python pipeline/embed.py --exp tn1 --repr phobert_base_v2 --trunc char_budget
python pipeline/embed.py --exp tn1 --repr phobert_base_v2 --trunc native
python pipeline/embed.py --exp tn1 --repr videberta_base --trunc char_budget
python pipeline/embed.py --exp tn1 --repr videberta_base --trunc native
python pipeline/embed.py --exp tn1 --repr sup_simcse_phobert --trunc char_budget
python pipeline/embed.py --exp tn1 --repr sup_simcse_phobert --trunc native
python pipeline/embed.py --exp tn1 --repr bkai_vietnamese_biencoder --trunc char_budget
python pipeline/embed.py --exp tn1 --repr bkai_vietnamese_biencoder --trunc native

# TN2 — 4 representations × 2 languages × 2 rounds = 16 configurations
python pipeline/embed.py --exp tn2 --repr tfidf_lsa --lang en --trunc token_budget
python pipeline/embed.py --exp tn2 --repr tfidf_lsa --lang vi --trunc token_budget
python pipeline/embed.py --exp tn2 --repr tfidf_lsa --lang en --trunc native
python pipeline/embed.py --exp tn2 --repr tfidf_lsa --lang vi --trunc native
# ... tương tự cho multilingual_e5_large, bge_m3, qwen3_embedding_0_6b
```

---

## Dataset

| Dataset | File | Apps | Nhãn | Trường text |
|---------|------|------|------|-------------|
| TN1 | `benchmark_v1.parquet` | 23,883 | 49 genreId | `description_clean` |
| TN2 | `paired_en.parquet` | 13,657 cặp | 49 genreId | EN: `description_original_clean`, VI: `description_clean` |

SHA-256 xem trong `FROZEN.md`.

---

## Cấu hình models

Xem `pipeline/models.yaml` — bao gồm:
- Revision hash HuggingFace (frozen, đảm bảo reproducibility)
- Pooling method (mean_attention_mask / cls / pooler_output / last_token)
- Clustering prefix (E5 dùng `query: `)
- max_len và word segmentation requirement

---

## Liên hệ

- Lâm Trần Tấn Phát — lamphat.344961@gmail.com
- Bùi Phúc Bình — phucbinh3008@gmail.com
- Đinh Quốc Khánh — quockhanh092005@gmail.com
