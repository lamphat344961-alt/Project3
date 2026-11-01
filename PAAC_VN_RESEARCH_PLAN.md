# PAAC-VN — KẾ HOẠCH TRIỂN KHAI NGHIÊN CỨU VÀ HOÀN THIỆN BÀI BÁO

**Phiên bản kế hoạch vận hành:** 1.1 — 24/10/2026

**Tình trạng:** Đang triển khai theo phê duyệt từng phần; tác giả đã phê duyệt bỏ Chặng 04/R02 ngày 24/10/2026.
**Cơ sở:** `Project_3_final.pdf`, các nhận xét trong `review.md` được nhắc đến trong V4.2.  
**Phạm vi tài liệu:** Xác định *làm gì, để đạt mục tiêu nào, bằng đầu ra nào*. Vai trò và prompt riêng của Codex/Claude sẽ được quy định sau, không nằm trong kế hoạch này.

## 1. Mục tiêu chung

Hoàn thiện nghiên cứu **PAAC-VN** và sửa bài báo từ bản thảo hiện có dựa trên phản biện, bằng các thực nghiệm có thể truy vết và kết luận đúng phạm vi bằng chứng.

- **RQ1:** Đánh giá mức tương ứng giữa các mô tả tiếng Anh–tiếng Việt của cùng ứng dụng: vector correspondence, retrieval, sự đồng thuận phân hoạch và mức phù hợp taxonomy theo ngôn ngữ.
- **RQ2:** Đánh giá các representation/input recipes đối với taxonomy của Google Play trên benchmark tiếng Việt, có xem xét input conditions và các nhóm provenance được ghi nhận.
- **R02 — loại khỏi phạm vi theo quyết định tác giả ngày 24/10/2026:** Không thực hiện đối chứng back-translation EN–EN_BT, không tạo EN_BT/embeddings tương ứng và không báo cáo S_BT hoặc D=S_XL−S_BT. RQ1/RQ2 giữ nguyên; các tương phản EN–VI chỉ phản ánh quan hệ quan sát được, không tách hiệu ứng dịch máy, diễn đạt lại và thay đổi ngôn ngữ.

**Sản phẩm cuối:** mã nguồn và cấu hình có thể tái lập; các kết quả đã kiểm chứng; bảng/hình phân tích; bản thảo LaTeX/PDF được sửa; tài liệu phản hồi đầy đủ nhận xét reviewer.

## 2. Phạm vi và các quyết định cần giữ

### Các ràng buộc khoa học hiện hành theo đề xuất V4.2

- Giữ snapshot dữ liệu hiện có; không tự crawl lại. Xác minh dữ liệu và thứ tự app IDs trước khi dùng kết quả.
- KMeans dùng **K=49**, outer seeds **42–51**; không điều chỉnh K/seed theo kết quả thuận lợi.
- **AMI** là chỉ số chính cho clustering/taxonomy; **ARI** là bổ sung.
- Retrieval EN→VI và VI→EN: **Recall@1, Recall@10, MRR** với tập ứng viên được xác định nhất quán.
- Linear probe: frozen embeddings + Logistic Regression; **Macro-F1** là chỉ số chính; chia train/test tránh leakage giữa các view của cùng app và các nhóm near-duplicate liên quan.
- Phân tích fixed-label subset/app composition chỉ mang ý nghĩa **dispersion mô tả**; không tuyên bố confidence interval quần thể hoặc p-value xác nhận.
- Phân tích provenance giữ các nhãn **translation-flagged**, **not-flagged**, **unknown**; không đồng nhất not-flagged với mô tả do người Việt viết gốc.
- Loại **ViDeBERTa khỏi bản revision theo quyết định tác giả được ghi trong V4.2**; giữ artifact lịch sử, không tuyên bố đã chứng minh mô hình lỗi hoặc loại vì điểm thấp.
- Loại phần bootstrap lịch sử WR/UNIQUE/D_MULT khỏi bài sửa; không xóa bằng chứng lịch sử.
- E5 phải chạy với một cấu hình/runtime có khai báo và kiểm chứng tính nhất quán; bổ sung baseline **LaBSE**.

**Lưu ý phê duyệt:** Hai bản V4.2 đang ở trạng thái `DRAFT_FOR_AUTHOR_APPROVAL` và tự ghi rằng chưa thay thế protocol freeze V4.1. Trước khi thực hiện một thay đổi trái với freeze cũ, tác giả phải xác nhận nội dung sửa đổi tương ứng. Kế hoạch này không tự cấp phép thay đổi protocol.

**Amendment phạm vi do tác giả phê duyệt ngày 24/10/2026:** Theo yêu cầu trực tiếp “bỏ Chặng 04 Back-translation Phân tích đối chứng R02”, R02 không còn bắt buộc cho revision này. Quyết định này thay thế riêng yêu cầu R02 bắt buộc trong `docs/decisions/PROTOCOL_FREEZE_V4_1_RECORD.md` và các phụ thuộc R02 trong tài liệu cũ; không phê duyệt thay đổi khoa học khác. Giữ tài liệu/artifact lịch sử để truy vết. Không tự bổ sung hệ dịch thứ hai hoặc đối chứng thay thế. Giữ số chặng hiện có để không đổi đường dẫn giao tiếp.

## 3. Nguyên tắc vận hành để tránh vòng lặp

1. **Mỗi chặng có một mục tiêu và đầu ra quan sát được.** Không tự mở thêm chặng phụ, project con, gate mới hoặc hệ thống quản trị mới.
2. **Công việc cần thiết mới được ưu tiên:** một phát hiện chỉ chặn tiến độ nếu có bằng chứng cho thấy nó làm sai kết quả, khiến kết quả không thể kiểm chứng, hoặc ngăn thực thi chặng đang làm. Cải thiện tùy chọn để lại sau.
3. **Tối đa hai lượt trao đổi/phản biện cho mỗi vấn đề của một chặng.** Nếu chưa thống nhất, ghi rõ điểm bất đồng. Với hai lựa chọn hợp lệ, dùng thử nghiệm nhỏ, đã định tiêu chí trước, để phân xử khi phù hợp. Nếu là lỗi tính đúng đắn hoặc thay đổi protocol chưa được duyệt, không chạy production để “xem kết quả”; chuyển tác giả quyết định.
4. **Thực thi theo readiness riêng:** E5 đủ điều kiện thì chạy E5, không chờ LaBSE. Một nhánh lỗi không chặn các nhánh độc lập. Chặng 04/R02 đã loại khỏi phạm vi, không còn là phụ thuộc thực thi.
5. **Không đồng nhất code PASS với kết quả thực nghiệm:** code/test → smoke ở môi trường đích → full run → validation kết quả là những trạng thái khác nhau.
6. **Không tối ưu theo score:** không thay recipe, metric, seed, split, model hoặc bỏ kết quả bất lợi sau khi xem số liệu, trừ quyết định thiết kế được giải trình minh bạch.
7. **Bảo toàn và truy vết:** thay đổi mã nguồn có ý nghĩa phải gắn với commit; lần chạy chính phải chỉ ra code/config/data/checkpoint đã sử dụng. Không ghi đè hay xóa artifact lịch sử.
8. **Không tái tạo tài liệu:** chỉ giữ các file giao tiếp theo chặng và artifact khoa học cần thiết. Báo cáo ngắn dẫn đường đến mã nguồn/log/kết quả, không chép lại toàn bộ.

## 4. Các chặng nghiên cứu

### Chặng 01 — Xác lập điểm xuất phát và phạm vi revision

**Làm gì:** Đối chiếu `review.md`, bản paper, kế hoạch V4.2 và hiện trạng code/dữ liệu/kết quả; nhận diện những thay đổi đã được phê duyệt, những bằng chứng còn dùng được và những việc còn thiếu. Xác định đường dẫn manuscript LaTeX hiện hành nếu có.

**Mục tiêu:** Chốt danh sách công việc nghiên cứu cần thiết, thứ tự ưu tiên và các đầu vào có thật; không làm lại những gì còn hợp lệ.

**Đầu ra tối thiểu:** Danh sách nhận xét reviewer → hành động cần thiết → bằng chứng/file liên quan; danh sách chặng được phép bắt đầu. **Không chạy lại model hoặc refactor trong chặng này.**

### Chặng 02 — Chạy lại E5 nhất quán

**Làm gì:** Hoàn thiện đường chạy E5 với đúng checkpoint/revision, prefix, pooling, input conditions, runtime và dataset đã chốt; kiểm thử logic liên quan; smoke trên GPU đích; sau đó chạy full các E5 cells đã được duyệt và xác minh artifact.

**Mục tiêu:** Có embedding E5 mới, nhất quán trong contract đã sử dụng và truy vết được; xử lý vấn đề bất nhất E5 cũ mà không khẳng định nguyên nhân lịch sử khi chưa có chứng cứ.

**Đầu ra tối thiểu:** Code/config tại commit cụ thể, kết quả probe tái lập, embeddings/row IDs/metadata, execution log, validation và ghi nhận khác biệt với kết quả lịch sử.

### Chặng 03 — Bổ sung baseline LaBSE

**Làm gì:** Chọn và ghi nhận checkpoint/revision LaBSE; triển khai đúng pipeline inference của checkpoint; đối chiếu output với implementation tham chiếu; chạy các cohort/condition được duyệt.

**Mục tiêu:** Có baseline cross-lingual bổ sung hợp lệ để so sánh cùng các recipes hiện có.

**Đầu ra tối thiểu:** Adapter/CLI hoặc reuse code hiện có, kiểm tra pooling/cap, embeddings và kết quả validation. **Không bắt E5 chờ LaBSE.**

### Chặng 04 — Đối chứng back-translation R02 — CANCELLED

**Quyết định:** Tác giả yêu cầu bỏ chặng ngày 24/10/2026. Không triển khai model dịch, chunking, pilot, full translation, EN_BT hoặc embeddings/metrics dành riêng cho R02. Đây là loại khỏi phạm vi, không phải hoàn thành thực nghiệm.

**Hệ quả:** Không có đối chứng EN–EN_BT để đặt S_XL cạnh S_BT; không tính D. Các chặng sau tiếp tục với RQ1/RQ2 và ghi rõ giới hạn này. Không xóa bằng chứng lịch sử, không tạo thư mục chặng chỉ để đánh dấu hủy.

### Chặng 05 — Đánh giá thực nghiệm

**Làm gì:** Từ embeddings hợp lệ, chạy clustering KMeans, AMI/ARI và các thành phần C/W/S cho EN–VI; retrieval hai chiều; frozen linear probe với split chống leakage; so sánh theo input conditions và provenance strata. Không tính S_BT/D hoặc chờ EN_BT. S của EN–VI là tương phản mô tả so với độ ổn định theo seed, không phải ước lượng hiệu ứng riêng của ngôn ngữ/dịch máy.

**Mục tiêu:** Trả lời RQ1 và RQ2 bằng nhiều góc đánh giá, không chỉ một con số AMI.

**Đầu ra tối thiểu:** Các bảng metric và artifact liên quan đến đúng cohort, view, recipe, seed và phiên bản embedding; kết quả tốt/xấu đều được giữ nếu chạy đúng.

### Chặng 06 — Độ bất định, độ ổn định và giới hạn

**Làm gì:** Phân tích sự khác biệt và mức biến thiên theo seeds, input conditions, provenance, app-composition dispersion; kiểm tra cách diễn giải xếp hạng và đối chứng.

**Mục tiêu:** Xác định phát hiện nào được bằng chứng hỗ trợ, phát hiện nào chỉ mang tính mô tả hoặc chưa chắc chắn; ngăn kết luận superiority, equivalence hay causality không có kiểm định phù hợp.

**Đầu ra tối thiểu:** Tổng hợp contrast, độ biến thiên, giới hạn và cách diễn giải đã kiểm tra; không tự thêm kiểm định mới chỉ để đạt ý nghĩa thống kê.

**Điều chỉnh sau khi bỏ R02:** Bỏ contrast và dispersion phụ thuộc EN_BT/S_BT/D; giữ phân tích seeds, input conditions, provenance và app-composition của RQ1/RQ2. Ghi rõ không thể tách thay đổi ngôn ngữ khỏi thay đổi văn bản/quy trình dịch bằng thiết kế hiện tại.

### Chặng 07 — Tổng hợp phát hiện và hình minh họa

**Làm gì:** Tạo bảng/hình từ kết quả đã xác minh; đối chiếu findings với RQ và reviewer; chọn ví dụ app/genre theo quy tắc minh bạch; cập nhật nguồn tham khảo khoa học đã xác minh.

**Mục tiêu:** Hình thành 2–3 phát hiện chính rõ ràng, truy ngược được tới dữ liệu và phương pháp, có giá trị đối với đánh giá app catalogue.

**Đầu ra tối thiểu:** Bảng kết quả tổng hợp, hình cần thiết, claim-to-evidence mapping và danh mục tài liệu tham khảo có nguồn kiểm chứng.

**Điều chỉnh sau khi bỏ R02:** Không lập bảng/hình hoặc claim EN–EN_BT/S_BT/D; tổng hợp findings từ RQ1/RQ2, không hứa xác định mức cấu trúc bị mất riêng do dịch máy.

### Chặng 08 — Hoàn thiện bài báo

**Làm gì:** Cập nhật bản LaTeX theo kết quả đã xác minh; sửa tiêu đề, abstract, introduction, related work, methods, results, discussion, limitations, conclusion và bibliography ở đúng mức cần thiết; loại ViDeBERTa/bootstrap lịch sử khỏi nội dung revision theo quyết định đã chốt; build và kiểm tra PDF cuối. Theo điều chỉnh của tác giả ngày26/10/2026, không soạn response letter hoặc bảng đối chiếu reviewer.

**Mục tiêu:** Có bài báo mạch lạc, không vượt quá bằng chứng, xử lý đầy đủ các nhận xét reviewer và đáp ứng quy định venue khi đã chọn.

**Đầu ra tối thiểu:** Source LaTeX `.tex`, bibliography `.bib`, các hình/bảng được sử dụng và PDF revision đã kiểm tra. Báo cáo thực hiện trong `research_stages/stage_08/codex.md`. Không có response letter hoặc bảng đối chiếu reviewer trong đầu ra. Không tự sửa đổi phần limitations đã khóa nếu chưa được tác giả duyệt.

**Hướng dẫn viết ưu tiên của tác giả:** Bộ20 gợi ý mang tên “Mandatory Rules” được lưu nguyên văn tại `research_stages/stage_08/codex.md` §2. Theo làm rõ mới nhất của tác giả, đây là lời khuyên, không phải bộ quy tắc cứng; có thể làm khác khi phù hợp với nội dung hoặc yêu cầu trình bày, nhưng cần giải thích ngắn lý do. Ưu tiên dẫn bằng phát hiện, đoạn văn có mục đích, thuật ngữ chuẩn, câu trực tiếp, số liệu thường3 chữ số thập phân và Discussion có diễn giải/đối chiếu nghiên cứu trước. Tránh kể lại bảng, suy diễn rankings, filler và disclaimers lặp lại; hình chính phục vụRQ/lập luận, chi tiết execution audit chuyển phụ lục/supplement khi cần. Mạch bài gợi ý: Problem → Research gap → Research question → Method → Evidence → Finding → Implication → Limitation. Các yêu cầu độc lập về tính trung thực khoa học và protocol đã khóa vẫn giữ nguyên: không bịa bằng chứng, thay phương pháp hoặc suy luận vượt phạm vi đã xác minh.

**Điều chỉnh sau khi bỏ R02:** Sửa các lời hứa về đối chứng BT và giữ kết luận trong phạm vi bằng chứng RQ1/RQ2 thực có. Theo chỉ đạo tác giả, không yêu cầu giải thích việc “thu hẹp phạm vi” hoặc quyết định bỏ R02 trong phản hồi reviewer. Không tuyên bố đã thực hiện đối chứng BT hoặc đã giải quyết góp ý bằng thực nghiệm BT khi chưa có. Giới hạn diễn giải EN–VI vẫn được trình bày đúng trong manuscript. Quyết định này cho phép cập nhật nội dung phạm vi/giới hạn liên quan R02, không mở quyền sửa các limitations đã khóa khác.

## 5. Thứ tự và song song

- **Bắt đầu:** Chặng 01 (ngắn, chỉ đủ để xác định đầu vào, quyền thực hiện và phụ thuộc).
- **Ưu tiên tạo kết quả:** Chặng 02 E5 và Chặng 03 LaBSE thực thi theo readiness riêng. Chặng 04 đã hủy; chặng nghiên cứu tiếp theo để tác giả giao là Chặng 05, không có phụ thuộc R02.
- **Trong lúc GPU chạy:** Có thể phát triển mã đánh giá của Chặng 05, chuẩn bị tài liệu tham khảo, cấu trúc hình/bài của Chặng 07–08, nhưng **không dùng kết quả lịch sử để viết kết luận mới**.
- **Khi từng GPU artifact về:** Validate và thực hiện đánh giá tương ứng ngay, không đợi tất cả model hoàn tất.
- **Chặng 06–08:** Chỉ kết luận và sửa các claim phụ thuộc khi dữ liệu/kết quả tương ứng đã hợp lệ; phần biên tập độc lập có thể chuẩn bị sớm.

## 6. Quy tắc trao đổi và tổ chức file tối thiểu

Giữ code, data, kết quả và manuscript tại các vị trí đang có; không sao chép cả repository theo chặng. Chỉ thêm một thư mục giao tiếp:

```text
research_stages/
  stage_01/
    codex.md
    claude.md
  stage_02/
    codex.md
    claude.md
  ... (chỉ tạo khi bắt đầu chặng)
```

- `codex.md`: mục tiêu chặng, thay đổi thực tế, file nguồn liên quan, commit, lệnh chạy, trạng thái và kết quả/đường dẫn bằng chứng.
- `claude.md`: đánh giá khoa học, bằng chứng và citation kiểm chứng được, rủi ro quan trọng, đề xuất cần thiết và kết luận.
- Mỗi bên chỉ cập nhật file của mình; dùng các mục **Lượt 1 / Lượt 2** nếu có phản hồi tiếp theo; không ghi đè lịch sử quan trọng.
- Các nguồn mà Agent tìm được chỉ được đưa vào bibliography sau khi xác minh paper thực sự hỗ trợ claim; không tự thêm citation không kiểm chứng.
- File giao tiếp **không** thay thế Git history, raw logs, config hoặc kết quả thí nghiệm.

## 7. Khi nào một chặng kết thúc?

Một chặng được xem là hoàn thành khi **đầu ra khoa học nêu trong chặng đã có và có thể kiểm chứng**; các lỗi thực sự ảnh hưởng độ hợp lệ đã được xử lý; những đề xuất không thiết yếu được để lại chứ không mở vòng sửa mới.

Các trạng thái nên dùng: **DONE**, **BLOCKED** (nêu đúng nguyên nhân), **WAITING_INPUT** (phụ thuộc chưa có). Nếu chỉ có code nhưng chưa chạy, không tuyên bố có kết quả thực nghiệm.

Sau tối đa hai lượt phản biện chưa giải quyết: (a) thí nghiệm phân xử nhỏ nếu cả hai lựa chọn khoa học đều hợp lệ và đã định tiêu chí trước; hoặc (b) trình tác giả nếu có lỗi đúng/sai, thay đổi protocol hoặc nguy cơ làm vô hiệu kết quả. Không tự sửa vô hạn.

## 8. Những việc không được tự ý làm

- Xóa/ghi đè dữ liệu, mã nguồn hoặc artifact lịch sử; tự ý xóa/đổi nội dung paper làm mất bằng chứng.
- Crawl lại dữ liệu; đổi cohort, seeds, K, metrics, split, pooling, checkpoint hoặc phương pháp đã chốt khi chưa có quyết định của tác giả.
- Chạy full experiment với cấu hình chưa xác minh hoặc chuyển CPU/GPU, precision, revision một cách âm thầm.
- Tạo nhiều phiên bản master plan, folder backup hoặc gate/report mới cho cùng một chặng.
- Nhận định rằng test PASS đồng nghĩa với production run PASS hoặc bằng chứng cho kết luận khoa học.
- Chọn kết quả thuận lợi rồi loại kết quả bất lợi, hoặc biến bằng chứng mô tả thành suy luận nhân quả.

## 9. Các quyết định còn cần tác giả chốt theo V4.2

Chỉ quyết định khi đến đúng chặng liên quan:

- LaBSE: checkpoint và immutable revision cuối (trước full run).
- Có bổ sung MPNet hay giải thích phạm vi không bổ sung (trước khi khóa baseline cuối).
- Back-translation/R02: đã loại khỏi phạm vi theo quyết định tác giả ngày 24/10/2026; không còn quyết định model/revision/chunking/pilot đang chờ.
- Venue mục tiêu và giới hạn trang (trước định dạng bài nộp).
- Tiêu đề cuối; xử lý average-linkage trong bản nộp.
- Mọi amendment cần thiết đối với freeze V4.1 và văn bản limitations đã khóa.

## 10. Tiêu chí hoàn thành toàn dự án

Nghiên cứu hoàn thành khi:

1. Các thực nghiệm đã được tác giả duyệt có kết quả hoặc được báo cáo rõ là chưa thực hiện và được xử lý phạm vi công bố thích hợp.
2. Các chỉ số, hình và kết luận trong manuscript mới có bằng chứng truy vết; không có claim vượt quá dữ liệu.
3. Các vấn đề khoa học từ review được xử lý phù hợp trong manuscript theo phạm vi đã chốt; theo yêu cầu tác giả ngày26/10/2026, không yêu cầu response letter hoặc bảng đối chiếu reviewer riêng.
4. Manuscript và PDF cuối được kiểm tra về nội dung, trích dẫn và yêu cầu venue.

**Ưu tiên số một:** Hoàn thành từng kết quả nghiên cứu hợp lệ, không hoàn thiện bộ máy điều phối.

## 11. Bảng tiến độ từng chặng

| Chặng | Công việc chính | Trạng thái | Tiến độ | Kết quả hoặc ghi chú |
|---|---|---|---:|---|
| 01 | Xác lập điểm xuất phát và phạm vi revision | DONE | 100% | Khảo sát đã hoàn thành và Claude ACCEPT; xem research_stages/stage_01/ |
| 02 | Chạy lại E5 nhất quán | DONE | 100% | Đủ6 cells inference-only, validation/historical comparison local PASS, runtime/source/checkpoint đồng nhất; actual paired repeat evidence; xem results/e5_stage02/validation_full_20261025_001/validation_receipt.json |
| 03 | Bổ sung baseline LaBSE | DONE | 100% | Bốn full cells local và validation PASS; xem research_stages/stage_03/codex.md và results/labse_stage03/validation_local_001/validation_receipt.json |
| 04 | Đối chứng back-translation R02 | CANCELLED | — | Tác giả bỏ ngày 24/10/2026; không thực hiện, không tính là DONE |
| 05 | Đánh giá clustering, retrieval và linear probe | DONE | 100% | Đủ34 clustering cells/340 seed rows,26 probes,9 paired conditions/245826 retrieval queries,480 provenance rows; results/stage05/combined_20261025_001/execution_receipt.json: STAGE05_COMPLETE_VALIDATED |
| 06 | Phân tích độ bất định, độ ổn định và giới hạn | DONE | 100% | N05 B200/f0.80 without replacement, fixed labels/models; contrasts/rank/provenance/limits đã có; STAGE06_N05_VALIDATION_PASS, xem research_stages/stage_06/codex.md |
| 07 | Tổng hợp phát hiện, bảng và hình minh họa | DONE — Claude ACCEPT Lượt3 | 100% | Bộ8 hình đã chọn giữ nguyên; đủ12 bảng CSV,5 bảng LaTeX,3 findings và dàn ý theo originalRQ1/RQ2,8 captions,claim-to-evidence,20 ví dụ app/245 genre rows và13 nguồn đã kiểm tra. Bundle results/stage07/synthesis_20261026_003/; verification PASS. Claude ACCEPT phần tổng hợp ngày26/10/2026, không có CRITICAL issue; xem research_stages/stage_07/claude.md Lượt3 và codex.md §15. Chặng08 chưa mở. |
| 08 | Hoàn thiện bài báo và kiểm tra PDF | READY_FOR_CLAUDE; chờ lựa chọn EN-half | Đã có bản sửa | Bản hiện hành output/pdf/revision_vi_fix.pdf:20trang,8hình,19nguồn; sửa spacing/văn phong/cấu trúc, xóa sectionCấu hình và điều kiện tái lập, thêm Data/Code availability vàAI disclosure. Bảng34cells/26probes vàPDF18trang trước giữ nguyên. EN-half khácR02; chưa tìm đủartifact để tái xác minh, chưa đưa điểm số vào bảnfix, đã hỏi tác giả cách trình bày. Build/document checksPASS; xem research_stages/stage_08/codex.md §5. Chưa DONE khoa học. |

## 12. Điểm tiếp tục phiên sau — ghi nhận ngày 24/10/2026

Tác giả yêu cầu dừng phiên hiện tại và tiếp tục hôm sau. Chưa khởi chạy Chặng05 trong phiên này; không thay đổi hoặc dừng các tiến trình GPU của tác giả.

- **LaBSE đã sẵn sàng cho đánh giá:** bốn full cells tại `results/labse_stage03/labse_local_20261024_204813_<cell>/`; bằng chứng `results/labse_stage03/validation_local_001/validation_receipt.json` có trạng thái `FOUR_CELLS_LOCAL_VALIDATION_PASS`. Không chạy lại inference hoặc reference probe khi chưa có lý do mới.
- **E5 đang chờ hoàn tất theo thông báo tác giả:** ZIP đã giải nén tại `stage02_embeddings_kg_20261024_134531_840336/` chỉ chứng minh một cell TN1 VI char950 hoàn tất. Không có bằng chứng live cho session sau reconnect/reset; không xem ZIP này là full run hoàn thành. Khi có ZIP mới, kiểm tra đủ sáu cells/receipt cuối rồi validation local trước khi sử dụng.
- **Công việc tiếp theo là Chặng05, ưu tiên LaBSE:** đọc code đánh giá hiện có và báo cáo Chặng03; tận dụng pipeline để chạy KMeans K=49/seeds42–51, AMI/ARI cho bốn cells; retrieval TN2 EN↔VI với Recall@1/Recall@10/MRR; vector correspondence và partition agreement C/W/S; chuẩn bị split chống leakage giữa các view cùng app/near-duplicate rồi chạy frozen Logistic Regression với Macro-F1. Giữ nguyên các cấu hình khoa học đã chốt, không tự thiết kế lại split/recipe nếu còn quyết định cần xác nhận.
- **Thực thi độc lập:** LaBSE không chờ E5. Khi E5 có artifacts hợp lệ, bổ sung đánh giá tương ứng vào cùng hệ thống kết quả. Chưa kết luận xếp hạng giữa các model khi kết quả còn thiếu. Ghi công việc Chặng05 vào `research_stages/stage_05/codex.md` khi thực sự bắt đầu; không tạo báo cáo hoặc framework mới trước đó.
- **Phạm vi đã chốt:** Chặng04/R02 đã hủy; không tạo EN_BT/S_BT/D và không tự thay bằng đối chứng dịch khác. Chặng08 không yêu cầu giải thích quyết định “thu hẹp phạm vi” trong phản hồi reviewer; mọi claim vẫn phải khớp bằng chứng thực có.

**Tiếp tục ngày 25/10/2026:** Đã mở Chặng05 theo yêu cầu tác giả. Đánh giá bốn cells LaBSE bằng pipeline hiện có trong namespace `results/stage05/labse_20261025_001/`; trạng thái thực thi và kết quả cuối xem `research_stages/stage_05/codex.md`. Chưa có ZIP E5 mới tại thời điểm bắt đầu phiên. Đã chuẩn bị folds và class-support cho linear probe tại `results/stage05/probe_split_proposal_20261025_001/`, xấp xỉ60/20/20, group-disjoint theo developer/duplicate components; đây là phương án chờ tác giả chốt, chưa phải kết quả probe. Không thay thế kế hoạch bằng tài liệu khác và không mở Chặng06–08.

**Điểm tiếp tục sau lượt thực thi25/10:** Job LaBSE đã kết thúc, có `LABSE_CLUSTERING_RETRIEVAL_CORRESPONDENCE_COMPLETE` và validation độc lập `results/stage05/validation_labse_20261025_001/validation_receipt.json`. Không khởi chạy lại cùng run ID hoặc coi job còn chạy. Tiếp theo đọc phản biện Claude của Chặng05 nếu có, nhận quyết định tác giả về split/tuning trước real probe, và kiểm tra E5 mới khi được cung cấp. Chặng05 vẫn IN_PROGRESS, không suy xếp hạng liên mô hình khi dữ liệu chưa đủ.

**E5 TN2 đã được giao sau đó ngày25/10:** Bốn cells trong `results/e5_stage02/kg_20261025_022156_432511_tn2_022259_065085/` đã kiểm tra, receipt `results/e5_stage02/validation_tn2_20261025_001/validation_receipt.json` có trạng thái `E5_TN2_LOCAL_ARTIFACT_VALIDATION_PASS`. Có thể tiếp tục Chặng05 cho TN2 E5 ngay, không đợi TN1; chưa chạy đánh giá E5 trong lượt kiểm tra file này. Giữ riêng native/T512 dù outputs trùng bitwise; không tự reuse/gộp cell. TN1 E5 mới chưa đủ để đóng Chặng02. Các ghi nhận “E5 chưa có” ở trên là trạng thái lịch sử trước khi nhận gói TN2 này.

**Tác giả chốt đi tiếp TN2 ngày25/10:** Đã triển khai và khởi chạy `pipeline.evaluate_stage05 --model e5_tn2`, run `e5_tn2_20261025_001`, commit `17eb619`; code giữ bốn cells độc lập, K49/n_init10/seeds42–51, paired metrics/retrieval cho native vàT512. Đã đọc Claude Chặng05 Lượt1 ACCEPT; các đính chính/giới hạn được ghi trong `research_stages/stage_05/codex.md`. Việc chốt đi tiếp TN2 không được coi là phê duyệt rõ split/tuning probe. Không khởi chạy lại khi chưa kiểm tra execution receipt/log của run đang có.

**Điểm tiếp tục mới nhất — E5 TN2 đánh giá xong:** Run `e5_tn2_20261025_001` đã hoàn thành40 lượt KMeans và paired metrics/retrieval cho cả native/T512. Validation `results/stage05/validation_e5_tn2_20261025_001/validation_receipt.json` đạt `E5_TN2_STAGE05_ARTIFACT_VALIDATION_PASS`. Cùng folder có các CSV đối chiếu native E5–LaBSE trên cùng cohort, ghi rõ cap512/256 và dim1024/768. Không còn job Chặng05 đang chạy, không chạy lại các run IDs đã có. Tiếp theo làm rõ tác giả có chốt cả split/tuning probe hay không (đã gửi câu hỏi), tiếp thu Claude cho đầu ra mới và bổ sung TN1 khi đủ artifacts. Chưa mở Chặng06 hoặc tuyên bố Chặng05 DONE.

**Cập nhật theo ủy quyền kế tiếp ngày25/10:** Tác giả yêu cầu “nếu CLAUDE đồng ý thì bạn cứ thực hiện” và giao TN1 E5. Claude Chặng05 Lượt1 đã ACCEPT split/tuning nên **probe được phép chạy, không còn chờ xác nhận split**. TN1 và aggregate6 cells E5 đã validation PASS; Chặng02 DONE trong phạm vi inference-only đã duyệt, với giới hạn standalone smoke receipt được ghi rõ trong `research_stages/stage_02/codex.md`. Probe LaBSE và E5 mỗi model4 cells (TN1 VI char/native, TN2 EN/VI native), cùng approved folds/C-grid, đã chạy xong; `results/stage05/validation_probes_20261025_001/validation_receipt.json` đạt `EIGHT_LABSE_E5_PROBES_VALIDATION_PASS`. TN1 E5 KMeans/provenance run `e5_tn1_20261025_001` đang chạy. Không chạy lại probe/KMeans đã có; các ghi nhận “probe chưa phê duyệt/TN1 thiếu” ở trên là lịch sử. Không tự mở Chặng06.

**Điểm tiếp tục mới nhất sau conditional approval:** TN1 E5 KMeans/provenance đã hoàn thành20 lượt, validation `results/stage05/validation_e5_tn1_20261025_001/validation_receipt.json` PASS; cùng folder có bảng TN1 E5–LaBSE.8 probes cũng đã validation PASS, comparison tại `results/stage05/validation_probes_20261025_001/probe_comparison.csv`. Không còn job chạy hoặc split chờ tác giả. Không chạy lại các phần E5/LaBSE đã hoàn tất. Chặng05 vẫn IN_PROGRESS vì còn kiểm tra lineage/tổng hợp hoặc đánh giá các recipes khác trong kế hoạch: TF-IDF–LSA/PhoBERT-family/BGE-M3/Qwen3 ở cohort áp dụng; không dùng outputs cũ chưa xác minh để lấp các phần này. Đọc `research_stages/stage_05/codex.md` và phản biện Claude mới nếu có trước khi tiếp tục. Không tự mở Chặng06–08 hoặc thêm sensitivity/kiểm định ngoài protocol.

**Trạng thái cuối ngày25/10 sau yêu cầu đọc Claude và tiếp tục:** Đã đọc/tiếp thu Claude Lượt2 ACCEPT, không có blocker; đã hoàn thành các recipes TF-IDF/PhoBERT-family/BGE-M3/Qwen3.24 cells historical được xác minh vector/label/metadata hashes trước reuse và recompute metrics. Lexical shared-space reconstruction khớp6 matrices với max abs7.45e-9;4 lexical probes fit vocabulary/IDF/SVD trên train only.18 probes bổ sung cùng approved folds/C-grid đã thực thi, tổng26 probes. Kết quả cuối tại `results/stage05/combined_20261025_001/` gồm taxonomy/retrieval/C-W-S/provenance/probe tables và artifact index theo source versions; receipt **STAGE05_COMPLETE_VALIDATED**, đủ34 clustering cells,340 seed rows,26 probes,9 paired conditions,1710 partition pairs,245826 retrieval queries,480 provenance rows.25/25 tests PASS và actual output validation PASS. **Chặng05 DONE**, không còn job chạy hoặc recipes bắt buộc còn thiếu trong scope Chặng05 hiện hành. Không chạy lại các run IDs đã có. Những ghi nhận IN_PROGRESS/missing recipes trước đoạn này là lịch sử. Chặng06–08 chưa mở; tác giả giao Chặng06 khi muốn tiếp tục. Giữ các giới hạn provenance/precision/lexical fit scope/input coverage; chưa có population CI, confirmatory significance hoặc causal claims.

**Sau khi tác giả giao Chặng06 ngày25/10:** Đã chạy N05-LOCK-001:200 draws/cohort, uniform80% apps không hoàn lại, PCG64 root20261023, fixed embeddings/partitions/probe models. Shared app indices across models/conditions/views; TN1 sample19106/TN2 sample10925. Exact-cache AMI không đổi metric,6/6 tests PASS và475 sklearn oracle pairs có max error8.94e-14; full computation97.907s CPU, validation55.752s, không GPU/model/clustering refit. Outputs `results/stage06/n05_20261025_001/`; validation `results/stage06/validation_n05_20261025_001/validation_receipt.json`: **STAGE06_N05_VALIDATION_PASS**. Đã có contrasts, finite-catalogue app-composition ranges, ranking stability, provenance support và giới hạn diễn giải tại `research_stages/stage_06/codex.md`. **Chặng06 computation/validation DONE**, không còn job chạy; chưa có Claude review riêng cho output Stage06, không tuyên bố đã ACCEPT. Không chạy lại run ID hoặc dùng ranges làm population CI/p-values/causal/equivalence evidence. Retrieval giữ full candidate catalogue, probe conditional on fitted models/folds, không training/reclustering uncertainty. Chặng07–08 chưa mở, tác giả giao Chặng07 khi tiếp tục.

### Điểm tiếp tục mới nhất — Chặng 07 chờ tác giả duyệt hình

Theo yêu cầu tác giả “tôi sẽ duyệt hình sau”, hiện dừng ở bước bàn giao để xem; không tự chọn hình hoặc triển khai chỉnh sửa tiếp trong lúc chờ.

- **Trạng thái:** Chặng01/02/03/05/06 DONE trong phạm vi đã ghi; Chặng04/R02 CANCELLED; Chặng07 IN_PROGRESS, chờ duyệt hình; Chặng08 NOT_STARTED. Ghi nhận “Chặng07 chưa mở” trong các đoạn trước là lịch sử và được thay thế bởi trạng thái này.
- **Bản duyệt chính — cập nhật26/10:** `results/stage07/gallery_20261025_005/PAAC_VN_Stage07_Review.ipynb` hiện có82 cells, gồm1 setup +26 cell code Matplotlib tạo hình trực tiếp +55 Markdown. Mỗi hình có hướng dẫn tiếng Việt về cách đọc, số liệu, ý nghĩa RQ1/RQ2, giới hạn và ô ghi chú duyệt được bảo toàn. Đã chạy27/27 code cells,26 PNG render mới trùng SHA256 ảnh gốc; chạy26 cells theo thứ tự đảo sau setup cũng PASS. Mở xem output không cần Run All; muốn chỉnh hình thì chạy setup một lần, sửa/chạy cell hình bất kỳ. Bản chỉ-hiển-thị-ảnh trước đó giữ tại `PAAC_VN_Stage07_Review_display_only.ipynb` cùng thư mục. Code chuyển đổi `pipeline/notebook_stage07_editable.py`, commit cuối `a33dc2b`; xem codex.md §10.
- **Nguồn hình bảo toàn:** `results/stage07/gallery_20261025_004/` có26 PNG/PDF/SVG, `index.html` và `all_figures.pdf`26 trang. Giữ cả ý tưởng Codex/Claude, chưa quyết định hình bài chính/phụ lục. Không dùng các lần render `_001`–`_003` làm bản giao hiện hành.
- **Code/báo cáo:** `pipeline/figures_stage07.py` commit `abb35cb`; `pipeline/notebook_stage07.py` commit `bfbba95`. Chi tiết tại `research_stages/stage_07/codex.md` §6–8; không sửa `claude.md`. Chưa có Claude ACCEPT riêng cho ảnh/notebook đã dựng.
- **Không có tác vụ nghiên cứu đang chạy trong phần việc này.** Không chạy lại inference/KMeans/probe/N05 hoặc dựng lại toàn bộ hình để tiếp tục xem. Lần sau đọc mã hình/quyết định và ghi chú tác giả trước; chỉ chỉnh các hình được yêu cầu, bảo toàn các bản hiện có.
- **Chưa hoàn thành Chặng07:** còn lựa chọn/chỉnh hình, caption cuối và tổng hợp findings/claim-to-evidence/ví dụ app theo phạm vi kế hoạch. Bộ hình chưa có panel riêng về vector cosine correspondence hoặc hiệu năng theo provenance; không tuyên bố đã bao trọn RQ. Chỉ hoàn thiện phần còn lại khi tác giả tiếp tục giao việc; không tự chuyển Chặng08.

### Cập nhật mới nhất26/10 — Đã áp dụng lựa chọn hình của tác giả

Tác giả chọn các mã01a–05c và yêu cầu chỉnh theo ghi chú, xóa phần còn lại. Trạng thái chờ lựa chọn ban đầu ở trên được thay bằng lựa chọn này.

- **Notebook hoạt động:** `results/stage07/gallery_20261025_005/PAAC_VN_Stage07_Review.ipynb`,46 cells:1 setup +14 code tạo hình +31 Markdown;15/15 code cells đã chạy,14 outputs nhúng. Bộ hình gồm13 mã gốc01a–05c và02c_text_length_histograms tách riêng theo yêu cầu sửa panel(c).
- **Chỉnh sửa:**01b còn5 khối tổng quát, giữ màu;02a giữ genre/provenance, bỏ ECDF;02b đồng bộ hai panel;02c dùng histogram độ dài trên trục ký tự tuyến tính.01a chưa có chỉ dẫn chỉnh cụ thể ngoài chữ “Chỉnh sửa” nên giữ nguyên. Các mục khác không có hướng dẫn cụ thể giữ nguyên code/hình.
- **Đã xóa khỏi bản duyệt chính:** nhóm06a–06d và9 heatmaps07, gồm cells và danh mục liên quan. Không tự đưa các hình này vào phụ lục. Nguồn dữ liệu/kết quả và gallery lịch sử `_004` vẫn được bảo toàn.
- **Bản trước lựa chọn:** `PAAC_VN_Stage07_Review_before_selection.ipynb` cùng thư mục giữ82 cells và ghi chú gốc;13 ô ghi chú giữ lại trong notebook mới không bị sửa. Không dùng bản này thay notebook hoạt động khi tiếp tục duyệt.
- **Kiểm chứng/code:**14 outputs hợp lệ,10 hình không yêu cầu sửa có code/PNG trùng bản trước; histogram bao gồm toàn bộ dữ liệu, tổng mỗi view=100%; source/historical output hashes không đổi. Code `pipeline/notebook_stage07_revision.py`, commit `2cc68ee`; báo cáo `research_stages/stage_07/codex.md` §11. Không chạy lại model, clustering, probe hoặc N05; không sửa `claude.md`.
- **Điểm tiếp tục:** Tác giả xem lại các bản sửa01b/02a/02b và histogram02c; nếu muốn sửa01a, cung cấp yêu cầu cụ thể. Chặng07 IN_PROGRESS, còn hoàn thiện caption/findings/claim-to-evidence và đầu ra liên quan; chưa chốt main/phụ lục hoặc mở Chặng08. Đọc notebook đã chọn, không phục hồi các cell06/07 hoặc chạy lại bộ26 hình khi chưa được giao.

**Chỉnh hình tiếp theo:**05b đã tô cam điểm có Macro-F1 cao nhất trong mỗi panel;01a đã được rút xuống4 khối tổng quát với nhãn ngắn, không liệt kê thông số/thao tác. Hai code cells và outputs đã chạy lại trong notebook `_005`; ghi nhận01a còn thiếu hướng dẫn ở trên là lịch sử. Bộ duyệt vẫn14 hình, không thay kết quả hoặc chạy lại thực nghiệm; xem codex.md các ghi chú sau §11.

**01a mới nhất:** Theo yêu cầu “chi tiết hơn chút nữa”, hình hiện có5 khối chính kèm một dòng giải thích ngắn mỗi khối; thêm rõ TN1/TN2 (VI và paired EN–VI), vẫn không liệt kê thông số kỹ thuật. Đã chạy/lưu code và output01a trong notebook `_005`; ghi nhận4 khối trước đó là lịch sử.

**01a hiện hành — bố cục hai nhánh:** Theo tác giả, đã cập nhật sơ đồ dựa trên cấu trúc hình paper cũ nhưng theo protocol hiện tại: TN1 VI/8 recipes → RQ2; paired TN2 EN–VI/5 recipes → RQ1, có retrieval/probe và ngoại lệ LaBSE/E5. Dải N05 là200 subsets/80% không hoàn lại trên outputs cố định, không bootstrap/CI. Code, Markdown và output01a đã chạy/lưu tại notebook `_005`; bản tuyến tính trước đó không còn là01a hiện hành. Bộ duyệt vẫn14 hình, không thêm thực nghiệm hoặc mở Chặng08.

**Bước biên tập trước khi viết paper:** Theo yêu cầu tác giả, đã ghi đề xuất ý tưởng viết,3 findings có bằng chứng,dàn ý7 phần và mapping bảng/hình vào `research_stages/stage_07/codex.md` §12. Các lựa chọn main/appendix và hai hình bổ sung cho cosine/provenance chỉ là đề xuất, chưa được coi là tác giả phê duyệt; không phục hồi các hình06/07 đã loại. Chưa viết lại manuscript hoặc tạo hình/thí nghiệm mới; Chặng07 IN_PROGRESS, Chặng08 chưa mở.

### Lựa chọn hình đã chốt26/10 — trạng thái hiện hành

Đã áp dụng02a thay02b;03a tách riêngTN1 vàTN2; retrieval dùng04c;05b kèm heatmap(b)05a. Giữ01a và04b theo dàn ý đang trao đổi. Steering mới nhất: mọi scatter05b cùng màu xanh, không tôF1 cao nhất. Notebook `_005/PAAC_VN_Stage07_Review.ipynb` có8 cell hình+1setup/28cells tổng,outputs đã chạy. Bản14hình trước chốt giữ ở `PAAC_VN_Stage07_Review_before_final_selection.ipynb`; source/historical artifacts nguyên vẹn. Code commit `51180c4`, báo cáo `research_stages/stage_07/codex.md` §13.

Đây là chốt lựa chọn hình, chưa phải chốt toàn bộ ý tưởng viết/dàn ý/findings. Các mô tả chờ chọn ban đầu hoặc bộ14hình ở trên là lịch sử. Chặng07 còn hoàn thiện mạchRQ, bảng/caption/evidence và đầu ra còn lại; chưa mở Chặng08, không chạy thêm thực nghiệm.

### Điểm bàn giao mới nhất — hoàn thiện Chặng 07 ngày26/10/2026

Theo yêu cầu tác giả “thực hiện độc lập phần còn lại và báo cáo trong research_stages”, Codex đã hoàn thành đầu ra tổng hợp của Chặng07. Trạng thái chờ thực hiện ở đoạn trên là lịch sử trước lượt này. Báo cáo chính: `research_stages/stage_07/codex.md` §14; bundle sử dụng: `results/stage07/synthesis_20261026_003/`; code commit `f0b5ea2c10c04ce46a61a1ffa05974d0d3909723`.

- Đã có ba findings bám originalRQ1/RQ2, dàn ý và mapping claims tới bảng/hình/reviewer; vector correspondence và provenance performance có bảng riêng, không bỏ khỏi mạch bài.
- Đã xuất12 bảng số CSV,5 bảng LaTeX làm tròn3 chữ số,8 hình đã chọn ởPNG/PDF/SVG,8 captions,20 ví dụ app được chọn minh bạch và bảng đủ49genres×5native paired recipes.
- Đã kiểm tra13 nguồn sơ cấp cho app-store mining/categorization, cross-lingual clustering, Vietnamese/multilingual embeddings và task-specific evaluation; không tự thêm baseline.
- Đã đối chiếu340 clustering seed rows,26 probe scores và480 provenance rows: PASS; giữ khác biệt lexical fit scopes, probe single evaluation, N05 fixed-output interpretation và provenance restricted-partition estimand.
- Giữ nguyên notebook duyệt `_005`, lựa chọn8hình và màu05b. Không inference/refit/crawl hoặc thay protocol; không phục hồi các hình đã loại. Chặng04/R02 tiếp tục CANCELLED.

**Bước tiếp theo:** Claude đánh giá nội dung tổng hợp mới trong `codex.md` §14, đặc biệt độ phủRQ, diễn giải ví dụ và định vị tài liệu. Codex chưa ghi thay `claude.md` hoặc tuyên bố Claude đã ACCEPT lượt này. Chặng07 **DONE về đầu ra phía Codex, chờ đánh giá tổng hợp**; Chặng08 **NOT_STARTED**, chỉ mở khi tác giả giao. Chưa sửa manuscript, response letter hoặc limitations đang khóa.

### Đóng Chặng 07 — Claude ACCEPT Lượt3, ngày26/10/2026

Đã đọc đánh giá tại `research_stages/stage_07/claude.md` Lượt3: **ACCEPT — Stage07 DONE**, không có CRITICAL issue. Claude xác nhận độ phủRQ, diễn giải ví dụ và định vị13 nguồn. Trạng thái chờ đánh giá ở trên đã kết thúc; bundle `_003` được chấp nhận cho bước viết tiếp theo.

Ghi nhận một IMPROVEMENT cho Chặng08: tách dẫn nguồn BKAI model card để xác định checkpoint khỏi bài Vietnamese RAG `arXiv:2403.01616` dùng cho bối cảnh nghiên cứu; không dùng điểm số RAG làm bằng chứng genre clustering. Không cần sửa hoặc chạy lại kết quả Chặng07. Chặng08 vẫn **NOT_STARTED**, chỉ mở khi tác giả giao.

### Điểm bàn giao hiện hành — Chặng 08, revision tiếng Việt ngày26/10/2026

Tác giả đã giao thực hiện Chặng08 và yêu cầu viết bằng tiếng Việt. Các ghi nhận Chặng08 NOT_STARTED hoặc chưa có bản thảo ở những đoạn lịch sử trên không còn là trạng thái hiện hành.

- **Đã có bản revision:** `paper/revision_vi.tex`, `paper/limitations_vi.tex`, `paper/references.bib`, các bảng/hình và `output/pdf/revision_vi.pdf` (18trang). Gói source: `output/pdf/PAAC_VN_revision_vi_source.zip`; hướng dẫn ở `paper/README.md`.
- **Đã thực thi:** sinh bảng/macros từ bundle Chặng07, build bằngTectonic0.17.0/XeTeX/BibTeX, render và xem18trang, kiểm tra số liệu/source/citations. Có8hình đã duyệt và17nguồn; bảng khớp34clusteringcells/26probes,9pairedconditions. Không có overfull/underfull boxes,missing glyph hoặc undefined citations/references trong log cuối. Chi tiết `paper/verification.json`.
- **Phạm vi đầu ra:** không có response letter hoặc bảng đối chiếu reviewer theo yêu cầu tác giả. Bộ20 gợi ý viết tiếp tục là lời khuyên ưu tiên, được điều chỉnh kèm giải thích ngắn; bản thảo giữ originalRQ1/RQ2 và protocol.
- **Phê duyệt limitations riêng:** tác giả trả lời “Đồng ý dùng bản cập nhật” cho đề xuất tiếng Việt đã trình trong hội thoại. Cho phép thay phần Giới hạn cũ bằng nội dung hiện tại ở `paper/limitations_vi.tex`, ghi đúng N05/restricted partitions và giới hạn khoa học. Nguyên văn Section6 cũ giữ ở `paper/limitations_locked_original.txt`; không mở quyền thay protocol hoặc các quyết định khác.
- **Bảo toàn:** không chạy inference/refit/crawl, không sửa dữ liệu/kết quả lịch sử/nhãn, không đổi bộ8hình. Code sinh bảng commit `0f815925a7098915382c593bff7931037069c9d5`; manuscript nằm trong `paper/` đang được gitignore theo quy tắc có sẵn, bàn giao local/ZIP cùng hashes, không force-add.

**Trạng thái:** Chặng08 **READY_FOR_CLAUDE**, đầu ra Codex đã có nhưng chưa tuyên bố DONE khoa học hoặc sẵn sàng nộp một venue cụ thể. Claude đánh giá bản thảo mới theo `research_stages/stage_08/codex.md` §4; chưa có đánh giá Chặng08 được ghi thay. Venue/template, thông tin liên hệ tác giả và disclosure theo venue chưa được chốt; không bịa các thông tin đó.
