# PAAC-VN — KẾ HOẠCH TRIỂN KHAI NGHIÊN CỨU VÀ HOÀN THIỆN BÀI BÁO

**Phiên bản kế hoạch vận hành:** 1.0 — 24/10/2026  
**Tình trạng:** Đề xuất để tác giả phê duyệt trước triển khai  
**Cơ sở:** `Project_3_final.pdf`, các nhận xét trong `review.md` được nhắc đến trong V4.2.  
**Phạm vi tài liệu:** Xác định *làm gì, để đạt mục tiêu nào, bằng đầu ra nào*. Vai trò và prompt riêng của Codex/Claude sẽ được quy định sau, không nằm trong kế hoạch này.

## 1. Mục tiêu chung

Hoàn thiện nghiên cứu **PAAC-VN** và sửa bài báo từ bản thảo hiện có dựa trên phản biện, bằng các thực nghiệm có thể truy vết và kết luận đúng phạm vi bằng chứng.

- **RQ1:** Đánh giá mức tương ứng giữa các mô tả tiếng Anh–tiếng Việt của cùng ứng dụng: vector correspondence, retrieval, sự đồng thuận phân hoạch và mức phù hợp taxonomy theo ngôn ngữ.
- **RQ2:** Đánh giá các representation/input recipes đối với taxonomy của Google Play trên benchmark tiếng Việt, có xem xét input conditions và các nhóm provenance được ghi nhận.
- **R02:** Bổ sung đối chứng EN–EN_BT trên tập ứng dụng tương ứng để đặt so sánh EN–VI vào bối cảnh thay đổi văn bản; không diễn giải kết quả thành tác động nhân quả thuần túy của dịch máy.

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

## 3. Nguyên tắc vận hành để tránh vòng lặp

1. **Mỗi chặng có một mục tiêu và đầu ra quan sát được.** Không tự mở thêm chặng phụ, project con, gate mới hoặc hệ thống quản trị mới.
2. **Công việc cần thiết mới được ưu tiên:** một phát hiện chỉ chặn tiến độ nếu có bằng chứng cho thấy nó làm sai kết quả, khiến kết quả không thể kiểm chứng, hoặc ngăn thực thi chặng đang làm. Cải thiện tùy chọn để lại sau.
3. **Tối đa hai lượt trao đổi/phản biện cho mỗi vấn đề của một chặng.** Nếu chưa thống nhất, ghi rõ điểm bất đồng. Với hai lựa chọn hợp lệ, dùng thử nghiệm nhỏ, đã định tiêu chí trước, để phân xử khi phù hợp. Nếu là lỗi tính đúng đắn hoặc thay đổi protocol chưa được duyệt, không chạy production để “xem kết quả”; chuyển tác giả quyết định.
4. **Thực thi theo readiness riêng:** E5 đủ điều kiện thì chạy E5, không chờ LaBSE/back-translation. Một nhánh lỗi không chặn các nhánh độc lập.
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

### Chặng 04 — Đối chứng back-translation R02

**Làm gì:** Xác định mô hình dịch, revision, chính sách chunk và kiểm tra chất lượng; chạy pilot đánh giá độ đầy đủ và bảo toàn mapping; sau đó tạo EN_BT và embeddings tương ứng trên matched support đã duyệt.

**Mục tiêu:** Có đối chứng văn bản đủ chất lượng để so sánh EN–VI với EN–EN_BT và phân tích S mà không gán nghĩa nhân quả quá mức.

**Đầu ra tối thiểu:** Văn bản EN_BT có ID/provenance, kết quả kiểm tra pilot, embeddings tương ứng và ghi nhận giới hạn đối chứng. **Không full translation trước khi các quyết định dịch/pilot được phê duyệt.**

### Chặng 05 — Đánh giá thực nghiệm

**Làm gì:** Từ embeddings hợp lệ, chạy clustering KMeans, AMI/ARI, các thành phần C/W/S và D khi đủ R02; retrieval hai chiều; frozen linear probe với split chống leakage; so sánh theo input conditions và provenance strata.

**Mục tiêu:** Trả lời RQ1, RQ2 và R02 bằng nhiều góc đánh giá, không chỉ một con số AMI.

**Đầu ra tối thiểu:** Các bảng metric và artifact liên quan đến đúng cohort, view, recipe, seed và phiên bản embedding; kết quả tốt/xấu đều được giữ nếu chạy đúng.

### Chặng 06 — Độ bất định, độ ổn định và giới hạn

**Làm gì:** Phân tích sự khác biệt và mức biến thiên theo seeds, input conditions, provenance, app-composition dispersion; kiểm tra cách diễn giải xếp hạng và đối chứng.

**Mục tiêu:** Xác định phát hiện nào được bằng chứng hỗ trợ, phát hiện nào chỉ mang tính mô tả hoặc chưa chắc chắn; ngăn kết luận superiority, equivalence hay causality không có kiểm định phù hợp.

**Đầu ra tối thiểu:** Tổng hợp contrast, độ biến thiên, giới hạn và cách diễn giải đã kiểm tra; không tự thêm kiểm định mới chỉ để đạt ý nghĩa thống kê.

### Chặng 07 — Tổng hợp phát hiện và hình minh họa

**Làm gì:** Tạo bảng/hình từ kết quả đã xác minh; đối chiếu findings với RQ và reviewer; chọn ví dụ app/genre theo quy tắc minh bạch; cập nhật nguồn tham khảo khoa học đã xác minh.

**Mục tiêu:** Hình thành 2–3 phát hiện chính rõ ràng, truy ngược được tới dữ liệu và phương pháp, có giá trị đối với đánh giá app catalogue.

**Đầu ra tối thiểu:** Bảng kết quả tổng hợp, hình cần thiết, claim-to-evidence mapping và danh mục tài liệu tham khảo có nguồn kiểm chứng.

### Chặng 08 — Sửa bài báo và phản hồi reviewer

**Làm gì:** Cập nhật bản LaTeX theo kết quả đã xác minh; sửa tiêu đề, abstract, introduction, related work, methods, results, discussion, limitations, conclusion và bibliography ở đúng mức cần thiết; loại ViDeBERTa/bootstrap lịch sử khỏi nội dung revision theo quyết định đã chốt; chuẩn bị response letter; build và kiểm tra PDF cuối.

**Mục tiêu:** Có bài báo mạch lạc, không vượt quá bằng chứng, xử lý đầy đủ các nhận xét reviewer và đáp ứng quy định venue khi đã chọn.

**Đầu ra tối thiểu:** File `.tex` mới, PDF, response letter và bảng đối chiếu reviewer → vị trí sửa → bằng chứng. Không tự sửa đổi phần limitations đã khóa nếu chưa được tác giả duyệt.

## 5. Thứ tự và song song

- **Bắt đầu:** Chặng 01 (ngắn, chỉ đủ để xác định đầu vào, quyền thực hiện và phụ thuộc).
- **Ưu tiên tạo kết quả:** Chặng 02 E5 bắt đầu ngay khi điều kiện riêng đã đủ. Chặng 03 LaBSE không cần chờ Chặng 04 R02.
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
- Back-translation: model, revision, chunking, pilot/human quality rubric và chính sách loại trừ.
- Venue mục tiêu và giới hạn trang (trước định dạng bài nộp).
- Tiêu đề cuối; xử lý average-linkage trong bản nộp.
- Mọi amendment cần thiết đối với freeze V4.1 và văn bản limitations đã khóa.

## 10. Tiêu chí hoàn thành toàn dự án

Nghiên cứu hoàn thành khi:

1. Các thực nghiệm đã được tác giả duyệt có kết quả hoặc được báo cáo rõ là chưa thực hiện và được xử lý phạm vi công bố thích hợp.
2. Các chỉ số, hình và kết luận trong manuscript mới có bằng chứng truy vết; không có claim vượt quá dữ liệu.
3. Các nhận xét reviewer có câu trả lời cụ thể, kèm vị trí sửa/bằng chứng hoặc lý do từ chối hợp lý.
4. Manuscript và PDF cuối được kiểm tra về nội dung, trích dẫn và yêu cầu venue.

**Ưu tiên số một:** Hoàn thành từng kết quả nghiên cứu hợp lệ, không hoàn thiện bộ máy điều phối.

## 11. Bảng tiến độ từng chặng

| Chặng | Công việc chính | Trạng thái | Tiến độ | Kết quả hoặc ghi chú |
|---|---|---|---:|---|
| 01 | Xác lập điểm xuất phát và phạm vi revision | NOT_STARTED | 0% | Chưa cập nhật |
| 02 | Chạy lại E5 nhất quán | NOT_STARTED | 0% | Chưa cập nhật |
| 03 | Bổ sung baseline LaBSE | NOT_STARTED | 0% | Chưa cập nhật |
| 04 | Thực hiện đối chứng back-translation R02 | NOT_STARTED | 0% | Chưa cập nhật |
| 05 | Đánh giá clustering, retrieval và linear probe | NOT_STARTED | 0% | Chưa cập nhật |
| 06 | Phân tích độ bất định, độ ổn định và giới hạn | NOT_STARTED | 0% | Chưa cập nhật |
| 07 | Tổng hợp phát hiện, bảng và hình minh họa | NOT_STARTED | 0% | Chưa cập nhật |
| 08 | Sửa bài báo, phản hồi reviewer và kiểm tra bản nộp | NOT_STARTED | 0% | Chưa cập nhật |
