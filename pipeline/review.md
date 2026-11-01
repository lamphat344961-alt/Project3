Nhận xét phản biện
Bài báo: From Pairwise Alignment to Cluster Stability: A Paired Cross-Lingual Study of English and Vietnamese App Descriptions
Đề xuất: Sửa lớn (Major revision)
Bài có thiết kế chặt chẽ và trung thực về phương pháp, nhưng đóng góp chưa được nêu rõ. Văn phong hiện nay giống một báo cáo kiểm định nội bộ hơn là một bài báo khoa học.
1. Tóm tắt nội dung
Bài so sánh mô tả ứng dụng Google Play bằng tiếng Anh gốc và bản dịch máy tiếng Việt trên ba cấp độ:
tương ứng vector (Δcos),
mức đồng thuận phân cụm (AMI, chỉ số S),
mức khớp với taxonomy thể loại của store.

Dữ liệu gồm 13.657 cặp app và một benchmark tiếng Việt 23.883 app với 8 recipe biểu diễn. Kết quả chính:
Cặp cùng app gần nhau hơn mức nền trong mọi điều kiện, nhưng phân cụm xuyên ngôn ngữ kém ổn định hơn phân cụm chỉ đổi seed.
Thí nghiệm đối chứng chia đôi văn bản tiếng Anh cho mức suy giảm còn lớn hơn.
E5, BGE-M3 và Qwen3 luôn nằm trong top 3.

2. Điểm mạnh
Tái lập tốt: checkpoint được pin theo revision, mô tả đầy đủ pooling, prefix và token cap, có audit truncation (Bảng A1, A2, B2). Hiếm bài trong nước làm kỹ đến mức này.
Thiết kế cặp trên cùng app ID loại bỏ được khác biệt về thành phần dữ liệu giữa hai ngôn ngữ.
Thí nghiệm đối chứng English-halves là ý hay. Nó tự phản biện giả thuyết "suy giảm là do dịch".
Phân tầng theo provenance (flagged / not-flagged) cùng cách dùng thuật ngữ thận trọng ("not-flagged" thay vì "human-written").
Giới hạn được nêu trung thực, không thổi phồng kết quả.

3. Vấn đề chính
3.1. Đóng góp không rõ – bài "tự rào" quá mức.
Gần như mỗi câu kết quả đều kèm một câu phủ định ("does not establish…", "without identifying…"). Đọc xong, reviewer sẽ hỏi: "Vậy người làm hệ thống app store song ngữ nên làm gì khác?" Phần 5.4 chỉ kết luận rằng nên báo cáo ba chỉ số riêng. Đây là đóng góp quá mỏng.
Cần viết lại Introduction với 2–3 phát hiện cụ thể có giá trị thực tiễn, ví dụ:
Không nên dùng độ tương đồng cặp để suy ra chất lượng phân cụm.
Với tiếng Việt, các mô hình đa ngữ lớn vượt các mô hình chuyên tiếng Việt (PhoBERT, SimCSE, BKAI) khoảng 0,06–0,15 AMI.

Phát hiện thứ hai thực ra là kết quả "bán được" nhất của bài, nhưng hiện đang bị chìm.
3.2. Mốc tham chiếu của S gần như chắc chắn cho giá trị âm.
So sánh "đổi văn bản + ngôn ngữ" với "chỉ đổi seed" thì S < 0 là điều dễ đoán. Đối chứng chia đôi cho thấy chỉ cần đổi nội dung là S đã giảm mạnh, nên S chưa tách được hiệu ứng ngôn ngữ. Cần một mốc so sánh hợp lý hơn, ví dụ:
EN gốc vs EN back-translated (EN→VI→EN): cùng ngôn ngữ nhưng đi qua dịch máy, giữ gần nguyên nội dung và độ dài.
Hoặc VI (Google) vs VI từ một hệ dịch khác.

Có thêm một trong hai mốc này, phần RQ1 sẽ trả lời được câu hỏi về dịch thay vì chỉ mô tả.
3.3. Δcos không so sánh được giữa các mô hình.
Chính tác giả thừa nhận thang đo của Δcos phụ thuộc recipe. Vậy nên báo cáo thêm chỉ số chuẩn cho truy hồi cặp: Recall@1, Recall@10, MRR (EN→VI và VI→EN). Các chỉ số này dễ hiểu, so sánh được giữa mô hình và là chuẩn trong tài liệu (LaBSE, MTEB bitext mining).
3.4. Phần bootstrap "lịch sử" nên bỏ và làm lại.
Mục 3.5, 4.6.1 và Phụ lục D (3 bảng) dành để chẩn đoán một quy trình bootstrap bị lỗi: lấy mẫu có hoàn lại với nhãn lặp, làm lệch AMI. Nhưng bài lại không đưa ra khoảng tin cậy đúng. Reviewer sẽ hỏi: biết sai rồi sao không chạy lại?
Đề xuất:
Dùng subsampling không hoàn lại (m-out-of-n, ví dụ 80% app) hoặc bootstrap theo developer.
Báo cáo CI thật.
Xóa toàn bộ khái niệm "historical", WR/UNIQUE, D_MULT.
Riêng việc này có thể rút ngắn bài khoảng 3 trang và loại bỏ phần khó hiểu nhất.
3.5. ViDeBERTa AMI ≈ 0,02–0,03 gần như chắc chắn là lỗi pipeline, không phải kết quả. Mức này gần ngẫu nhiên với 49 cụm. Cần kiểm tra:
ViDeBERTa có cần PyVi segmentation không,
fast tokenizer có khớp không,
vector có bị suy biến không (đo anisotropy, kiểm tra cosine trung bình).

Sau đó sửa lỗi hoặc loại mô hình khỏi bài. Để nguyên con số này, reviewer sẽ nghi ngờ toàn bộ pipeline.
3.6. E5 cùng cấu hình 512 nhưng hai lần chạy cho vector khác nhau (mục 3.3 và 5.3, "remains unresolved"). Đây là vấn đề tái lập. Cần cố định thiết bị, dtype và deterministic flags, chạy lại, hoặc chỉ dùng một bộ vector.
3.7. Khác biệt giữa top 3 quá nhỏ để xếp hạng.
BGE, Qwen và E5 chênh nhau 0,001–0,01 AMI, cùng cỡ với SD theo seed (0,003–0,007). Nên nói rằng ba mô hình tương đương và bỏ việc phân tích thứ tự nội bộ, vốn chiếm khá nhiều chữ ở mục 4.4 và 4.5.
3.8. Taxonomy alignment chỉ dùng KMeans với k = 49.
AMI quanh 0,5 khó diễn giải. Nên bổ sung linear probe (logistic regression trên embedding, macro-F1) làm thước đo bổ sung cho mức khớp thể loại. Cách này phổ biến và dễ thuyết phục hơn.
4. Vấn đề trình bày
Văn phong quá đặc và quá nhiều thuật ngữ tự đặt ("initialization-relative partition deficit", "bootstrap multiplicity-sensitivity diagnostic", "support-matched diagnostic"). Bài đọc rõ là do AI soạn với giọng phòng thủ. Cần viết lại bằng ngôn ngữ tự nhiên, câu ngắn, và mỗi đoạn có một ý chính.

Số thập phân quá dài (6–9 chữ số). Nên làm tròn còn 3 chữ số.

Tiêu đề hứa nhiều hơn nội dung. "From Pairwise Alignment to Cluster Stability" gợi ý có cơ chế liên kết hai cấp độ, nhưng bài lại phủ nhận điều đó. Gợi ý tiêu đề: "Does Machine Translation Preserve App Clustering? A Paired English–Vietnamese Embedding Study."

Không có hình nào. Nên thêm:
biểu đồ cột AMI của 8 recipe,
một hình minh họa (heatmap) các genre bị tách hoặc gộp khi đổi ngôn ngữ,
Sửa đánh số tài liệu tham khảo, thông tin tác giả, phần AI disclosure; thêm tóm tắt tiếng Việt nếu tạp chí yêu cầu.

Gợi ý nơi nộp: Tạp chí Khoa học HUFLIT, Tạp chí Khoa học & Công nghệ Thông tin và Truyền thông (JSTIC – PTIT), VNU Journal of Computer Science and Communication Engineering, Tạp chí Phát triển KH&CN (ĐHQG-HCM), Tạp chí KH&CN – ĐH Đà Nẵng. Thầy nên đối chiếu danh mục và điểm HĐGSNN ngành CNTT năm hiện hành, vì danh mục thay đổi theo năm.
Lựa chọn cao hơn: Nếu làm thêm mốc back-translation (mục 3.2) và linear probe (mục 3.8), bài đủ sức gửi hội nghị Scopus như KSE, SoICT hoặc RIVF, hoặc tạp chí Q3/Q4. Đóng góp lúc đó sẽ không chỉ là mô tả mà trả lời được câu hỏi "dịch máy làm mất bao nhiêu cấu trúc nhóm so với chỉ diễn đạt lại". Với lượng dữ liệu đã có, phần việc thêm này không lớn: chủ yếu là dịch ngược khoảng 13 nghìn mô tả và chạy lại pipeline.
vài ví dụ app cụ thể bị đổi cụm.


Đánh số tài liệu tham khảo bị thủng: thiếu [10], [23], [24], [26]–[29]. Đây là dấu hiệu bản thảo còn sót.

Related work mỏng: khoảng 20 tài liệu. Thiếu các phần sau:
các nghiên cứu khai phá app store gần đây,
phân cụm xuyên ngôn ngữ,
embedding tiếng Việt mới.

Bài trích LaBSE [4] nhưng không đánh giá. Reviewer sẽ hỏi tại sao không có LaBSE và paraphrase-multilingual-mpnet, hai baseline xuyên ngữ kinh điển.

Mục 4.6.2 (average-linkage) cho kết quả suy biến và không đóng góp gì. Nên bỏ hoặc chuyển xuống phụ lục một dòng.

Cần làm rõ "Google-translated": đây là bản dịch tự động do Google Play hiển thị, hay nhóm tự dịch qua API? Cờ dịch lấy từ trường nào của google-play-scraper?

Các mục hành chính chưa hoàn thiện:
Tác giả ghi "1st/2nd/3rd/4th" (mẫu IEEE còn sót), chưa có email và tác giả liên hệ.
Phần AI disclosure còn câu nháp "require author confirmation".
Cần nêu vấn đề điều khoản sử dụng khi scrape Google Play.



5. Có gửi tạp chí trong nước được không?
Được, và khả năng được nhận khá cao nếu sửa. Độ chặt chẽ phương pháp của bài đã vượt mặt bằng chung nhiều tạp chí trong nước. Rủi ro bị từ chối không nằm ở kỹ thuật mà ở hai điểm:
Phản biện trong nước đọc không hiểu đóng góp, do văn phong quá đặc và phòng thủ.
Bài dài với nhiều phụ lục, vượt giới hạn trang (thường 10–15 trang).

Việc tối thiểu cần làm trước khi nộp:
Viết lại Abstract, Introduction và Conclusion với 3 phát hiện rõ ràng, giảm khoảng 70% câu rào đón.
Sửa hoặc loại ViDeBERTa; giải quyết vấn đề E5 không nhất quán.
Bỏ phần bootstrap lịch sử; thay bằng subsampling CI hoặc chỉ giữ seed CI.
Thêm Recall@k cho RQ1, thêm 2–3 hình, làm tròn số liệu.

HƯỚNG DẪN CHẠY LẠI PIPELINE
Bài báo: From Pairwise Alignment to Cluster Stability: A Paired Cross-Lingual Study of English and Vietnamese App Descriptions
Tài liệu dành cho nhóm tác giả – theo các góp ý phản biện
1. Mục tiêu
Không cần chạy lại toàn bộ pipeline. Phần lớn các góp ý phản biện có thể xử lý trên vector và phân hoạch đã lưu. Chỉ có hai việc cần sinh embedding mới: (1) sửa hoặc loại ViDeBERTa, chuẩn hóa E5; (2) thêm mốc back-translation cho chỉ số S.
Sau khi hoàn thành, bài báo cần trả lời được câu hỏi trọng tâm mà bản hiện tại còn thiếu: mức suy giảm đồng thuận phân cụm đến từ việc đổi ngôn ngữ hay từ việc đi qua dịch máy?
2. Nguyên tắc chung
1.	Đóng băng dữ liệu. Dùng đúng snapshot cào ngày 2026-09-17, không cào lại (mô tả trên store thay đổi liên tục). Nếu dữ liệu gốc có trường developerId, giữ lại để dùng cho lấy mẫu theo nhóm.
2.	Cache mọi embedding dưới dạng .npy, kèm file manifest ghi: model, revision, recipe, điều kiện input, hash văn bản đầu vào, phiên bản thư viện, thiết bị, dtype, batch size. Phân hoạch KMeans cũng lưu lại để không phải fit lại.
3.	Chạy thử nhỏ trước. Mỗi bước mới chạy trên khoảng 500 app để kiểm tra code, sau đó mới chạy toàn bộ.
4.	Sinh bảng tự động. Kết quả xuất ra CSV, script chuyển thành bảng LaTeX, làm tròn 3 chữ số. Không chép số bằng tay.
Gợi ý cấu trúc thư mục:
data/            snapshot gốc + bản dịch mới (chỉ đọc)
embeddings/      <recipe>__<view>__<condition>.npy + manifest.json
partitions/      nhãn KMeans theo seed
results/         CSV kết quả từng bước
tables/          bảng LaTeX sinh tự động
scripts/         01_e5_check.py, 02_videberta_diag.py, ...

3. Các bước thực hiện
Bước 1 – Khắc phục tính tái lập của E5 (bắt buộc, làm đầu tiên)
Mục tiêu: có một bộ vector chuẩn duy nhất cho mỗi điều kiện.
•	Cố định: cùng GPU (hoặc CPU), float32, cùng batch size, torch.use_deterministic_algorithms(True), model.eval(), torch.no_grad().
•	Sắp văn bản theo độ dài trước khi chia batch, hoặc dùng batch size 1 cho mẫu kiểm tra. Padding khác nhau giữa các batch có thể gây sai khác số học nhỏ trên GPU – đây là nghi phạm số một.
•	Kiểm tra: encode cùng 1.000 văn bản hai lần, so max |v1 − v2| và cosine từng cặp. Đạt khi sai khác < 1e-5.
•	Khi đã ổn định: xóa nhận định “unresolved” trong bài. Với E5, native và 512-token trùng nhau (cap đều là 512) nên chỉ giữ một dòng và ghi chú.
Đầu ra: log so sánh hai lần chạy; bộ vector E5 chuẩn.
Bước 2 – Chẩn đoán ViDeBERTa (bắt buộc)
AMI ≈ 0,02 gần như chắc chắn là lỗi pipeline. Kiểm tra theo thứ tự:
1.	Log khi load model: có cảnh báo “Some weights … newly initialized” không? Nếu có, mô hình đang chạy với trọng số ngẫu nhiên – nguyên nhân phổ biến nhất.
2.	Tokenizer và segmentation: đọc model card của Fsoft-AIC/videberta-base để biết mô hình cần văn bản đã tách từ PyVi hay văn bản thô; so sánh tokenizer fast và slow. In thử 3 câu đã tokenize để kiểm tra bằng mắt.
3.	Kiểm tra vector: NaN, norm trước chuẩn hóa, cosine trung bình giữa 1.000 cặp ngẫu nhiên (> 0,95 nghĩa là vector suy biến).
4.	Thử CLS pooling thay mean pooling; thử centering (trừ vector trung bình) như một chẩn đoán.
# Kiểm tra nhanh mức suy biến của vector (X đã L2-normalize)
idx = np.random.default_rng(0).choice(len(X), (1000, 2))
cos = np.sum(X[idx[:, 0]] * X[idx[:, 1]], axis=1)
print(cos.mean(), cos.std(), np.isnan(X).any())

Quyết định: sửa được thì chạy lại ViDeBERTa; không sửa được thì loại khỏi bài kèm một câu giải thích. Không giữ con số 0,02.
Đầu ra: báo cáo chẩn đoán ngắn (1 trang) và quyết định sửa/loại.
Bước 3 – Recall@k cho RQ1 (dùng vector có sẵn)
•	Truy hồi bản tiếng Việt từ bản tiếng Anh (EN→VI) và ngược lại (VI→EN). Tính Recall@1, Recall@10, MRR.
•	Ma trận 13.657 × 13.657 nên tính theo khối (mỗi lần ~1.000 hàng) hoặc dùng FAISS IndexFlatIP để tránh tràn RAM.
•	Đưa vào Bảng 2 cạnh Δcos; Δcos có thể chuyển xuống phụ lục.
# Recall@k theo khối, E và V là vector EN/VI đã L2-normalize, cùng thứ tự app
ranks = []
for s in range(0, len(E), 1000):
    sim = E[s:s+1000] @ V.T                      # (B, N)
    true = sim[np.arange(sim.shape[0]), np.arange(s, s+sim.shape[0])]
    ranks.append((sim > true[:, None]).sum(1) + 1)
ranks = np.concatenate(ranks)
r1, r10, mrr = (ranks == 1).mean(), (ranks <= 10).mean(), (1 / ranks).mean()

Bước 4 – Thay phần bootstrap “lịch sử” (dùng phân hoạch đã lưu)
•	Xóa toàn bộ WR/UNIQUE/D_MULT và Phụ lục D.
•	Thay bằng subsampling không hoàn lại: lấy ngẫu nhiên 80% app, lặp 500 lần; trên mỗi tập con tính lại AMI và S từ phân hoạch đã lưu (giới hạn nhãn cụm vào tập con, không fit lại KMeans); báo cáo khoảng phân vị 2,5–97,5%.
•	Nếu có developerId: lấy mẫu theo developer thay vì theo app, vì app cùng nhà phát triển thường có mô tả na ná nhau.
•	Ghi rõ trong bài: khoảng này phản ánh biến động theo thành phần app, với điều kiện phân hoạch cố định.
Bước 5 – Linear probe cho taxonomy alignment
•	Logistic regression trên embedding, 5-fold phân tầng theo genre; nếu có developerId thì dùng StratifiedGroupKFold theo developer.
•	Báo cáo macro-F1 và accuracy cho từng recipe, ở benchmark tiếng Việt và hai phía EN/VI của tập cặp.
•	Phần này bổ sung cho AMI–KMeans: AMI đo cấu trúc tự nhiên của không gian vector, probe đo lượng thông tin về genre mà vector chứa.
Bước 6 – Mốc back-translation cho S (quan trọng nhất)
Mục tiêu: tách hiệu ứng đổi ngôn ngữ khỏi hiệu ứng đi qua dịch máy. Sinh thêm hai view:
View mới	Cách tạo	So với	Trả lời câu hỏi
EN_bt	Dịch VI (Google) → EN	EN gốc	Cùng ngôn ngữ, chỉ khác do qua dịch máy
VI_mt2	Dịch EN → VI bằng hệ dịch thứ hai	VI (Google)	Cùng ngôn ngữ, khác hệ dịch

Công cụ dịch – ưu tiên mô hình mở chạy GPU:
•	vinai/vinai-translate-vi2en-v2 và vinai/vinai-translate-en2vi-v2, hoặc
•	NLLB-200 (bản distilled 600M hoặc 1.3B).
•	Nếu dùng Google Cloud Translation: kiểm tra giá hiện hành trước (tổng khoảng 25 triệu ký tự).
Lưu ý kỹ thuật:
•	Mô tả dài ~1.800 ký tự: tách câu trước khi dịch rồi ghép lại, tránh bị cắt cụt hoặc bỏ đoạn.
•	Lưu bản dịch thành file riêng, kèm tên và revision mô hình dịch.
•	Kiểm tra ngẫu nhiên khoảng 30 bản dịch bằng mắt; báo cáo tỷ lệ độ dài trước/sau dịch.
Chạy: encode các view mới bằng 4 recipe của nghiên cứu cặp (TF-IDF–LSA, E5, BGE-M3, Qwen3), fit KMeans 10 seed như cũ, tính S cho từng cặp view. Bảng kết quả dự kiến:
Cặp view	Đổi ngôn ngữ?	Qua dịch máy?	S
EN / VI (Google)	có	có	đã có
EN / EN_bt	không	có	mới
VI (Google) / VI_mt2	không	có (khác hệ)	mới
EN nửa đầu / nửa sau	không	không	đã có (chỉ lexical)

Cách đọc: nếu S của EN/EN_bt gần với S của EN/VI, phần lớn mức suy giảm đến từ dịch máy chứ không phải từ việc đổi ngôn ngữ. Đây là kết luận có giá trị mà bản hiện tại còn thiếu.
Bước 7 – Tùy chọn (nếu còn thời gian)
•	Thêm LaBSE và paraphrase-multilingual-mpnet-base-v2 làm baseline xuyên ngữ kinh điển.
•	Bỏ phần average-linkage, không cần chạy lại.
4. Thứ tự ưu tiên và đầu ra
Ưu tiên	Bước	Embedding mới?	Đầu ra cần nộp
1	E5 tái lập	Có (chỉ E5)	Log so sánh 2 lần chạy, bộ vector chuẩn
2	Chẩn đoán ViDeBERTa	Có (chỉ ViDeBERTa)	Báo cáo chẩn đoán, quyết định sửa/loại
3	Recall@k	Không	Bảng 2 mới
4	Subsampling CI	Không	CI cho S và AMI; xóa Phụ lục D
5	Linear probe	Không	Bảng macro-F1
6	Back-translation	Có (2 view × 4 recipe)	Bảng S theo cặp view, mẫu bản dịch
