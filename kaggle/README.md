# Chạy E5 Chặng 02 trên Kaggle

1. Tạo **Private Dataset** từ `paac_vn_stage02_kaggle.zip`. Gói chứa code ở đúng commit, hai Parquet đã freeze và đủ sáu NPY lịch sử; không cần clone GitHub hay tự tìm thêm file.
2. Tạo Notebook → **File → Import Notebook** → chọn `PAAC_VN_Stage02_Kaggle.ipynb`. Thêm dataset vừa tạo bằng **Add Input**. Bật **Accelerator → GPU** và **Internet**. Notebook dùng `cuda:0` kể cả khi có hai GPU.
3. Chạy từ đầu qua phần smoke. Không cần sửa đường dẫn khi chỉ gắn một gói input. Notebook cài dependency trong venv riêng, giữ PyTorch CUDA của Kaggle, rồi chạy preflight → smoke hai lần encode trên sáu cells. `READY_FOR_SMOKE` là preflight; `SMOKE_PASS` mới là smoke đạt.
4. Tải `stage02_smoke_results.zip` từ Output `/kaggle/working/`. ZIP chứa receipts, log, môi trường, token audit và hai lượt vectors smoke. Xem bảng max absolute difference trong notebook; gửi smoke receipt để tác giả quyết định full.
5. **Chỉ sau khi được tác giả phê duyệt full**, điền tên người duyệt và tham chiếu phê duyệt trong cell “Phê duyệt full”, đổi `AUTHOR_APPROVED_FULL=True`, rồi chạy lại hai code cells cuối. Notebook tạo authorization riêng gắn đúng smoke và chạy full; không sửa contract hoặc đổi package/GPU giữa smoke và full. Nếu chưa được duyệt, giữ False: “Run All” sẽ bỏ qua full.
6. Khi full đạt, tải `stage02_full_results.zip` từ Output. Gồm sáu embeddings mới, row sidecars, metadata, historical comparisons và receipts. Không tải cache checkpoint hoặc venv.

Giữ nguyên session sau smoke để dùng cùng runtime. Nếu session mất, cần chạy lại setup/smoke trong session mới; không dùng smoke từ GPU/package khác. Nếu smoke OOM hoặc gặp lỗi deterministic op, tải log/receipt rồi gửi lại; không tự giảm batch, dùng FP16 hay tắt deterministic flags. Các lần chạy sử dụng run IDs mới, không ghi đè kết quả cũ.

Notebook chưa được chạy trên Kaggle trong phiên local. Local đã kiểm tra cú pháp, đóng gói/khôi phục đúng commit và đầu vào; GPU smoke/full phải thực hiện tại Kaggle. Để dựng lại gói từ checkout: `python -m pipeline.package_e5_kaggle --output-dir <thu_muc_moi>`.

Tham khảo giao diện: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks). Môi trường GPU dùng PyTorch có sẵn trong image; notebook lưu phiên bản thực tế và không nhận bản CPU làm GPU runtime.
