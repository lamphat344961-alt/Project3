# Kaggle chỉ sinh embeddings E5 — Chặng 02

Notebook cũng chạy trên Colab GPU: upload ZIP vào Files (`/content`) hoặc điền `INPUT_PATH`; working directory tự dùng `/content` khi không có Kaggle Input. Tải ZIP kết quả qua Files. Không cần đổi code E5 hoặc upload lại input khi chỉ sửa cell môi trường.

1. Upload `paac_vn_stage02_kaggle.zip` thành **Private Dataset**. Gói có code đúng commit và **hai Parquet đầu vào**; không cần NPY lịch sử.
2. Import `PAAC_VN_Stage02_Kaggle.ipynb`, Add Input dataset vừa tạo, bật **GPU + Internet**.
3. Sau khi tác giả phê duyệt chạy đủ sáu cells, điền `APPROVED_BY`, `APPROVAL_REFERENCE`, đặt `AUTHOR_APPROVED_FULL=True` ở cell đầu rồi **Run All**. Giữ False nếu chưa được duyệt: không chạy inference. Không sửa contract.
4. Tải **`stage02_embeddings.zip`** từ Output `/kaggle/working/` về local. Nếu inference lỗi, ZIP chứa phần đã lưu cùng log; trạng thái hoàn thành đủ sáu cells phải là `INFERENCE_COMPLETE_UNVALIDATED`.

Mỗi cell xuất:

- `<cell>.npy`: embeddings theo đúng thứ tự app ban đầu.
- `<cell>.app_ids.json`: danh sách app IDs theo thứ tự hàng NPY.
- `<cell>.rows.parquet`: row_idx, appId, genreId.
- `<cell>.json`: recipe, runtime, prepared-text/batch-order/input hashes, source commit và trạng thái `PENDING_LOCAL`.

ZIP còn có execution metadata, checkpoint asset hashes, môi trường và log. Hai TN1 cells có 23,883 apps/cell; bốn TN2 cells có 13,657 apps/cell. Sáu cells encode riêng; giữ FP32, batch8, cap512, query prefix, masked mean pooling và char_budget950 như contract.

**Kaggle không chạy smoke comparison, token-equality audit, validation vectors, historical comparison, clustering hoặc thống kê.** Thực hiện các bước này tại local sau khi tải ZIP; `INFERENCE_COMPLETE_UNVALIDATED` chỉ nghĩa là đã xuất xong, không phải kết quả đã được nghiệm thu. Các guards nhỏ về input hashes/IDs, runtime và ghi file vẫn giữ để tránh dùng nhầm dữ liệu hoặc ghi đè.

Notebook dùng PyTorch GPU sẵn có của Kaggle, cài dependencies riêng và ghi phiên bản thực tế. Không đổi GPU/package/precision/batch giữa chừng. Khi lỗi, tải ZIP/log rồi xử lý; không tự giảm batch hoặc chuyển CPU. Nếu session mất, dùng run ID mới, không ghi đè kết quả đã có.

Dựng lại artifact: `python -m pipeline.package_e5_kaggle --output-dir <thu_muc_moi>`. Bản giao mới ở `output/kaggle_inference/`; gói cũ có historical NPY không còn là đầu vào cần dùng.

Nếu đã gặp `No module named pip` hoặc lỗi startup `sitecustomize` thiếu `wrapt`: dùng notebook đã sửa, hoặc chạy nội dung `REPAIR_ENVIRONMENT_CELL.py` trong một cell của session đang lỗi rồi chạy lại cell môi trường cũ. Không xóa dữ liệu/venv, không cài lại torch. Bản sửa dùng pip của Python gốc với `--python` để quản lý venv thiếu pip, cài startup dependency và kế thừa các thư mục packages nền của image, ưu tiên packages venv. Tham khảo [pip --python](https://pip.pypa.io/en/stable/topics/python-option/).
