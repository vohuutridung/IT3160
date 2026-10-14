# Data Pipeline
**Repo huggingface toàn bộ data:** https://huggingface.co/datasets/vohuutridung/IT3160

## Mô tả rõ từng folder
- **`raw/`**: Các file ZIP chứa ván cờ gốc ở định dạng PGN, dùng làm nguồn đầu vào cho pipeline.

- **`interim/candidates/<run-id>/`**: Các thế cờ ứng viên được lấy mẫu từ ván gốc, chia theo `train/validation/test`. Mỗi mẫu chứa FEN, nước đi người chơi và metadata; chưa phải dữ liệu cuối cùng sau khử trùng lặp và chọn đủ quota.

- **`processed/<run-id>/`**: Dữ liệu đã chọn đủ quota, khử trùng lặp và kiểm tra hợp lệ, lưu thành nhiều shard Parquet. Mỗi dòng gồm thế cờ `fen`, nhãn `action_id`, danh sách `legal_action_ids` và metadata. Bàn cờ vẫn ở dạng FEN, chưa chuyển thành tensor đầu vào.

- **`manifests/<run-id>/`**: Các file JSON ghi nguồn dữ liệu, cấu hình xử lý, số lượng mẫu, checksum và kết quả validation; dùng kiểm tra tính toàn vẹn và truy vết quá trình tạo dataset.

- **`tensors/<run-id>/`**: Dữ liệu được tạo từ `processed` sau khi chạy bước export, **sẵn sàng đưa vào huấn luyện**. Folder gồm đúng **3 file HDF5 có nén**, hỗ trợ đọc từng mẫu hoặc từng batch:
  - **`train.h5`**: 1.350.000 mẫu để huấn luyện.
  - **`validation.h5`**: 75.000 mẫu để theo dõi chất lượng và chọn checkpoint.
  - **`test.h5`**: 75.000 mẫu để đánh giá cuối cùng.

Mỗi file trong **`tensors`** chứa các mảng sau, với `N` là số mẫu của tập đó:

- **`x` — `uint8 [N,17,8,8]`**: Tensor bàn cờ gồm 12 kênh vị trí quân, 4 kênh quyền nhập thành và 1 kênh bắt tốt qua đường. Đã chuẩn hóa theo bên đang đi; khi train chỉ đổi sang `float32`, không xoay hoặc mã hóa lại.

- **`y` — `int64 [N]`**: Nhãn nước đi người chơi đã thực hiện, có giá trị từ **0 đến 4671**. Dùng trực tiếp làm target cho cross-entropy với đầu ra model `[B,4672]`.

- **`legal_mask` — `bool [N,4672]`**: `True` tại các nước hợp lệ của từng thế cờ. Dùng để loại nước không hợp lệ khỏi logits trước khi tính loss hoặc chọn nước đi.

- **`phase` — `uint8 [N]`**: Giai đoạn ván cờ: **0 = khai cuộc, 1 = trung cuộc, 2 = tàn cuộc**; hỗ trợ thống kê và đánh giá riêng từng giai đoạn.

- **`metadata` — thuộc tính trong file**: Ghi commit nguồn, phiên bản encoder và quy ước biểu diễn để truy vết.

**Các phần tử cùng chỉ số luôn thuộc cùng một mẫu:** `x[i]`, `y[i]`, `legal_mask[i]`, `phase[i]`. Phía train chỉ cần đọc các mảng này, giữ chúng đi cùng nhau khi shuffle; không cần xử lý luật cờ hay tính lại nhãn.

## 1. Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## 2. Chạy data pipeline end2end
```bash
hf auth login
python scripts/data/run_pipeline.py --run-id official-v1
```

`hf auth login`: đăng nhập tài khoản Hugging Face có quyền ghi vào repo
`vohuutridung/IT3160`.

`python scripts/data/run_pipeline.py --run-id official-v1`: chạy lần lượt toàn bộ
quy trình sau, tự động upload khi xử lý và kiểm tra thành công:

1. Lấy đủ 5 ZIP đã có trong `raw/` của repo HF. Lưu ZIP và danh sách nguồn tại `data/raw/`;
   file hợp lệ đã tải được dùng lại.
2. Xử lý PGN theo `configs/data.yaml`: chia train/validation/test, lấy mẫu theo
   phase, khử trùng lặp và mã hóa nhãn nước đi. Mục tiêu là 1.500.000 vị trí.
3. Kiểm tra checksum, từng mẫu, nước đi hợp lệ, nhãn, split, rò rỉ dữ liệu và quota.
   Nếu thiếu quota hoặc có lỗi, script dừng trước bước upload.
4. Upload candidates, processed data và manifests vào các folder tương ứng
   trong cùng repo `vohuutridung/IT3160`:

Nếu xử lý đã xong nhưng upload thất bại, chạy riêng (Lệnh này kiểm tra lại run đã lưu rồi upload, không xử lý lại ZIP):

```bash
python scripts/data/upload.py --run-id official-v1
```

## 3. Đã có processed trên HF: xuất tensor rồi upload đúng 3 file

Chạy riêng bước này sau khi pipeline trước đã hoàn tất. Không chạy lại
`run_pipeline.py`.

```bash
python scripts/data/export_tensors.py --run-id official-v1
```

Lệnh tải **chỉ processed và manifest cần thiết** từ repo `vohuutridung/IT3160`,
chốt nguồn tại một commit cụ thể, kiểm tra checksum, rồi chuyển FEN bằng
`encode_board()` của repo. Nhãn và mask được đối chiếu với thế cờ gốc cho từng
mẫu; giữ nguyên thứ tự hàng và các tập đã chia. Không cần tải lại raw/candidates.

Sau khi kiểm tra lại các tensor đã ghi, lệnh upload cả ba file train.h5, validation.h5, test.h5.
Mỗi file chứa:

| Dataset | Shape | Kiểu | Cách dùng |
| --- | --- | --- | --- |
| `x` | `[N, 17, 8, 8]` | `uint8` | Đổi sang float32 khi đưa vào model; đã canonical hóa |
| `y` | `[N]` | `int64` | Nhãn nước đi 0–4671, dùng trực tiếp với cross entropy |
| `legal_mask` | `[N, 4672]` | `bool` | True tại các nước hợp lệ; dùng mask logits |
| `phase` | `[N]` | `uint8` | 0=opening, 1=middlegame, 2=endgame; để đánh giá theo phase |
