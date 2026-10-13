# IT3160 — Data

## 1. Setup

Chạy từ thư mục gốc repo, dùng Python 3.10 trở lên.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Ba lệnh trên tạo môi trường, kích hoạt môi trường và cài các thư viện cần thiết.
Khi mở terminal mới, chạy lại `source .venv/bin/activate`.

## 2. Chạy data chính thức từ đầu đến cuối

```bash
hf auth login
python scripts/data/run_pipeline.py --run-id official-v1
```

`hf auth login`: đăng nhập tài khoản Hugging Face có quyền ghi vào repo
`vohuutridung/IT3160`.

`python scripts/data/run_pipeline.py --run-id official-v1`: chạy lần lượt toàn bộ
quy trình sau, tự động upload khi xử lý và kiểm tra thành công:

1. Lấy đủ 5 ZIP đã có trong `raw/` của repo HF, theo snapshot trong
   `configs/hf_sources.yaml`. Lưu ZIP và danh sách nguồn tại `data/raw/`;
   file hợp lệ đã tải được dùng lại.
2. Xử lý PGN theo `configs/data.yaml`: chia train/validation/test, lấy mẫu theo
   phase, khử trùng lặp và mã hóa nhãn nước đi. Mục tiêu là 1.500.000 vị trí.
3. Kiểm tra checksum, từng mẫu, nước đi hợp lệ, nhãn, split, rò rỉ dữ liệu và quota.
   Nếu thiếu quota hoặc có lỗi, script dừng trước bước upload.
4. Upload candidates, processed data và manifests vào các folder tương ứng
   trong cùng repo `vohuutridung/IT3160`:

```text
raw/                                         # ZIP nguồn đã có
interim/candidates/official-v1/{train,validation,test}/
processed/official-v1/{train,validation,test}/
manifests/official-v1/
```

Kết quả local nằm ở `data/runs/official-v1/`. Mỗi bản dựng mới dùng một `--run-id`
mới, ví dụ `official-v2`; script từ chối ghi đè run đã tồn tại.
Nếu xử lý đã xong nhưng upload thất bại, chạy riêng:

```bash
python scripts/data/upload.py --run-id official-v1
```

Lệnh này kiểm tra lại run đã lưu rồi upload, không xử lý lại ZIP.

## 3. Pilot và kiểm thử

```bash
python scripts/data/run_pipeline.py --run-id pilot-v1 \
  --config configs/data.pilot.yaml --no-upload
sh scripts/data/check.sh -q
```

Lệnh pilot chạy cùng quy trình với mục tiêu 15.000 vị trí, kiểm tra đầy đủ và lưu
local; `--no-upload` tắt bước công bố lên HF. `check.sh` chạy tests và lint.

## 4. Lấy bù nếu cần

Chỉ dùng khi run còn thiếu quota và bạn có các tháng **chưa được dùng trong run**.
Nếu đã xử lý đủ 5 ZIP hiện tại thì cần bổ sung nguồn mới trước khi lấy bù.
Ví dụ dưới đây chỉ áp dụng nếu `official-v1` chưa dùng tháng `2021-03`:

```bash
python scripts/data/download.py --months 2021-03 \
  --manifest data/raw/extra-sources.json
python scripts/data/refill.py --sources data/raw/extra-sources.json --run-id official-v1
python scripts/data/upload.py --run-id official-v1
```

`download.py` lấy các tháng được chọn từ HF và ghi manifest riêng.
`refill.py` thêm nguồn chưa dùng, giữ cấu hình của run và dựng lại processed data.
`upload.py` chỉ công bố khi run đủ quota và vượt qua kiểm tra đầy đủ.

Xem tùy chọn của mỗi script bằng `python scripts/data/<ten_script>.py --help`.
