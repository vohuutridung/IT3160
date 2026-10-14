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

## 3. Đã có processed trên HF: xuất tensor rồi upload đúng 3 file

Chạy riêng bước này sau khi pipeline trước đã hoàn tất. Không chạy lại
`run_pipeline.py`.

```bash
python -m pip install -e .
hf auth login
python scripts/data/export_tensors.py --run-id official-v1
```

Lệnh tải **chỉ processed và manifest cần thiết** từ repo `vohuutridung/IT3160`,
chốt nguồn tại một commit cụ thể, kiểm tra checksum, rồi chuyển FEN bằng
`encode_board()` của repo. Nhãn và mask được đối chiếu với thế cờ gốc cho từng
mẫu; giữ nguyên thứ tự hàng và các tập đã chia. Không cần tải lại raw/candidates.

Sau khi kiểm tra lại các tensor đã ghi, lệnh upload cả ba file trong **một commit**:

```text
tensors/official-v1/train.h5
tensors/official-v1/validation.h5
tensors/official-v1/test.h5
```

HDF5 có nén và đọc theo batch, không cần nạp toàn bộ file vào RAM. Mỗi file chứa:

| Dataset | Shape | Kiểu | Cách dùng |
| --- | --- | --- | --- |
| `x` | `[N, 17, 8, 8]` | `uint8` | Đổi sang float32 khi đưa vào model; đã canonical hóa |
| `y` | `[N]` | `int64` | Nhãn nước đi 0–4671, dùng trực tiếp với cross entropy |
| `legal_mask` | `[N, 4672]` | `bool` | True tại các nước hợp lệ; dùng mask logits |
| `phase` | `[N]` | `uint8` | 0=opening, 1=middlegame, 2=endgame; để đánh giá theo phase |

Với quota chính thức, N là 1.350.000 / 75.000 / 75.000 tương ứng ba tập.
Provenance, commit nguồn và quy ước encoding nằm trong thuộc tính `metadata`
của từng file; folder HF chỉ có ba file trên.

File local nằm trong `data/tensors/official-v1/`. Nếu upload lỗi, chạy lại
**cùng lệnh** để dùng lại các file đã xuất và kiểm tra checksum trước khi upload.
Nếu folder HF đã có dữ liệu khác, script từ chối ghi đè; chọn folder mới bằng
`--export-id tensors-v2`. Folder đã chứa đúng ba file giống hệt được coi là đã xong.
Có thể dùng `--no-upload` để chỉ xuất và kiểm tra local. Nguồn cũng có thể được
chốt bằng `--revision <commit-sha>`; mặc định lấy main rồi phân giải thành commit.

Ví dụ phía train tải đúng một file và đọc một batch, **không dùng thư viện chess**:

```python
import h5py
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="vohuutridung/IT3160",
    repo_type="dataset",
    filename="tensors/official-v1/train.h5",
)
with h5py.File(path, "r") as data:
    x = data["x"][:256].astype("float32")  # [B, 17, 8, 8], giá trị 0/1
    y = data["y"][:256]                   # [B], int64
    legal_mask = data["legal_mask"][:256] # [B, 4672], bool
```

Trainer cần PyTorch riêng; có thể chuyển các mảng trên bằng `torch.from_numpy()`.
Policy head xuất `[B, 4672]`, hoặc flatten `[B, 73, 8, 8]` theo channel-first;
đặt logits của nước không hợp lệ thành `-inf` rồi tính cross entropy với `y`.
Không canonical hóa, xoay/lật bàn cờ hoặc mã hóa nhãn lại.
Khi shuffle, dùng cùng chỉ số cho `x`, `y`, mask. HDF5 hỗ trợ lấy từng hàng;
nếu lấy nhiều chỉ số qua fancy indexing, h5py yêu cầu chỉ số tăng dần, sau đó
đảo thứ tự cả batch cùng nhau nếu cần. Với DataLoader nhiều worker, mỗi worker
mở file HDF5 riêng sau khi worker được tạo, không chia sẻ handle đã mở.

## 4. Pilot và kiểm thử

```bash
python scripts/data/run_pipeline.py --run-id pilot-v1 \
  --config configs/data.pilot.yaml --no-upload
sh scripts/data/check.sh -q
```

Lệnh pilot chạy cùng quy trình với mục tiêu 15.000 vị trí, kiểm tra đầy đủ và lưu
local; `--no-upload` tắt bước công bố lên HF. `check.sh` chạy tests và lint.

## 5. Lấy bù nếu cần

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
