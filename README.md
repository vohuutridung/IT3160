# IT3160 — Học chính sách cờ vua có giám sát (ResNet-16×256)

Project môn Nhập môn Trí tuệ nhân tạo (IT3160) — Lớp TTNT-172886, Nhóm G14.

| Thành viên | MSSV |
| --- | --- |
| Võ Hữu Trí Dũng | 20235057 |
| Nguyễn Văn Tiến Dũng | 20235306 |
| Nguyễn Đức Huân | 20235338 |

## Bài toán

Cho một thế cờ (trước lượt đi), dự đoán nước đi mà người chơi mạnh đã thực hiện. Mô hình
xuất phân phối xác suất trên **4672 hành động** (64 ô xuất phát × 73 mặt phẳng nước đi), các
nước không hợp lệ bị che (mask) trước softmax. Khi chơi, hệ thống chọn nước có xác suất cao
nhất sau **một lần suy luận** — không tìm kiếm cây, không self-play.

- **Dữ liệu:** ván cờ Lichess Elite, ~1.500.000 thế cờ cân bằng theo 3 giai đoạn
  (khai cuộc / trung cuộc / tàn cuộc), chia train/validation/test 90/5/5 **theo ván**
  và loại bỏ thế cờ trùng lặp giữa các tập.
- **Mô hình:** ResNet 16 khối residual × 256 kênh, policy head conv 1×1 → 73×8×8.
- **Đánh giá:** loss, Top-1/3/5 accuracy, tách riêng theo giai đoạn ván cờ.

Chi tiết đề xuất và tiến độ: [`Proposal.pdf`](Proposal.pdf), [`Project report N14.ipynb`](Project%20report%20N14.ipynb).

## Cấu trúc repo

```
configs/                 Cấu hình data pipeline (quota, split, phase, nguồn HF)
scripts/
  data/                  Entry point của data pipeline (download → build → validate → upload, export tensor)
  train.py               Huấn luyện ResNet-16×256
  evaluate.py            Đánh giá checkpoint / baseline, ghi predictions + metrics
  compare.py             So sánh nhiều model, kiểm định cặp
  export_meta.py         Metadata từng hàng (FEN, nước đi, Elo) khớp với file HDF5
  play_match.py          Đấu cờ giữa hai model / baseline, xuất W/D/L, Elo, PGN
src/IT3160/
  data/                  Parse PGN, lấy mẫu, canonical hóa, mã hóa nước đi, khử trùng lặp, validate, export HDF5
  models/                Dataset HDF5, mạng ResNet, metric
  evaluation/            Inference, metric, baseline, CI/kiểm định, lát cắt, agent chơi cờ
tests/                   Unit test (pytest) cho data, models và evaluation
run_train.bat            Lệnh train mẫu trên Windows (GPU 4 GB, batch 128 × accum 4)
run_test.bat             Đánh giá checkpoints/best_model.pt trên test set
```

Thư mục `data/` và `checkpoints/` được sinh ra khi chạy và nằm trong `.gitignore`.

## Cài đặt

Yêu cầu Python ≥ 3.10.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Muốn train trên GPU, cài bản PyTorch có CUDA phù hợp theo hướng dẫn tại
<https://pytorch.org/get-started/locally/> **trước** khi chạy lệnh `pip install -e` ở trên.

## Dữ liệu

Toàn bộ dữ liệu (raw, processed, manifest, tensor) được lưu trên Hugging Face:
<https://huggingface.co/datasets/vohuutridung/IT3160>.

### Chỉ cần train: tải sẵn tensor

```bash
hf download vohuutridung/IT3160 --repo-type dataset \
  --include "tensors/official-v1/*" --local-dir data
```

Lệnh trên tạo `data/tensors/official-v1/{train,validation,test}.h5` — đúng đường dẫn mặc
định của `scripts/train.py` và `scripts/evaluate.py`.

| Tập | Số mẫu |
| --- | --- |
| `train.h5` | 1.350.000 |
| `validation.h5` | 75.000 |
| `test.h5` | 75.000 |

Mỗi file HDF5 chứa:

| Dataset | Shape | Kiểu | Ý nghĩa |
| --- | --- | --- | --- |
| `x` | `[N, 17, 8, 8]` | `uint8` | 12 kênh quân + 4 kênh nhập thành + 1 kênh bắt tốt qua đường, đã canonical hóa theo bên đi |
| `y` | `[N]` | `int64` | Nhãn nước đi `0..4671` (`action_id = plane * 64 + from_square`) |
| `legal_mask` | `[N, 4672]` | `bool` | `True` tại các nước hợp lệ |
| `phase` | `[N]` | `uint8` | 0 = khai cuộc, 1 = trung cuộc, 2 = tàn cuộc |

### Tự xây dựng lại dataset

Xem hướng dẫn đầy đủ tại [`src/IT3160/data/README.md`](src/IT3160/data/README.md). Tóm tắt:

```bash
hf auth login
python scripts/data/run_pipeline.py --run-id <run-id> --no-upload   # bỏ --no-upload để publish lên HF
python scripts/data/export_tensors.py --run-id <run-id> --no-upload
```

Muốn chạy thử nhanh, dùng `configs/data.pilot.yaml` (quota nhỏ) qua tùy chọn `--config`.

## Huấn luyện

```bash
python scripts/train.py --batch-size 128 --grad-accum-steps 4 --lr 3e-4 --epochs 8
```

- Optimizer AdamW (`lr=3e-4`, `weight_decay=1e-4`), CosineAnnealingLR theo epoch,
  gradient clipping 1.0, mixed precision (AMP) khi có CUDA.
- Effective batch = `batch-size × grad-accum-steps` (mặc định 512).
- Mỗi epoch lưu `checkpoints/last_checkpoint.pt`; checkpoint có validation loss tốt nhất
  lưu ở `checkpoints/best_model.pt`; lịch sử metric ở `checkpoints/history.json`.
- Tiếp tục train: `--resume checkpoints/last_checkpoint.pt`.
- Chạy thử nhanh: `--limit-train 2000 --limit-val 500 --epochs 1`.

Trên Windows có thể chạy trực tiếp `run_train.bat`.

## Đánh giá

Code nằm trong `src/IT3160/evaluation/` (bảng "module nào làm gì" ở docstring của `__init__.py`).
Model chỉ chạy **một lần** để ghi kết quả từng mẫu ra `predictions.parquet`. Mọi phân tích sau đó
(metric, khoảng tin cậy, lát cắt, so sánh) đọc từ file này. Giải thích dễ hiểu cho người không chuyên:
[`docs/EVAL_OVERVIEW.md`](docs/EVAL_OVERVIEW.md); bản kỹ thuật và kết quả: [`docs/EVALUATION_SUMMARY.md`](docs/EVALUATION_SUMMARY.md).

> Các số trong `history.json` khi train chỉ để theo dõi quá trình học. Số đưa vào báo cáo lấy từ
> `scripts/evaluate.py` (NLL tính trên từng mẫu, có khoảng tin cậy và baseline).

```bash
# 1) (Tùy chọn, một lần) metadata từng hàng: FEN, nước đi, ply, Elo, game_id.
#    Cần cho phân tích theo loại nước / ply / Elo và bootstrap theo ván.
python scripts/export_meta.py --h5 data/tensors/official-v1/test.h5

# 2) Đánh giá checkpoint (mặc định trên test.h5)
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt \
  --meta data/tensors/official-v1/test_meta.parquet

# 3) Baseline để đối chiếu
python scripts/evaluate.py --baseline random
python scripts/evaluate.py --baseline prior --train-path data/tensors/official-v1/train.h5

# 4) So sánh nhiều model trên cùng tập (model đầu tiên là mốc)
python scripts/compare.py --runs checkpoints/eval/best_model__test results/baselines/prior__test \
  --names Model Prior --meta data/tensors/official-v1/test_meta.parquet --out results/compare
```

- `evaluate.py` ghi vào `<thư mục checkpoint>/eval/<checkpoint>__<split>/`: `predictions.parquet`,
  `metrics.json`, `metrics.md`. Kiến trúc mạng được suy ra từ checkpoint. Mặc định chạy fp32
  (CPU và GPU cho cùng kết quả); thêm `--amp` để chạy fp16 nhanh hơn trên GPU.
- `--meta` được kiểm tra khớp với `--data` (số dòng và nhãn từng dòng); ghép nhầm split sẽ báo lỗi.
- `predictions.parquet` và thư mục `runs/` không commit (xem `.gitignore`); chỉ commit `metrics.*` nhỏ.
- Chỉ số: NLL, Top-1/3/5, hạng trung bình của nước thật, MRR, entropy, ECE, tỷ lệ argmax bất hợp lệ
  khi bỏ mask. Tất cả tính tổng và theo từng giai đoạn, kèm **CI 95% bootstrap** (theo ván nếu có
  `--meta`). Có thêm lát cắt theo số nước hợp lệ, loại nước, quân đi, chiếu, ply, Elo, và các ví dụ dự đoán sai nhất.
- `compare.py`: bảng so sánh, ΔTop-1 / ΔNLL so với model mốc kèm CI, kiểm định McNemar cho Top-1.
- `IT3160.evaluation.agent.PolicyAgent`: chọn nước cho một `chess.Board` (argmax hoặc sampling theo nhiệt độ).

### Đấu cờ giữa các model (arena)

```bash
python scripts/play_match.py --a runs/e1/best_model.pt --b random --openings 25
python scripts/play_match.py --a runs/e1/best_model.pt --b runs/e2/best_model.pt --names E1 E2 --openings 50
```

- Người chơi là một checkpoint `.pt`, `random` (nước hợp lệ ngẫu nhiên) hoặc `greedy` (nhìn 1 nước:
  chiếu hết nếu được, nếu không thì ăn quân giá trị cao nhất).
- Khai cuộc là các vị trí khác nhau lấy từ **ván test** ở ply 8–12 (`--min-ply/--max-ply`). Mỗi
  khai cuộc đánh 2 ván đổi màu. Áp dụng luật hòa chuẩn (lặp 3 lần, 50 nước, không đủ quân), và xử hòa
  sau `--max-plies` (300).
- Mặc định model đi nước xác suất cao nhất và bỏ qua nước cho phép đối phương đòi hòa do lặp 3 lần.
  `--temperature 0.3` để sampling, `--allow-repetition` để tắt việc tránh lặp.
- Kết quả ở `results/matches/<A>_vs_<B>/`: `summary.json` (W/D/L của A, điểm, chênh lệch Elo kèm CI 95%
  tính theo cặp khai cuộc, kết quả khi cầm Trắng/Đen, thống kê cách kết thúc ván), `games.pgn`,
  `games.jsonl`, `openings.txt`.

Trên Windows: `run_test.bat`.

## Kiểm thử

```bash
python -m pytest
python -m ruff check src tests scripts
```

Hoặc `sh scripts/data/check.sh` để chạy cả hai. Test đọc dữ liệu thật
(`tests/models/test_models.py::test_dataset_loader`) sẽ tự bỏ qua nếu chưa có
`data/tensors/official-v1/validation.h5`.

## Giấy phép

[Apache License 2.0](LICENSE).
