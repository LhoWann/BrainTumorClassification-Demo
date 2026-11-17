<div align="center">

<img src="assets/banner.svg" alt="Brain Tumor MRI Classification" width="100%" />

**Simulasi tugas W10 Studi Kasus Pembelajaran Mesin: klasifikasi citra MRI tumor otak dengan 3 model dan 5-fold
cross validation.**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-ee4c2c)

</div>

## Ringkasan

Tugas aslinya (MonkeyPox) dikerjakan ulang dengan dataset lain, yaitu
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Dataset ini
punya 4 kelas (`glioma`, `meningioma`, `notumor`, `pituitary`), 5.600 gambar `Training` dan 1.600 gambar `Testing`.

| Model | Arsitektur | Parameter |
| --- | --- | --- |
| `cnn_scratch` | 3 blok Conv-ReLU-MaxPool (16-32-64) → Flatten → Linear | 224 rb |
| `cnn_modified` | 4 blok Conv-**BatchNorm**-ReLU-MaxPool (32-64-128-256) → **Global Avg Pool** → **Dropout 0.3** → Linear | 390 rb |
| `efficientnet_b0` | EfficientNet-B0 [1] pretrained ImageNet, layer terakhir diganti 4 kelas | 4,0 jt |

## Cara Menjalankan

1. Ekstrak `archive.zip` dari Kaggle ke folder `data/` sehingga ada `data/Training/` dan `data/Testing/`.
2. Install PyTorch CUDA 12.8, lalu paket lainnya:

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # harus: 2.11.0+cu128 True
```

3. Jalankan:

```bash
python main.py                                     # 3 model x 5 fold + evaluasi + grafik
python main.py --epochs 1                          # uji cepat
python main.py --models cnn_scratch cnn_modified   # hanya model tertentu
pytest                                             # tes dengan dataset sintetis
```

Semua hyperparameter ada di `src/config.py`. Default `device = "cuda"` dengan mixed precision bfloat16; program
berhenti jika CUDA tidak terdeteksi (ubah ke `"cpu"` bila tidak punya GPU NVIDIA).

## Struktur

```text
main.py                  pipeline lengkap
data/                    Training/ dan Testing/ (tidak di-upload)
src/
├── config.py            hyperparameter dan path
├── data/
│   ├── dataset.py       baca gambar, Dataset PyTorch, 5 fold stratified
│   └── preprocess.py    crop tepi hitam, resize, augmentasi, normalisasi
├── models/
│   ├── blocks.py        conv_block bersama
│   ├── cnn_scratch.py   model 1
│   ├── cnn_modified.py  model 2
│   └── efficientnet.py  model 3
├── training/
│   └── trainer.py       loop train/valid, simpan bobot terbaik, prediksi
└── evaluation/
    ├── metrics.py       accuracy, precision, recall, F1, ROC-AUC
    └── plots.py         kurva training, confusion matrix, ROC, perbandingan
tests/test_pipeline.py   tes semua komponen
```

## Alur Kerja

### Alur kerja keseluruhan

```mermaid
flowchart TD
    START["python main.py<br/>opsi: --epochs, --folds, --models"]

    subgraph PREP["1. Persiapan data"]
        direction TB
        SEED["Set seed 42<br/>random, numpy, torch<br/>main.py"]
        CEK{"Ada data/Training<br/>dan data/Testing?"}
        ERR["Berhenti: FileNotFoundError<br/>ekstrak archive.zip ke data/"]
        LIST["Baca kelas dan path gambar<br/>Training/ dan Testing/<br/>glioma, meningioma, notumor, pituitary<br/>src/data/dataset.py"]
        PPNG["Contoh preprocessing 1 gambar per kelas<br/>asli vs crop + resize<br/>src/evaluation/plots.py"]
        TF["Muat semua gambar sekali ke RAM: crop + resize 224<br/>transform train: augmentasi, valid dan test: tanpa<br/>src/data/preprocess.py"]
        FOLD["Bagi Training jadi 5 fold<br/>StratifiedKFold, shuffle, seed 42<br/>src/data/dataset.py"]
    end

    subgraph CV["2. Cross validation: loop model x fold"]
        direction TB
        MODEL["Untuk setiap model<br/>cnn_scratch, cnn_modified, efficientnet_b0"]
        subgraph PERFOLD["Untuk setiap fold k = 1 sampai 5"]
            direction TB
            BUILD["Buat model baru dengan bobot awal<br/>build_model<br/>src/models/"]
            LOADER["DataLoader batch 16<br/>train: 4/5 Training, diacak, drop_last<br/>valid: 1/5 Training<br/>main.py"]
            TRAIN["Latih 10 epoch<br/>CrossEntropyLoss, AdamW lr 3e-4<br/>grad accumulation 2, bf16<br/>simpan bobot val loss terendah<br/>src/training/trainer.py"]
            PRED["Prediksi data Testing<br/>probabilitas softmax<br/>src/training/trainer.py"]
            MET["Hitung metrik test<br/>accuracy, precision, recall, F1 macro<br/>ROC-AUC one-vs-rest macro<br/>src/evaluation/metrics.py"]
            PLOT["Grafik fold k<br/>kurva training, confusion matrix, ROC<br/>src/evaluation/plots.py"]
        end
        NEXTF{"Masih ada fold?"}
        NEXTM{"Masih ada model?"}
    end

    subgraph SUM["3. Rangkuman hasil"]
        direction TB
        AGG["Rata-rata ± std metrik dari 5 fold<br/>per model<br/>src/evaluation/metrics.py"]
        TAB["Tabel Markdown per fold + perbandingan<br/>main.py"]
        BAR["Diagram batang perbandingan model<br/>src/evaluation/plots.py"]
    end

    subgraph OUT["Folder outputs/"]
        direction TB
        O1["preprocessing.png"]
        O2["figures/nama_model/<br/>foldk_history.png<br/>foldk_cm.png<br/>foldk_roc.png"]
        O3["results.md"]
        O4["comparison.png"]
    end

    START --> SEED --> CEK
    CEK -->|"tidak"| ERR
    CEK -->|"ya"| LIST
    LIST --> PPNG --> TF --> FOLD --> MODEL
    MODEL --> BUILD --> LOADER --> TRAIN --> PRED --> MET --> PLOT --> NEXTF
    NEXTF -->|"ya"| BUILD
    NEXTF -->|"tidak"| NEXTM
    NEXTM -->|"ya"| MODEL
    NEXTM -->|"tidak"| AGG
    AGG --> TAB
    AGG --> BAR
    PPNG -.-> O1
    PLOT -.-> O2
    TAB -.-> O3
    BAR -.-> O4

    classDef data fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef train fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef eval fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef out fill:#fce7f3,stroke:#db2777,color:#831843
    classDef dec fill:#f1f5f9,stroke:#64748b,color:#0f172a
    class START,SEED,LIST,TF,FOLD,LOADER data
    class MODEL,BUILD,TRAIN train
    class PRED,MET,AGG eval
    class PPNG,PLOT,TAB,BAR,O1,O2,O3,O4 out
    class CEK,NEXTF,NEXTM,ERR dec
```

### Pembagian data 5-fold

```mermaid
flowchart TD
    KAG["Folder data/<br/>hasil ekstrak archive.zip"]
    TRN["Folder Training/<br/>4 kelas"]
    TST["Folder Testing/<br/>4 kelas"]
    SKF["StratifiedKFold<br/>5 bagian, shuffle, seed 42<br/>proporsi kelas tiap bagian sama"]

    subgraph ROT["Rotasi fold: bagian valid selalu berganti"]
        direction TB
        F1["Fold 1: valid = bagian 1 | train = bagian 2, 3, 4, 5"]
        F2["Fold 2: valid = bagian 2 | train = bagian 1, 3, 4, 5"]
        F3["Fold 3: valid = bagian 3 | train = bagian 1, 2, 4, 5"]
        F4["Fold 4: valid = bagian 4 | train = bagian 1, 2, 3, 5"]
        F5["Fold 5: valid = bagian 5 | train = bagian 1, 2, 3, 4"]
        F1 ~~~ F2 ~~~ F3 ~~~ F4 ~~~ F5
    end

    subgraph ONE["Isi satu fold k"]
        direction TB
        TR["TRAIN: 4/5 Training<br/>preprocessing + augmentasi<br/>flip horizontal, rotasi 10 derajat, brightness dan contrast<br/>diacak, drop_last"]
        VA["VALID: 1/5 Training<br/>preprocessing saja, tanpa augmentasi"]
        TE["TEST: seluruh Testing<br/>preprocessing saja, tanpa augmentasi<br/>sama untuk semua fold dan model"]
    end

    W["Update bobot model<br/>backpropagation + AdamW"]
    B["Pilih epoch terbaik<br/>val loss terendah<br/>bobot tidak diubah"]
    M["Metrik akhir fold k<br/>bobot tidak diubah"]

    KAG --> TRN
    KAG --> TST
    TRN --> SKF --> ROT
    ROT --> TR
    ROT --> VA
    TST --> TE
    TR -->|"dipakai untuk belajar"| W
    VA -->|"hanya dinilai"| B
    TE -->|"hanya dinilai, sekali di akhir"| M

    classDef data fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef train fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef valid fill:#ede9fe,stroke:#8b5cf6,color:#3b0764
    classDef test fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef dec fill:#f1f5f9,stroke:#64748b,color:#0f172a
    class KAG,TRN,TST,SKF data
    class F1,F2,F3,F4,F5 dec
    class TR,W train
    class VA,B valid
    class TE,M test
```

### Arsitektur model

```mermaid
flowchart LR
    subgraph S["cnn_scratch, sekitar 224 ribu parameter"]
        direction TB
        S0["Input<br/>3x224x224"]
        S1["Blok 1: Conv3x3 + ReLU + MaxPool2<br/>16x112x112"]
        S2["Blok 2: Conv3x3 + ReLU + MaxPool2<br/>32x56x56"]
        S3["Blok 3: Conv3x3 + ReLU + MaxPool2<br/>64x28x28"]
        S4["Flatten<br/>64 x 28 x 28 = 50176"]
        S5["Linear 50176 ke 4<br/>sekitar 200 ribu parameter di sini"]
        S6["Output 4 kelas"]
        S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6
    end

    subgraph M["cnn_modified, sekitar 390 ribu parameter"]
        direction TB
        M0["Input<br/>3x224x224"]
        M1["Blok 1: Conv3x3 + BatchNorm + ReLU + MaxPool2<br/>32x112x112"]
        M2["Blok 2: Conv3x3 + BatchNorm + ReLU + MaxPool2<br/>64x56x56"]
        M3["Blok 3: Conv3x3 + BatchNorm + ReLU + MaxPool2<br/>128x28x28"]
        M4["Blok 4 BARU: Conv3x3 + BatchNorm + ReLU + MaxPool2<br/>256x14x14"]
        M5["Global Average Pooling BARU<br/>256x1x1"]
        M6["Flatten<br/>256"]
        M7["Dropout 0.3 BARU<br/>256"]
        M8["Linear 256 ke 4"]
        M9["Output 4 kelas"]
        M0 --> M1 --> M2 --> M3 --> M4 --> M5 --> M6 --> M7 --> M8 --> M9
    end

    subgraph E["efficientnet_b0, sekitar 4,0 juta parameter"]
        direction TB
        E0["Input<br/>3x224x224"]
        E1["Stem: Conv3x3 stride 2 + BatchNorm + SiLU<br/>32x112x112"]
        E2["16 blok MBConv dalam 7 tahap<br/>16x112x112, 24x56x56, 40x28x28<br/>80x14x14, 112x14x14, 192x7x7, 320x7x7"]
        E3["Conv1x1 + BatchNorm + SiLU<br/>1280x7x7"]
        E4["Global Average Pooling + Flatten<br/>1280"]
        E5["Dropout 0.2<br/>1280"]
        E6["classifier[1] diganti<br/>Linear 1280 ke 4"]
        E7["Output 4 kelas"]
        E0 --> E1 --> E2 --> E3 --> E4 --> E5 --> E6 --> E7
    end

    S ~~~ M ~~~ E

    classDef io fill:#f1f5f9,stroke:#64748b,color:#0f172a
    classDef layer fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef head fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef changed fill:#fee2e2,stroke:#dc2626,color:#7f1d1d,stroke-width:2px
    classDef pre fill:#dcfce7,stroke:#16a34a,color:#14532d
    class S0,S6,M0,M9,E0,E7 io
    class S1,S2,S3 layer
    class S4,S5,M6,M8 head
    class M1,M2,M3,M4,M5,M7 changed
    class E1,E2,E3,E4,E5 pre
    class E6 changed
```

### Loop training per epoch

```mermaid
flowchart TD
    CALL["train_fold dipanggil<br/>model, train_loader, val_loader, 10 epoch, lr 3e-4, accum 2"]
    INIT["Pindahkan model ke GPU atau CPU<br/>loss = CrossEntropyLoss<br/>optimizer = AdamW lr 3e-4<br/>best_loss = tak hingga<br/>best_state = salinan bobot awal"]
    EP{"Masih ada epoch?<br/>epoch ke-1 sampai ke-10"}

    subgraph RT["run_epoch mode train, data train"]
        direction TB
        T0["model.train<br/>gradien aktif, Dropout aktif<br/>BatchNorm pakai statistik batch"]
        T1["Ambil 1 batch: 16 gambar teraugmentasi"]
        T2["Forward: logits = model gambar"]
        T3["loss = CrossEntropy logits vs label"]
        T4["backward loss dibagi 2<br/>gradien dijumlahkan"]
        ACC{"Sudah 2 batch atau<br/>batch terakhir?"}
        UPD["optimizer.step + zero_grad<br/>BOBOT DIPERBARUI"]
        T5["Tambah loss dan jumlah tebakan benar"]
        T6{"Masih ada batch?"}
    end

    subgraph RV["run_epoch mode evaluasi, data valid"]
        direction TB
        V0["model.eval + tanpa gradien<br/>Dropout mati, BatchNorm pakai running stats"]
        V1["Ambil 1 batch valid tanpa augmentasi"]
        V2["Forward + hitung loss<br/>TANPA update bobot"]
        V3["Tambah loss dan jumlah tebakan benar"]
        V4{"Masih ada batch?"}
    end

    HIST["Catat ke history<br/>train loss, val loss, train acc, val acc"]
    CMP{"val loss lebih kecil<br/>dari best_loss?"}
    SAVE["best_loss = val loss<br/>best_state = salinan bobot epoch ini"]
    LOAD["Muat best_state ke model"]
    RET["Kembalikan model terbaik + history<br/>lanjut prediksi data Testing"]

    CALL --> INIT --> EP
    EP -->|"ya"| T0 --> T1 --> T2 --> T3 --> T4 --> ACC
    ACC -->|"ya"| UPD --> T5
    ACC -->|"tidak"| T5 --> T6
    T6 -->|"ya"| T1
    T6 -->|"tidak: rata-rata loss dan akurasi train"| V0
    V0 --> V1 --> V2 --> V3 --> V4
    V4 -->|"ya"| V1
    V4 -->|"tidak: rata-rata loss dan akurasi valid"| HIST
    HIST --> CMP
    CMP -->|"ya"| SAVE --> EP
    CMP -->|"tidak"| EP
    EP -->|"tidak, 10 epoch selesai"| LOAD --> RET

    classDef data fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef train fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef valid fill:#ede9fe,stroke:#8b5cf6,color:#3b0764
    classDef best fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef dec fill:#f1f5f9,stroke:#64748b,color:#0f172a
    class CALL,INIT,HIST data
    class T0,T1,T2,T3,T4,T5,UPD train
    class V0,V1,V2,V3 valid
    class SAVE,LOAD,RET best
    class EP,T6,V4,CMP,ACC dec
```

## Metode

- **Pembagian data:** `Training` dibagi 5 fold stratified (4 bagian train, 1 bagian valid). `Testing` hanya untuk
  testing. Data valid/test tidak pernah dipakai untuk update bobot.
- **Preprocessing:** crop tepi hitam MRI dan resize 224x224 dilakukan sekali di awal (hasil disimpan di RAM, ~1,1 GB),
  lalu normalisasi ImageNet. Augmentasi khusus train: flip horizontal, rotasi ±10°, brightness/contrast jitter.
- **Loss / optimizer:** CrossEntropyLoss, AdamW (lr 3e-4), 10 epoch, GPU CUDA + autocast bfloat16. Batch 16 dengan
  gradient accumulation 2 langkah (batch efektif 32). Bobot dengan val loss terendah dipakai saat testing.
- **Evaluasi:** confusion matrix, accuracy, precision, recall, F1 (macro), ROC-AUC (one-vs-rest) pada `Testing`,
  dirata-rata dari 5 fold.

## Hasil

Tempel isi `outputs/results.md` setelah `python main.py` selesai:

| Model | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| cnn_scratch | ... | ... | ... | ... | ... |
| cnn_modified | ... | ... | ... | ... | ... |
| efficientnet_b0 | ... | ... | ... | ... | ... |

Gambar: `outputs/preprocessing.png`, `outputs/comparison.png`, dan `outputs/figures/<model>/fold{k}_{history,cm,roc}.png`.

## Kerangka Laporan (Template Engineering)

Abstrak (150-200 kata) → Pendahuluan → Kajian Teori (CNN, BatchNorm, dropout, transfer learning, metrik) → Metode
(bagian di atas) → Hasil dan Pembahasan (tabel + grafik, efek tiap modifikasi, scratch vs pretrained, kelas yang
sering tertukar) → Kesimpulan → Referensi (IEEE).

> Jangan upload dataset (`data/`) atau bobot model (`*.pt`, `*.pth`) ke GitHub; sudah diatur di `.gitignore`.

## Referensi

[1] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. ICML*,
1.    [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)