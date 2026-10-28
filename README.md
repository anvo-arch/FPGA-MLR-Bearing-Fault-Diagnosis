# FPGA Acceleration of Bearing Fault Diagnosis Using Multinomial Logistic Regression

## Quick links

- **Source code DOI:** Pending first Zenodo/GitHub release
- **GitHub repository:** `https://github.com/anvo-arch/FPGA-MLR-Bearing-Fault-Diagnosis`
- **Paper DOI:** Pending IEEE publication DOI
- **Target platform:** AMD/Xilinx PYNQ-Z2 (Zynq-7020)
- **Classifier:** 10-class multinomial logistic regression (MLR)

This repository contains the hardware source code, C-simulation testbenches, training/evaluation notebooks, and fixed-point MLR parameter headers used for FPGA-based bearing-fault diagnosis with the Case Western Reserve University (CWRU) Bearing Data Center recordings.

Two implementations are included:

1. **3-feature accelerator:** Peak, mean absolute value (MAV), and zero crossing (ZC).
2. **6-feature accelerator:** Peak, MAV, ZC, kurtosis, Gaussian negative log-likelihood (NLL), and skewness.

The project evaluates four window sizes and four input precisions:

- Window sizes: `256`, `512`, `1024`, `2048` samples
- Raw input widths: `4`, `8`, `12`, `16` bits
- Raw fixed-point formats: `Q4.0`, `Q4.4`, `Q4.8`, `Q4.12`

The 6-feature implementation uses a single-pass raw-moment formulation for variance, skewness, kurtosis, and NLL so that the input window does not need to be reread from external memory.

---

## Repository layout

The original research files from the project ZIP are intentionally kept in the same directory hierarchy. The files inside `MLR_3feat/` and `MLR_6feat/` are unchanged from the supplied archive.

```text
.
├── README.md
│
├── MLR_3feat/
│   ├── scripts/
│   │   ├── 01_train_cwru_3_feat.ipynb
│   │   └── 02_test_cwru_mlr_3feat.ipynb
│   ├── src/
│   │   ├── logistic_fault.cpp
│   │   ├── logistic_fault.h
│   │   └── logistic_params.h
│   └── testbench/
│       └── tb_logistic.cpp
│
└── MLR_6feat/
    ├── scripts/
    │   ├── 01_train_cwru_6feat.ipynb
    │   └── 02_test_cwru_6feat.ipynb
    ├── src/
    │   ├── logistic_fault.cpp
    │   ├── logistic_fault.h
    │   └── logistic_params_6feat.h
    └── testbench/
        └── tb_logistic.cpp
```

---

## File descriptions

### `MLR_3feat/`

| File | Purpose |
|---|---|
| `scripts/01_train_cwru_3_feat.ipynb` | Downloads/loads CWRU data, generates the 16 window/precision experiments, trains 3-feature MLR models, selects regularization `C`, quantizes coefficients, and generates the master parameter/data files. |
| `scripts/02_test_cwru_mlr_3feat.ipynb` | PYNQ-Z2 hardware verification and timing notebook. Packs raw samples into four 64-bit AXI4-Stream inputs, compares FPGA predictions with the fixed-point software reference, and measures streamed processing time. |
| `src/logistic_fault.cpp` | Vitis HLS implementation of the four-stream input pipeline, Peak/MAV/ZC feature extraction, MLR inference, and DDR prediction write. |
| `src/logistic_fault.h` | Compile-time configuration, data types, fixed-point types, model selector, packet geometry, and HLS top-level declaration. |
| `src/logistic_params.h` | Master 3-feature MLR parameter header containing the trained configurations. |
| `testbench/tb_logistic.cpp` | C-simulation testbench for the HLS implementation. |

### `MLR_6feat/`

| File | Purpose |
|---|---|
| `scripts/01_train_cwru_6feat.ipynb` | Full 4×4 training/quantization sweep for Peak, MAV, ZC, kurtosis, NLL, and skewness. Generates the six-feature model header and PYNQ experiment archive. |
| `scripts/02_test_cwru_6feat.ipynb` | PYNQ-Z2 correctness and timing notebook for the six-feature accelerator. |
| `src/logistic_fault.cpp` | Vitis HLS implementation of Peak/MAV/ZC plus the single-pass higher-order moment pipeline and 10-class MLR. |
| `src/logistic_fault.h` | Compile-time model selection, packet geometry, Q formats, feature packet definition, and HLS interface declaration. |
| `src/logistic_params_6feat.h` | Master six-feature MLR parameter header containing the trained configurations. |
| `testbench/tb_logistic.cpp` | C-simulation testbench for the six-feature HLS implementation. |

---

## Method overview

### Dataset

The notebooks use drive-end vibration recordings from the CWRU Bearing Data Center. The all-load protocol uses four motor speeds: 1797, 1772, 1750, and 1730 r/min.

Ten classes are evaluated: Normal; B007/IR007/OR007; B014/IR014/OR014; and B021/IR021/OR021.

Normal recordings are downsampled to 12 kHz to match the selected fault recordings. Each source recording is split chronologically before window extraction: the first 70% is used for training and the final 30% for testing. Candidate windows use a stride of 16 samples. For each class and RPM, 350 training windows and 150 testing windows are selected, producing 14,000 training windows and 6,000 testing windows for each window-size experiment.

### Fixed-point input sweep

| Input width | Format | LSB |
|---:|---|---:|
| 4 bit | Q4.0 | 1 |
| 8 bit | Q4.4 | 1/16 |
| 12 bit | Q4.8 | 1/256 |
| 16 bit | Q4.12 | 1/4096 |

### Feature sets

**3-feature model**
- Peak
- Mean absolute value (MAV)
- Zero-crossing count (ZC)

**6-feature model**
- Peak
- MAV
- ZC
- Kurtosis
- Gaussian NLL
- Skewness

Peak/MAV/ZC use 12 integer bits with fractional precision following the raw input format. Kurtosis, NLL, and skewness use a fixed `Q16.32` representation.

### MLR training

A `StandardScaler` and scikit-learn multinomial logistic-regression model are trained for each of the 16 configurations. The regularization parameter is selected by balanced-accuracy grid search over:

```text
C ∈ {50, 100, 150, 200, 250, 300, 400, 500, 750, 1000}
```

The scaler is folded into the trained MLR coefficients before hardware export:

```text
W_hw = W / scaler.scale
B_hw = B - W_hw @ scaler.mean
```

Weights are quantized to 18-bit fixed-point representations on a per-feature-column basis. The FPGA selects the class with the largest linear MLR score directly, so softmax is not required.

---

## Software sweep results

The values below are the Q18 coefficient-quantized software-reference accuracies stored in the supplied training notebooks.

### 3-feature MLR accuracy

| Window | 4 bit | 8 bit | 12 bit | 16 bit |
|---:|---:|---:|---:|---:|
| 256 | 36.18% | 68.73% | 76.82% | 76.85% |
| 512 | 46.95% | 82.55% | 88.92% | 89.38% |
| 1024 | 57.22% | 91.27% | 95.33% | 95.50% |
| 2048 | 62.88% | 92.92% | 96.77% | **96.85%** |

### 6-feature MLR accuracy

| Window | 4 bit | 8 bit | 12 bit | 16 bit |
|---:|---:|---:|---:|---:|
| 256 | 46.17% | 77.98% | 79.92% | 79.93% |
| 512 | 58.00% | 91.17% | 91.53% | 91.68% |
| 1024 | 65.00% | 96.93% | 96.87% | 96.97% |
| 2048 | 67.97% | 97.92% | 97.68% | **98.12%** |

The best six-feature model is the 2048-sample, 16-bit configuration, reaching approximately **98.12% accuracy** and **98.11% macro F1**.

---

## FPGA architecture

The HLS accelerator exposes four 64-bit AXI4-Stream inputs, one `m_axi` prediction output to DDR, and AXI-Lite control for the prediction address, `window_count`, `window_size`, and block control.

The four physical input words are combined into one logical 256-bit packet.

| Raw width | Samples / 64-bit bus | Samples / logical packet |
|---:|---:|---:|
| 4 bit | 16 | 64 |
| 8 bit | 8 | 32 |
| 12 bit | 5 | 20 |
| 16 bit | 4 | 16 |

The six-feature HLS DATAFLOW path is conceptually:

```text
4 × 64-bit AXIS inputs
        ↓
256-bit packet
        ↓
MAV → Peak → ZC → raw-moment/statistics stage
        ↓
six independent feature-result FIFOs
        ↓
join six features
        ↓
10-class MLR
        ↓
argmax
        ↓
prediction FIFO
        ↓
M_AXI write to DDR
```

Packet FIFOs have depth 8. Feature/prediction FIFOs have depth 4.

---

## Single-pass higher-order statistics

The six-feature hardware accumulates the first four raw power sums during one pass:

```text
Σx, Σx², Σx³, Σx⁴
```

Power-of-two window sizes allow division by `N` to be implemented by shifts. The raw moments are converted to central moments:

```text
variance = m2 - μ²
μ3 = m3 - 3 μ m2 + 2 μ³
μ4 = m4 - 4 μ m3 + 6 μ² m2 - 3 μ⁴
```

The final statistics are:

```text
skewness = μ3 / variance^(3/2)
kurtosis = μ4 / variance²
```

For Gaussian NLL:

```text
NLL = (N/2) [ ln(2πσ²) + 1 ]
```

This avoids a second residual accumulation pass.

---

# Reproducing the project

## 1. Clone

```bash
git clone https://github.com/anvo-arch/FPGA-MLR-Bearing-Fault-Diagnosis.git
cd FPGA-MLR-Bearing-Fault-Diagnosis
```

## 2. Python environment

```bash
python -m venv .venv
pip install numpy pandas matplotlib scipy scikit-learn jupyter
```

The PYNQ test notebooks must be run in a PYNQ environment with the `pynq` package and board access.

## 3. Train the 3-feature models

```text
MLR_3feat/scripts/01_train_cwru_3_feat.ipynb
```

Run all cells. If needed, the notebook automatically downloads `srigas/CWRU_Bearing_NumPy`.

Generated files include:

```text
logistic_params.h
experiment_data.npz
sweep_model_summary.csv
cwru_mlr_output_.../
```

Copy the generated header to `MLR_3feat/src/logistic_params.h` before HLS synthesis.

## 4. Train the 6-feature models

```text
MLR_6feat/scripts/01_train_cwru_6feat.ipynb
```

Generated files include:

```text
logistic_params_6feat.h
experiment_data_6feat.npz
sweep_model_summary_6feat.csv
cwru_mlr6_output_.../
```

Copy the generated header to `MLR_6feat/src/logistic_params_6feat.h` before HLS synthesis.

## 5. Select one HLS configuration

Open the corresponding `src/logistic_fault.h` and select exactly one model:

```cpp
#define MODEL_W<WINDOW>_B<BITS>
```

Example for the best six-feature paper configuration:

```cpp
#define MODEL_W2048_B16
```

## 6. Vitis HLS

Create an HLS component/project with:

```text
Top function: logistic_fault_predict
Part:         xc7z020-clg400-1
Clock:        10 ns (100 MHz target)
```

3-feature sources:

```text
MLR_3feat/src/logistic_fault.cpp
MLR_3feat/src/logistic_fault.h
MLR_3feat/src/logistic_params.h
MLR_3feat/testbench/tb_logistic.cpp
```

6-feature sources:

```text
MLR_6feat/src/logistic_fault.cpp
MLR_6feat/src/logistic_fault.h
MLR_6feat/src/logistic_params_6feat.h
MLR_6feat/testbench/tb_logistic.cpp
```

Run:

1. C Simulation
2. C Synthesis
3. C/RTL Co-simulation if desired
4. Package/export IP for Vivado

## 7. Vivado integration

The supplied ZIP contains the research source hierarchy, not a full Vivado project reconstruction script. The HLS interface contract is:

```text
input_stream0 : 64-bit AXI4-Stream
input_stream1 : 64-bit AXI4-Stream
input_stream2 : 64-bit AXI4-Stream
input_stream3 : 64-bit AXI4-Stream
predictions   : M_AXI master write to DDR
control       : AXI4-Lite
```

The intended input architecture uses four DMA MM2S paths for the four input streams. Generate the final `.bit` and `.hwh` files and place them beside the appropriate PYNQ notebook, or update the notebook paths.

## 8. PYNQ verification

3-feature notebook:

```text
MLR_3feat/scripts/02_test_cwru_mlr_3feat.ipynb
```

Required alongside the notebook:

```text
logistic_fault.bit
logistic_fault.hwh
experiment_data.npz
```

6-feature notebook:

```text
MLR_6feat/scripts/02_test_cwru_6feat.ipynb
```

Required:

```text
logistic_fault.bit
logistic_fault.hwh
experiment_data_6feat.npz
```

Set:

```python
TEST_WINDOW_SIZE = ...
TEST_SAMPLE_BITS  = ...
```

These must match the `MODEL_W<window>_B<bits>` macro used for the current bitstream.

The notebook performs a one-window debug run, full test-set execution, FPGA-vs-Q18 agreement check, classification metrics, and repeated timing.

---

## Timing interpretation

The reported `µs/window` value is batch-normalized:

```text
batch execution time / window_count
```

It is a steady-state streamed processing time per window rather than the isolated startup latency of one invocation.

---

## Paper result

The six-feature 2048-sample, 16-bit configuration reported in the paper achieves approximately:

```text
Accuracy:   98.12%
Macro F1:   98.11%
Processing: 30.709 µs/window
Normalized: ~15.00 ns/sample
```

---

# Citation

After Zenodo assigns the DOI, replace `DOI_PENDING`:

```bibtex
@software{vo2026fpga_mlr,
  author    = {Vo, Andy and Grover, Radhika S.},
  title     = {FPGA Acceleration of Bearing Fault Diagnosis Using
               Multinomial Logistic Regression: Source Code},
  version   = {1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {DOI_PENDING},
  url       = {https://doi.org/DOI_PENDING}
}
```

---

## License

MIT. See LICENSE.
