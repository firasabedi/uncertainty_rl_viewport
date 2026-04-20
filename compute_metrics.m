# Datasets — Setup Instructions

This file explains how to download and prepare the three public head-movement
datasets used in the paper.

---

## Expected File Locations

Place raw dataset files in `data/raw/`:

```
uncertainty_rl_viewport/
└── data/
    └── raw/
        ├── david_mmsys.csv     ← David MMSys dataset
        ├── xu_pami.csv         ← Xu PAMI dataset
        └── xu_cvpr.csv         ← Xu CVPR dataset
```

If the files are **not found**, `load_dataset.m` automatically generates
**synthetic data** for development and testing purposes.

---

## CSV Format

Each CSV file must have the following columns (no header required):

| Column | Name       | Type  | Description                        |
|--------|------------|-------|------------------------------------|
| 1      | user_id    | int   | Participant identifier             |
| 2      | video_id   | int   | Video sequence identifier          |
| 3      | timestamp  | float | Time in seconds                    |
| 4      | x          | float | Head position X (3D Cartesian)     |
| 5      | y          | float | Head position Y                    |
| 6      | z          | float | Head position Z                    |

---

## Dataset 1: David MMSys

**Citation:** David, E.J. et al., "A Dataset of Head and Eye Movements for
360 Videos," ACM MMSys 2018.

**Download:** https://dl.acm.org/doi/10.1145/3204949.3208139

**Stats:** 57 participants, 19 panoramic videos, ~19,100 samples

---

## Dataset 2: Xu PAMI (Xu_PAMI)

**Citation:** Xu, M. et al., "Predicting Head Movement in Panoramic Video:
A Deep Reinforcement Learning Approach," IEEE TPAMI 41(11), 2019.

**Download:** https://github.com/YuhangSong/DHP

**Stats:** 58 users, 75 videos, ~65,000 samples

---

## Dataset 3: Xu CVPR (Xu_CVPR)

**Citation:** Xu, Y. et al., "Gaze Prediction in Dynamic 360° Immersive
Videos," CVPR 2018.
*(Note: different first author — Xu Yanyu vs. Xu Mingliang above)*

**Download:** https://github.com/xuyanyu-shh/VR-EyeTracking

**Stats:** 208 users, 208 videos, ~340,000 samples

---

## Preprocessing Notes

1. Convert spherical coordinates (yaw/pitch) to 3D Cartesian (x,y,z) if
   the raw data is in angular format:
   ```matlab
   x = cos(pitch) .* cos(yaw);
   y = cos(pitch) .* sin(yaw);
   z = sin(pitch);
   ```

2. Normalize positions to zero mean and unit variance per axis (optional
   but improves LMS convergence):
   ```matlab
   positions = (positions - mean(positions)) ./ std(positions);
   ```

3. Sort rows by `(user_id, video_id, timestamp)` before saving.

---

## Running with Synthetic Data (no download required)

```matlab
>> quick_demo    % fast smoke-test (~2 min)
>> main          % full experiment suite
```

The synthetic generator in `data/load_dataset.m` produces AR(1) head-motion
traces with injected saccades, matching the statistical properties of the
real datasets at approximate scale.
