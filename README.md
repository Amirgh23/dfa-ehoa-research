# DFA/SFIG-EHOA Research Framework

[![Tests](https://github.com/Amirgh23/dfa-ehoa-research/actions/workflows/tests.yml/badge.svg)](https://github.com/Amirgh23/dfa-ehoa-research/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](#english) | [فارسی](#فارسی)

## English

This repository is a leakage-safe, reproducible research implementation of
**DFA/SFIG-EHOA** (Dual-Feedback / Stability-Feedback Interaction-Guided EHOA)
for binary feature selection. It extends the published Enhanced Hiking
Optimization Algorithm while preserving the original baseline in `ehoa.py`.

**Deliverables:** [English paper (PDF)](output/pdf/DFA_SFIG_EHOA_Paper.pdf) ·
[LaTeX source](paper/main.tex) · [final evidence report](FINAL_RESEARCH_REPORT.md)
· [raw paired runs](results/raw/paper_revised.csv)

### Research contribution

The proposed method adds two ablatable mechanisms outside the baseline fitness:

1. **Regime-aware stability feedback.** Cross-resampling selection reliability,
   population diversity, search progress, and stagnation control the sweep
   factor and inertia. Instability alone never triggers exploration; this avoids
   a self-reinforcing instability loop.
2. **Confidence-gated reliability and interaction guidance.** The binary
   transition combines continuous position, entropy-gated feature reliability,
   and a training-only relevance/redundancy interaction signal.

The four mandatory variants are `EHOA`, `SF-EHOA`, `IG-EHOA`, and full
`DFA-EHOA`. The same seeds, folds, population, iterations, classifier, and
fitness budget are used within each paired experiment.

### Scientific safeguards

- imputation, scaling, and SMOTE are fitted only on training folds;
- reliability and interaction estimates use training data only;
- all variants retain the baseline wrapper fitness (5-NN accuracy plus subset size);
- raw masks, indices, traces, runtime, memory, and evaluation counts are saved;
- balanced accuracy, macro/binary F1, MCC, ROC-AUC, PR-AUC, sensitivity,
  specificity, reduction, redundancy, Jaccard, and Nogueira stability are reported;
- Friedman, paired Wilcoxon, Holm correction, confidence intervals, and effect
  sizes are generated without selective deletion of failed runs.

The bundled experiment uses repeated stratified hold-outs on the same two
versioned scikit-learn datasets (`breast_cancer` and `wine`). It is an honest,
small-benchmark study and is not presented as a reproduction of the original
paper's 33-dataset evaluation.

### Install and run

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python run_experiments.py --config configs\paper_revised.yaml
.venv\Scripts\python generate_artifacts.py --experiment paper_revised
```

Runs are resumable. Completed `(dataset, method, seed)` tuples are skipped.
External numeric CSV datasets can be registered using
`configs/home_datasets.example.yaml`; dataset files are intentionally ignored by Git.

### Repository map

- `ehoa.py`: unchanged EHOA baseline;
- `proposed/`: feedback controller, reliability, interaction, transition, DFA-EHOA;
- `configs/`: reproducible experiment specifications;
- `tests/`: deterministic unit, integration, and edge-case tests;
- `results/`: raw runs, traces, statistics, tables, and figures;
- `paper/`: English LaTeX manuscript and references;
- `output/pdf/`: visually verified compiled manuscript;
- `FINAL_RESEARCH_REPORT.md`: evidence-based claim audit and limitations.

### Claim policy

This repository distinguishes implementation novelty from empirical superiority.
Claims are accepted only when the recorded paired results support them. Negative,
partial, or non-significant findings remain in the report.

## فارسی

این مخزن پیاده‌سازی پژوهشی و قابل‌بازتولید **DFA/SFIG-EHOA** برای انتخاب ویژگی
باینری است. خط مبنای EHOA در فایل `ehoa.py` بدون تغییر نگه داشته شده و نوآوری‌ها
در پوشه `proposed/` قرار دارند.

**خروجی‌ها:** [مقاله انگلیسی PDF](output/pdf/DFA_SFIG_EHOA_Paper.pdf) ·
[سورس LaTeX](paper/main.tex) · [گزارش نهایی شواهد](FINAL_RESEARCH_REPORT.md) ·
[نتایج خام](results/raw/paper_revised.csv)

### نوآوری پژوهش

1. **بازخورد پایداری آگاه از وضعیت جست‌وجو:** قابلیت اعتماد ویژگی‌ها در
   بازنمونه‌گیری، تنوع جمعیت، میزان پیشرفت و رکود، ضریب Sweep و Inertia را کنترل
   می‌کنند. کمبود پایداری به‌تنهایی باعث اکتشاف بیشتر نمی‌شود.
2. **گذار باینری هدایت‌شده با اعتماد و تعامل:** موقعیت پیوسته با قابلیت اعتماد
   دروازه‌گذاری‌شده بر اساس آنتروپی و سیگنال ارتباط/افزونگیِ محاسبه‌شده فقط از
   داده آموزش ترکیب می‌شود.

چهار حالت اجباری `EHOA`، `SF-EHOA`، `IG-EHOA` و `DFA-EHOA` با seed، fold،
جمعیت، تکرار، طبقه‌بند و بودجه یکسان مقایسه می‌شوند.

### تضمین‌های علمی

- پیش‌پردازش و SMOTE فقط روی داده آموزش هر fold انجام می‌شوند؛
- هیچ اطلاعاتی از test وارد انتخاب ویژگی، قابلیت اعتماد یا تعامل نمی‌شود؛
- معیارهای پیش‌بینی، کاهش ویژگی، پایداری، افزونگی، زمان، حافظه و هزینه محاسباتی ذخیره می‌شوند؛
- آزمون‌های آماری جفت‌شده و اصلاح Holm تولید می‌شوند؛
- نتایج ناموفق یا غیرمعنادار حذف نمی‌شوند و ادعای برتری فقط با شواهد پذیرفته می‌شود.

### اجرای کامل

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python run_experiments.py --config configs\paper_revised.yaml
.venv\Scripts\python generate_artifacts.py --experiment paper_revised
```

این آزمایش روی همان دو دیتاست نسخه‌بندی‌شده `breast_cancer` و `wine` اجرا
می‌شود. دامنه محدود دو دیتاست صریحاً در مقاله و گزارش ذکر شده و با ارزیابی ۳۳
دیتاست مقاله اصلی یکسان تلقی نمی‌شود.

## Citation and license

Copyright (c) 2026 Amirgh23. Released under the [MIT License](LICENSE).
Please cite both this software via [`CITATION.cff`](CITATION.cff) and the original
EHOA article: Hegazy et al., *Cluster Computing* 29, 244 (2026),
https://doi.org/10.1007/s10586-026-05946-9.
