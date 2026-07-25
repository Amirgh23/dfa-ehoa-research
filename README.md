# English Overview

This repository contains the source code and documentation for Amirgh23's **ehoa-feature-selection** project. Follow the English sections and commands below for setup and usage.

# EHOA Feature Selection

[![Tests](https://github.com/Amirgh23/ehoa-feature-selection/actions/workflows/tests.yml/badge.svg)](https://github.com/Amirgh23/ehoa-feature-selection/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[فارسی](#فارسی) | [English](#english)

Reproducible Python implementation of the Enhanced Hiking Optimization Algorithm (EHOA) for feature selection, based on:

> Hegazy et al. (2026), *An enhanced Hiking optimization algorithm for accurate and interpretable feature selection in medical data classification*, Cluster Computing. [DOI: 10.1007/s10586-026-05946-9](https://doi.org/10.1007/s10586-026-05946-9)

---

## فارسی

این مخزن یک پیاده‌سازی آموزشی و قابل‌بازتولید از الگوریتم بهبودیافته بهینه‌سازی کوهنوردی (EHOA) برای انتخاب ویژگی است. چارچوب ارزیابی به‌گونه‌ای طراحی شده که از نشت داده جلوگیری کند و علاوه بر دقت، میزان کاهش ویژگی‌ها و توضیح‌پذیری مدل را نیز گزارش دهد.

### مسیر سریع برای ارائه

- [گزارش کامل فارسی](docs/REPORT_FA.md)
- [پاورپوینت آماده ارائه](docs/EHOA_Presentation_FA.pptx)
- [سناریوی ارائه و پاسخ به سؤال‌های استاد](docs/PRESENTATION_FA.md)
- [جدول تطبیق معادلات مقاله با کد](docs/ARTICLE_TO_CODE.md)
- [گزارش خودکار آخرین اجرا](results/REPORT.md)

تست و اجرای دموی ارائه با یک فرمان:

```powershell
.\run_demo.ps1
```

### قابلیت‌های پیاده‌سازی‌شده

- مقداردهی جمعیت با ۱۰ نگاشت آشوبی و نگاشت پیش‌فرض **Tent**؛
- ضریب sweep تطبیقی خطی، inertia weight و به‌روزرسانی PSO-like مطابق معادلات ۷ و ۹؛
- تبدیل باینری S-shaped مطابق معادلات ۱۰ و ۱۱؛
- تابع برازندگی چندهدفه با `alpha=0.99` و 5-NN مطابق معادله ۱۲؛
- اعتبارسنجی متقاطع stratified ده‌تایی در پروفایل مقاله؛
- پیش‌پردازش بدون نشت داده: imputation، scaling و SMOTE فقط روی بخش آموزش هر fold؛
- ارزیابی hold-out با KNN، Logistic Regression، SVM و Random Forest؛
- baseline تمام ویژگی‌ها، چند seed مستقل و ثبت زمان اجرا؛
- permutation importance واقعی و SHAP اختیاری؛
- تولید خروجی‌های CSV، JSON، گزارش Markdown و نمودارهای آماده ارائه.

### نتایج اجرای سریع

اعداد زیر با `seed=42` و فقط روی مجموعه آزمون دست‌نخورده محاسبه شده‌اند:

| دیتاست | ویژگی‌های منتخب | کاهش ویژگی | دقت KNN | Balanced Accuracy |
|---|---:|---:|---:|---:|
| Breast Cancer Wisconsin | 15 / 30 | 50.00% | 92.11% | 92.76% |
| Wine | 6 / 13 | 53.85% | 97.22% | 97.62% |

![نمودار همگرایی Breast Cancer](results/breast_cancer/convergence.png)

نتایج کامل طبقه‌بندها در مسیر `results/<dataset>/classifier_metrics.csv` قرار دارند.

### نصب و اجرا

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py --profile quick --verbose
```

نصب قابلیت اختیاری SHAP:

```powershell
.venv\Scripts\python -m pip install -r requirements-explain.txt
```

اجرای تنظیمات مقاله، شامل ۳۰ عامل، ۵۰ تکرار، ۱۰ fold و ۲۰ اجرای مستقل، زمان‌بر است:

```powershell
.venv\Scripts\python main.py --profile paper --datasets breast_cancer --explain shap
```

دموی داده‌های پُربعد برای درس کلان‌داده:

```powershell
.venv\Scripts\python main.py --profile quick --datasets high_dimensional
```

### خروجی‌ها

برای هر دیتاست در `results/<dataset>/` فایل‌های زیر تولید می‌شوند:

- `runs.csv`: نتایج seedهای مستقل؛
- `classifier_metrics.csv`: مقایسه ویژگی‌های منتخب EHOA با تمام ویژگی‌ها؛
- `selected_features.csv`: نام و اندیس ویژگی‌های منتخب؛
- `convergence.csv/png`: روند fitness، accuracy، تعداد ویژگی و تنوع جمعیت؛
- `confusion_matrix_knn.png`: ماتریس درهم‌ریختگی روی test دست‌نخورده؛
- `feature_importance.csv/png`: اهمیت واقعی ویژگی‌ها؛
- `classifier_comparison.png`: مقایسه طبقه‌بندها روی hold-out؛
- `REPORT.md`: گزارش خودکار تنظیمات و نتایج اجرا.

### نکته علمی مهم

این مخزن چارچوب صحیح پیاده‌سازی و آزمایش را فراهم می‌کند؛ اما پروفایل `quick` فقط برای smoke test و دموی کلاسی است. ادعای بازتولید کامل مقاله نیازمند همان ۳۳ دیتاست، ۲۰ اجرای مستقل و مقایسه آماری با الگوریتم‌های رقیب است.

### تست

```powershell
.venv\Scripts\python -m pytest -q
```

---

## English

This repository provides an educational and reproducible implementation of the Enhanced Hiking Optimization Algorithm (EHOA) for feature selection. Its evaluation protocol prevents data leakage and reports predictive performance, feature reduction, convergence, and model interpretability.

### Presentation package

- [Full Persian report](docs/REPORT_FA.md)
- [Ready-to-present PowerPoint deck](docs/EHOA_Presentation_FA.pptx)
- [Presentation script and defense questions](docs/PRESENTATION_FA.md)
- [Paper-to-code traceability table](docs/ARTICLE_TO_CODE.md)
- [Automatically generated experiment report](results/REPORT.md)

Run the tests and classroom demo with one command:

```powershell
.\run_demo.ps1
```

### Implemented features

- Population initialization using ten chaotic maps, with **Tent** as the default;
- linear adaptive sweep factor, inertia weight, and PSO-like update following Equations 7 and 9;
- S-shaped binary transfer following Equations 10 and 11;
- multi-objective fitness with `alpha=0.99` and 5-NN following Equation 12;
- stratified 10-fold cross-validation in the paper profile;
- leakage-safe preprocessing: imputation, scaling, and SMOTE are fitted only on each training fold;
- held-out evaluation using KNN, Logistic Regression, SVM, and Random Forest;
- all-feature baselines, independent seeds, and runtime tracking;
- permutation importance and optional SHAP explanations;
- reproducible CSV, JSON, Markdown, and chart artifacts.

### Quick-profile results

The following values were calculated with `seed=42` on an untouched held-out test set:

| Dataset | Selected features | Reduction | KNN accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|
| Breast Cancer Wisconsin | 15 / 30 | 50.00% | 92.11% | 92.76% |
| Wine | 6 / 13 | 53.85% | 97.22% | 97.62% |

![Breast Cancer convergence](results/breast_cancer/convergence.png)

Complete classifier results are available at `results/<dataset>/classifier_metrics.csv`.

### Installation and usage

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py --profile quick --verbose
```

Install the optional SHAP dependencies:

```powershell
.venv\Scripts\python -m pip install -r requirements-explain.txt
```

The paper profile uses 30 hikers, 50 iterations, 10 folds, and 20 independent runs, so it is computationally expensive:

```powershell
.venv\Scripts\python main.py --profile paper --datasets breast_cancer --explain shap
```

Run the synthetic high-dimensional big-data demo:

```powershell
.venv\Scripts\python main.py --profile quick --datasets high_dimensional
```

All main parameters can also be overridden explicitly:

```powershell
.venv\Scripts\python main.py --hikers 12 --iterations 20 --folds 5 --runs 3 --seed 42
```

### Generated artifacts

Each `results/<dataset>/` directory contains:

- `runs.csv`: independent-seed results;
- `classifier_metrics.csv`: EHOA-selected versus all-feature comparison;
- `selected_features.csv`: selected feature names and indices;
- `convergence.csv/png`: fitness, accuracy, feature-count, and population-diversity history;
- `confusion_matrix_knn.png`: evaluation on the untouched test set;
- `feature_importance.csv/png`: measured feature importance;
- `classifier_comparison.png`: held-out classifier comparison;
- `REPORT.md`: automatically generated configuration and result summary.

### Scientific scope

This repository implements a valid experimental framework, but the `quick` profile is intended only for smoke testing and classroom demonstration. A full reproduction claim requires the paper's 33 datasets, 20 independent trials, and statistical comparisons against the reported competing algorithms.

### Tests

```powershell
.venv\Scripts\python -m pytest -q
```

## License and citation

The code is released under the [MIT License](LICENSE). For academic use, cite the original paper using the metadata provided in [`CITATION.cff`](CITATION.cff).

---

## فارسی — Appendix

این بخش ترجمهٔ فارسی و راهنمای کوتاه این مخزن است. برای نصب، اجرا و جزئیات کامل، ابتدا بخش انگلیسی را بخوانید و دستورهای همان بخش را اجرا کنید.
