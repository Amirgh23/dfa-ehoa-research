# گزارش اجرای EHOA

این گزارش به‌صورت خودکار از artifactهای همین اجرا تولید شده است؛ بنابراین اعداد آن
با فایل‌های CSV پوشه‌ی نتایج یکسان‌اند.

## تنظیمات

- Profile: `quick`
- Chaotic map: `tent`
- Population / iterations: `8` / `8`
- CV folds / independent runs: `5` / `1`
- Fitness alpha: `0.99`
- Random seed: `42`
- SMOTE: `training folds only`

## خلاصه نتایج

| Dataset | Selected/total | Reduction | CV accuracy | Test accuracy | Test balanced accuracy |
|---|---:|---:|---:|---:|---:|
| breast_cancer | 15/30 | 50.00% | 0.9714 | 0.9211 | 0.9276 |
| wine | 6/13 | 53.85% | 0.9786 | 0.9722 | 0.9762 |

## تفسیر صحیح

`CV accuracy` فقط روی داده‌ی آموزش و برای بهینه‌سازی subset محاسبه شده است.
`Test accuracy` و `Test balanced accuracy` روی hold-out دست‌نخورده محاسبه شده‌اند
و معیار اصلی تعمیم هستند. پروفایل quick برای دموی کد است؛ ادعای بازتولید مقاله
به پروفایل paper، ۲۰ اجرای مستقل و مجموعه‌داده‌های اصلی نیاز دارد.

برای جزئیات هر classifier، ویژگی‌های منتخب و نمودارها به زیرپوشه‌ی هر dataset
مراجعه کنید.
