# СППР управления инвестиционным портфелем

Программная реализация диссертационного исследования **Дурбенова Р.М.** «Система поддержки принятия решений при управлении инвестиционным портфелем» (PhD, 8D06103 «Информационные системы», ЕНУ им. Л.Н. Гумилёва).

*Decision support system for investment portfolio management: a four-module ML pipeline (macro forecasting, tenge FX risk, asset clustering, signal-based portfolio optimisation) with an integration module and a Russian-language web interface.*

## Что делает система

Сигналы четырёх модулей последовательно агрегируются в одно решение о составе портфеля:

| Модуль | Методы | Выход → куда передаётся |
|---|---|---|
| **I. Макропрогноз** [A1] | Logistic Regression, Random Forest, Gradient Boosting, SVM (RBF) — рецессии; Linear, Ridge, RF, GB — рост ВВП; K-means и DTW-кластеризация стран; ARIMA (отбор по AIC), Prophet-style, LSTM-style для Казахстана; сценарии и чувствительность | P(рецессии), прогноз роста, фаза цикла → **бюджет риска** |
| **II. Валютные риски** [A2] | Описательная статистика, корреляции Пирсона, линейный тренд и R², детектор девальвационных шоков, индекс валютного риска | Индекс риска, ожидаемое ослабление тенге → **USD-хедж безрисковой части** |
| **III. Кластеризация** [A3] | K-means, K-medoids (PAM), Fuzzy C-means; силуэт, Дэвис–Боулдин; бэктест кластерных портфелей | Метки кластеров, матрица U → **лимиты диверсификации** |
| **IV. Оптимизация** [A4] | 11 прогнозных моделей (+MLP), walk-forward, сглаживание 10, ранги sᵢ = Rank(r̂ᵢ)/N, max wᵀs/√(wᵀΣw) (SLSQP), бэктест с издержками | **Веса** внутри рисковой части |
| **Интеграция** | Правила агрегации (см. `dss/integration.py`) | Рекомендация, веса, VaR, объяснение каждого шага |

## Быстрый старт

```bash
git clone https://github.com/programmankz/phd-dss.git
cd phd-dss
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m dss.cli warmup        # первый расчёт всех модулей (~1 мин), дальше — из кэша
python -m webapp.app            # веб-интерфейс: http://127.0.0.1:8050
```

Без файлов данных система работает в **демо-режиме** на синтетических данных: это видно по жёлтой плашке на каждой странице. Результаты демо-режима не являются эмпирическими выводами.

Другие команды:

```bash
python -m dss.cli decide --profile conservative --scenario stress   # рекомендация в консоли
python -m dss.cli report -o report.html                              # автономный HTML-отчёт
python tests/run.py                                                  # тесты (или: pytest -q)
```

## Данные

Положите CSV в папку `data/` (или укажите другую через `DSS_DATA_DIR`) и нажмите в интерфейсе «Данные → Сбросить кэш».

| Файл | Формат | Источник | Модуль |
|---|---|---|---|
| `gdp_growth.csv` | `country,1980,…,2024` (рост реального ВВП, %) | `python -m dss.cli fetch gdp` (IMF WEO DataMapper) или ваш `world_gdp_data.csv` | I |
| `fx_rates.csv` | `Date,USD,EUR,CNY,RUB` (тенге за единицу) | `python -m dss.cli fetch fx` (НБ РК, с докачкой) | II |
| `prices_kase.csv` | `Date,HSBK,KSPI,…` (цены закрытия) | экспорт с kase.kz | III, IV |
| `fundamentals_kase.csv` | `ticker,sector,PB,PE` | kase.kz / отчётность эмитентов | III |
| `prices_nse.csv` | `Date,HDFCBANK.NS,…` | `python -m dss.cli fetch nse` (нужен `yfinance`) | III, IV |

## Структура

```
dss/
  config.py        параметры из статей [A1]–[A4], профили инвестора, сценарии
  data/            загрузчики (CSV, IMF, НБ РК, Yahoo) и синтетический демо-набор
  macro.py         Модуль I
  fx.py            Модуль II
  clustering.py    Модуль III (PAM, FCM, DTW реализованы на numpy)
  signals.py       Модуль IV, этап A (прогнозы → ранговые сигналы)
  portfolio.py     Модуль IV, этап B (оптимизация, бэктест)
  integration.py   модуль интеграции (бюджет риска, хедж, лимиты, веса)
  pipeline.py      оркестрация и кэш
  plots.py, report.py, cli.py
webapp/            Flask-интерфейс (6 страниц)
tests/             13 тестов научных инвариантов (нет утечки, ограничения, табл. 9–10)
docs/methodology.md  соответствие кода статьям и известные расхождения
scripts/make_screenshots.py  скриншоты интерфейса для презентации
```

## Опциональные зависимости

* `statsmodels` — точная ML-оценка ARIMA (без него — встроенная оценка методом условных наименьших квадратов);
* `tensorflow` — модели RNN, LSTM, CNN-LSTM, LSTM-MLP (набор «11 моделей статьи»);
* `yfinance` — загрузка цен NSE.

## Публикации

* [A1] Lamasheva Zh., Lakhno V., Nazyrova A., Karlygash A., Durbenov R. GDP Growth Forecasting Using Machine Learning: Global Evidence with a Kazakhstan Case Study // IJIES. 2026. Vol. 19, No. 6. P. 626–648. DOI: 10.22266/ijies2026.0630.33
* [A2] Ламашева Ж.Б., Шекербек А.А., Баегизова А.С., Дурбенов Р.М., Лахно В., Искаков Е.К. Exchange rate dynamics of USD-KZT, EUR-KZT, RUB-KZT and CNY-KZT (2000–2024) // Вестник НИА РК. 2025. № 2 (96). DOI: 10.47533/2025.1606-146X.2-07
* [A3] Durbenov R.M., Lamasheva Zh., Lakhno V., Zhartybayeva M. Clustering in compiling investment portfolios // Вестник КазАТК. 2024. № 3 (132). С. 206–215. DOI: 10.52167/1609-1817-2024-132-3-206-215
* [A4] Iskakov Y., Shekerbek A., Nazyrova A., Lamasheva Zh., Durbenov R. Machine Learning-Based Portfolio Optimization Using Cross-Sectional Signals: Evidence from the NSE Market // ETASR. 2026. Vol. 16, No. 4. P. 37880–37888. DOI: 10.48084/etasr.19240

Система носит исследовательский характер и не является индивидуальной инвестиционной рекомендацией.
