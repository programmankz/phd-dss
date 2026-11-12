Положите сюда CSV с реальными данными (см. корневой README, раздел «Данные»).
Без них система работает на синтетическом демо-наборе.

Ставки безрисковых инструментов (Модуль интеграции, сквозной бэктест `scripts/end_to_end.py`):

* `nbk_base_rate.csv` — `effective_from,rate`: базовая ставка Национального Банка РК, % годовых, по датам вступления решений в силу; источник — https://nationalbank.kz/en/news/grafik-prinyatiya-resheniy-po-bazovoy-stavke (страницы по годам 2018–2026); обновление: `python -m dss.cli fetch rates`.
* `tb3ms.csv` — `Date,rate`: ставка 3-месячных казначейских векселей США, % годовых, месячные значения FRED (серия TB3MS, https://fred.stlouisfed.org/series/TB3MS), 2018-01 – 2026-09; обновление: `python -m dss.cli fetch rates`.

Корпоративные события KASE:

* `dividends_kase.csv` — `ticker,record_date,per_share,protocol_date,payment_start_date,period_start,period_end,is_paid`: история дивидендов акций KASE, как её публикует биржа (https://kase.kz/api/instruments/shares/<тикер>/dividends — список «Дивиденды → история выплат» на странице бумаги); ₸ на акцию, дата фиксации реестра; загружено 12.11.2026, выплаты с 1999 г.; обновление: `python -m dss.cli fetch kase-dividends [--tickers …]`. Оговорки: у части выплат KSPI флаг `is_paid=false` при объявленной выплате; две выплаты KZTK с датой фиксации 25.05.2025 (19 923,74 и 6 741,14 ₸) — это одна отсечка, в `corporate_actions.csv` они суммированы.
* `corporate_actions.csv` — дивидендные строки пересобраны 12.11.2026 из `dividends_kase.csv` (`python scripts/check_corporate_actions.py --write`): суммы и даты фиксации — по KASE, ex-дата — по правилу «реестр на 00:00 даты фиксации» (`loaders.kase_ex_date`: T+2 с 25.11–23.12.2015, T+0 раньше). Прежняя версия (коммит 2e98c78) ставила ex-дату на один торговый день позже для 46 из 59 выплат и не содержала 12 выплат KSPI 2021–2024 гг. и 2 выплат KMGZ 2023–2024 гг. Отчёт: `docs/corporate_actions_check.md`.
* Объяснённые дневные изменения скорректированных цен больше 25 % по модулю: KZTK 22.05.2025, +26,0 % (дивиденд 45,7 % цены; цена дошла до нового уровня за два дня: −31,5 % и −14,3 %).
