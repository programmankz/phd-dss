Положите сюда CSV с реальными данными (см. корневой README, раздел «Данные»).
Без них система работает на синтетическом демо-наборе.

Ставки безрисковых инструментов (Модуль интеграции, сквозной бэктест `scripts/end_to_end.py`):

* `nbk_base_rate.csv` — `effective_from,rate`: базовая ставка Национального Банка РК, % годовых, по датам вступления решений в силу; источник — https://nationalbank.kz/en/news/grafik-prinyatiya-resheniy-po-bazovoy-stavke (страницы по годам 2018–2026); обновление: `python -m dss.cli fetch rates`.
* `tb3ms.csv` — `Date,rate`: ставка 3-месячных казначейских векселей США, % годовых, месячные значения FRED (серия TB3MS, https://fred.stlouisfed.org/series/TB3MS), 2018-01 – 2026-09; обновление: `python -m dss.cli fetch rates`.
