"""Default parameters of the DSS.

Every constant below is traceable to one of the four dissertation articles:

* [A1] Lamasheva et al., "GDP Growth Forecasting Using Machine Learning: Global
  Evidence with a Kazakhstan Case Study", IJIES 19(6), 2026.          -> Module I
* [A2] Lamasheva et al., "Exchange rate dynamics of USD-KZT, EUR-KZT, RUB-KZT
  and CNY-KZT (2000-2024)", Vestnik NIA RK, 2025, No 2 (96).           -> Module II
* [A3] Durbenov et al., "Clustering in compiling investment portfolios",
  Vestnik KazATC, 2024, No 3 (132).                                    -> Module III
* [A4] Iskakov et al., "Machine Learning-Based Portfolio Optimization Using
  Cross-Sectional Signals: Evidence from the NSE Market", ETASR 16(4), 2026.
                                                                        -> Module IV
"""
from __future__ import annotations

from dataclasses import dataclass, field

RANDOM_SEED = 42  # [A1] Appendix A.3

# --------------------------------------------------------------------------
# Module I - macro forecasting  [A1]
# --------------------------------------------------------------------------
MACRO_TRAIN_END_YEAR = 2014          # train <= 2014, test 2015-2024
MACRO_FORECAST_HORIZON = 5           # 2025-2029
TARGET_COUNTRY = "Kazakhstan"

# Scenario adjustments to the ARIMA baseline, p.p. per forecast year (Table 9).
# NB: Table 9 is used rather than the text of Sec. 5.3.2, because the text
# ("narrowing to -1.7 p.p. by 2028-2029") does not match the published table.
SCENARIOS = {
    "base":   {"label": "Base",            "offsets": [0.0, 0.0, 0.0, 0.0, 0.0], "brent": "75-80"},
    "bull":   {"label": "Bull (нефтяной бум)",     "offsets": [0.86, 0.90, 0.90, 0.90, 0.90], "brent": "100-110"},
    "bear":   {"label": "Bear (падение нефти)",    "offsets": [-1.10, -1.10, -1.10, -1.10, -1.10], "brent": "50-55"},
    "stress": {"label": "Stress (нефть + геополитика)", "offsets": [-3.30, -3.30, -2.90, -2.80, -2.70], "brent": "< 50 + шок"},
}

# One-at-a-time sensitivity of the 2025 forecast (Table 10). Each factor is a
# piecewise-linear curve through (baseline, 0) and the two published endpoints.
SENSITIVITY = {
    "brent":    {"label": "Цена Brent, $/барр.", "baseline": 75.0, "points": [(45.0, -2.4), (75.0, 0.0), (110.0, 1.8)]},
    "cpc":      {"label": "Сбой КТК, мес.", "baseline": 0.0, "points": [(0.0, 0.0), (3.0, -0.6), (12.0, -2.3)]},
    "rate":     {"label": "Реальная ставка, +б.п.", "baseline": 0.0, "points": [(0.0, 0.0), (100.0, -0.4), (300.0, -1.2)]},
    "china":    {"label": "Рост ВВП Китая, %", "baseline": 4.5, "points": [(2.0, -0.5), (4.5, 0.0), (6.0, 0.4)]},
    "usdkzt":   {"label": "Ослабление тенге, %", "baseline": 0.0, "points": [(0.0, 0.0), (10.0, -0.2), (30.0, -0.7)]},
}

# IMF WEO October 2024 benchmark for Kazakhstan used in [A1] Table 8.
IMF_WEO_KZ_BENCHMARK = {2025: 4.8, 2026: 4.5, 2027: 4.6, 2028: 4.7, 2029: 4.8}

# --------------------------------------------------------------------------
# Module II - FX risk  [A2]
# --------------------------------------------------------------------------
CURRENCIES = ["USD", "EUR", "CNY", "RUB"]
FX_RISK_WEIGHTS = {"USD": 0.45, "EUR": 0.20, "CNY": 0.15, "RUB": 0.20}
FX_VOL_WINDOW = 30      # days
FX_MOMENTUM_WINDOW = 60  # days

# --------------------------------------------------------------------------
# Module IV - signals and optimisation  [A4]
# --------------------------------------------------------------------------
LOOKBACK = 60            # sliding window, trading days
SMOOTHING = 10           # average of the last 10 predictions
COV_WINDOW = 252         # rolling covariance window, annualised x252
TRAIN_SHARE = 0.80       # chronological 80/20 split ("paper" protocol, [A4])
WF_INITIAL_TRAIN = 0.50  # walk-forward: first training window, share of dates
WF_RETRAIN_EVERY = 126   # walk-forward: retrain every 126 trading days (half a year)

# Evaluation protocols of Module IV (FR-IV-04): (train share, retrain every N days; 0 = never)
PROTOCOLS = {
    "walk_forward": {"label": "Walk-forward (переобучение раз в 126 дн.)",
                     "train_share": WF_INITIAL_TRAIN, "retrain_every": WF_RETRAIN_EVERY},
    "paper":        {"label": "Как в статье [A4]: одно разбиение 80/20",
                     "train_share": TRAIN_SHARE, "retrain_every": 0},
}
# One-click reproduction of the [A4] experiment (scenario 6 of the spec)
PAPER_PRESET = {"protocol": "paper", "cov": "sample", "rebalance": "D", "tc": "0", "caps": "0"}
RISK_FREE = 0.05         # reproduces the Sharpe ratios of [A4] Table IV

# Risk-free rate per universe (FR-IV-12). NSE: 5 %, inferred from [A4] Table IV.
# KASE: the NBK base rate is to be used, but no value with a source is in the repository yet,
# so the entry is a PLACEHOLDER (None) that falls back to RISK_FREE and is flagged in the UI.
# Replace None with the rate as a fraction (e.g. 0.15 for 15 %) once the author provides it.
RISK_FREE_BY_UNIVERSE = {"nse": RISK_FREE, "kase": None}


def risk_free_for(universe: str) -> tuple[float, str]:
    """(annual risk-free rate, Russian note on its origin) for a universe."""
    rate = RISK_FREE_BY_UNIVERSE.get(universe)
    if rate is not None:
        return float(rate), ("5 % — восстановлена по табл. IV [A4]" if universe == "nse" and rate == RISK_FREE
                             else f"задана в config.RISK_FREE_BY_UNIVERSE[{universe!r}]")
    if universe in RISK_FREE_BY_UNIVERSE:
        return RISK_FREE, ("заглушка: ставка НБ РК для этого набора ещё не задана "
                           "(config.RISK_FREE_BY_UNIVERSE), временно используется 5 %")
    return RISK_FREE, "5 % по умолчанию (config.RISK_FREE)"


@dataclass(frozen=True)
class InvestorProfile:
    """Investor profile used by the integration module."""

    key: str
    label: str
    base_risky_share: float   # share of risky assets in normal conditions
    max_asset_weight: float   # per-asset cap (fixes the 48.8 % concentration of [A4])
    cluster_cap: float        # max total weight of one Module III cluster
    min_risky_share: float = 0.10
    max_risky_share: float = 0.95


PROFILES = {
    "conservative": InvestorProfile("conservative", "Консервативный", 0.40, 0.15, 0.40),
    "moderate":     InvestorProfile("moderate", "Умеренный", 0.60, 0.20, 0.50),
    "aggressive":   InvestorProfile("aggressive", "Агрессивный", 0.80, 0.30, 0.65),
}

# --------------------------------------------------------------------------
# Integration module - decision rules (Sec. 10 of docs/techspecs.md)
# --------------------------------------------------------------------------
# Scenario and cycle-phase multipliers of the risk budget.
SCENARIO_RISK_MULT = {"base": 1.00, "bull": 1.10, "bear": 0.85, "stress": 0.60}
PHASE_RISK_MULT = {"Расширение": 1.05, "Восстановление": 1.00, "Замедление": 0.92, "Спад": 0.75}


@dataclass(frozen=True)
class IntegrationRules:
    """Coefficients of the integration rules; their sensitivity is analysed by
    :func:`dss.integration.sensitivity_analysis`.

    Rule 1: d = clip(d0 (1 - rec_penalty P_rec) k_scen k_phase - neg_growth_cut 1[g < 0], d_min, d_max)
    Rule 2: h = clip(hedge_base + hedge_slope R_FX, 0, hedge_max)
    """

    rec_penalty: float = 0.60       # risky share falls by 60 % of P(recession)
    neg_growth_cut: float = 0.05    # extra cut when the growth forecast is negative
    hedge_base: float = 0.15        # USD share of the safe part in a calm FX regime
    hedge_slope: float = 0.60       # extra USD share per unit of the FX risk index
    hedge_max: float = 0.80
    fx_risk_warn: float = 0.66      # FX risk index from which a warning is issued ("high" regime)
    low_growth_warn: float = 1.0    # growth forecast, %, below which a warning is issued
    scenario_mult: dict = field(default_factory=lambda: dict(SCENARIO_RISK_MULT))
    phase_mult: dict = field(default_factory=lambda: dict(PHASE_RISK_MULT))


RULES = IntegrationRules()


@dataclass
class Settings:
    """User-facing settings of a DSS run (one decision)."""

    profile: str = "moderate"
    scenario: str = "base"
    universe: str = "kase"               # "kase" | "nse" | "custom"
    models: tuple = ("ridge", "rf", "gb")  # Module IV ensemble
    rebalance: str = "W"                 # D | W | M
    tc_bps: float = 10.0                 # transaction costs, basis points
    cov_method: str = "ledoit_wolf"      # "sample" reproduces [A4]
    use_fuzzy_caps: bool = False
    risk_free: float | None = None       # None: by universe, see risk_free_for()
    protocol: str = "walk_forward"       # Module IV evaluation: "walk_forward" | "paper" (80/20)
    backtest_caps: bool = True           # False: only 0 <= w_i <= 1 in the backtest, as in [A4]
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.risk_free is None:
            self.risk_free = risk_free_for(self.universe)[0]

    @property
    def risk_free_note(self) -> str:
        rate, note = risk_free_for(self.universe)
        return note if abs(rate - self.risk_free) < 1e-12 else "задана вручную"

    def key(self) -> tuple:
        return (self.profile, self.scenario, self.universe, tuple(self.models),
                self.rebalance, float(self.tc_bps), self.cov_method,
                bool(self.use_fuzzy_caps), float(self.risk_free), self.protocol, bool(self.backtest_caps))
