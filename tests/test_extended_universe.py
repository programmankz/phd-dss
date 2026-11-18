"""Extended universe: AIX feed parsing and tenge conversion of the new price files."""
import pandas as pd

from dss.data import loaders


def test_parse_aix_history_keeps_days_with_deals_in_date_order():
    recs = [{"date": "2024-01-04T00:00:00Z", "close": 11.0, "closingPrice": 11.0, "value": 2200.0, "volume": 200},
            {"date": "2024-01-03T00:00:00Z", "close": 10.5, "closingPrice": 10.5, "value": 0, "volume": 0},
            {"date": "2024-01-02T00:00:00Z", "close": 10.0, "closingPrice": 10.0, "value": 1000.0, "volume": 100}]
    df = loaders.parse_aix_history(recs)
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")]
    assert df["close"].tolist() == [10.0, 11.0] and df["value"].tolist() == [1000.0, 2200.0]


def test_load_extended_prices_converts_aix_to_tenge_and_removes_dividends(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "data_dir", lambda: tmp_path)
    (tmp_path / "prices_kase_new.csv").write_text("Date,AAA\n2024-01-03,100\n2024-01-04,90\n2024-01-05,91\n")
    (tmp_path / "corporate_actions_new.csv").write_text("ticker,ex_date,method,factor,note\nAAA,2024-01-04,dividend,10,test\n")
    (tmp_path / "prices_aix.csv").write_text("Date,CORE\n2024-01-03,2\n2024-01-06,3\n2024-01-09,4\n")
    usd = pd.Series([450.0, 460.0, 470.0], index=pd.to_datetime(["2024-01-03", "2024-01-05", "2024-01-08"]))
    px = loaders.load_extended_prices(usd)
    assert px.loc["2024-01-03", "AAA"] == 90.0                 # dividend removed from the earlier close
    assert px.loc["2024-01-03", "CORE"] == 900.0 and px.loc["2024-01-06", "CORE"] == 3 * 460.0
    assert pd.Timestamp("2024-01-09") not in px.index          # no rate known for that day
