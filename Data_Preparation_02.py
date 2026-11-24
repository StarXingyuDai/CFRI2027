"""Read the supplied daily inputs and construct synchronized weekly marks."""
from pathlib import Path

import pandas as pd


DATA = Path(__file__).resolve().parent / "data"
PRICE_FILE = "Data_4_newpaper.xlsx"
YIELD_FILE = "China_Treasury_Yield_1Y.xlsx"
INDEX_LAUNCH = pd.Timestamp("2009-10-28")  # CSI CN Mainland New Energy Index (000941)
PRICE_COLUMNS = {
    "Date": "date",
    "Nanhua Commodity Index": "commodity_nanhua",
    "CSI 300": "stock_csi300",
    "ChinaBond Treasury Bond Net Price Index (Aggregate)": "bond_treasury",
    "ChinaBond Corporate Bond Net Price Index (Aggregate)": "bond_corporate",
    "CSI Mainland New Energy Theme Index": "new_energy",
}
ASSETS = list(PRICE_COLUMNS.values())[1:]


def weekly_inputs(data_dir=DATA):
    # region 1. Read daily index levels and Treasury yields
    prices = pd.read_excel(data_dir / "raw" / PRICE_FILE, sheet_name="Index Levels")
    prices.columns = prices.columns.str.strip()
    prices = prices.rename(columns=PRICE_COLUMNS)[["date", *ASSETS]]
    prices["date"] = pd.to_datetime(prices["date"])
    for asset in ASSETS:
        prices[asset] = pd.to_numeric(prices[asset], errors="coerce")
    if prices["date"].duplicated().any():
        raise ValueError("Duplicate dates in the price input.")
    yields = pd.read_excel(data_dir / "raw" / YIELD_FILE, sheet_name="Treasury Yield 1Y")
    yields.columns = yields.columns.str.strip()
    yields = yields.rename(columns={"Date": "yield_date", "China 1-Year Treasury Yield (%)": "yield_1y_pct"})
    yields["yield_date"] = pd.to_datetime(yields["yield_date"])
    yields["yield_1y_pct"] = pd.to_numeric(yields["yield_1y_pct"], errors="coerce")
    yields = yields[yields["yield_1y_pct"].notna() & yields["yield_1y_pct"].ne(0)]
    yields = yields[["yield_date", "yield_1y_pct"]].sort_values("yield_date")
    # endregion

    # region 2. Last common trading mark per Friday-ending week
    eligible = prices[prices["date"].ge(INDEX_LAUNCH) & prices[ASSETS].gt(0).all(axis=1)]
    weekly = eligible.sort_values("date").groupby(pd.Grouper(key="date", freq="W-FRI")).tail(1)
    weekly = pd.merge_asof(weekly, yields, left_on="date", right_on="yield_date", allow_exact_matches=False)
    if weekly["yield_1y_pct"].isna().any() or not weekly["date"].is_monotonic_increasing:
        raise ValueError("Weekly prices or formation yields are misaligned.")
    return weekly
    # endregion


if __name__ == "__main__":
    weekly_inputs().to_csv(DATA / "weekly_inputs.csv", index=False)

