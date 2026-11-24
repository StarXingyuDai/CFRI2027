# README

Replication materials for **The Role of Chinese New-Energy Equity in Portfolio Optimization: A Multi-Horizon Analysis**.

## Step 1. Software and package requirements

The package was executed with Python 3.12 on Windows. The exact versions of NumPy, pandas, SciPy, statsmodels, Matplotlib and openpyxl are in `requirements.txt`. The source scripts use paths relative to this folder.

Open a terminal in this folder and run:

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -B Main_01.py

On macOS or Linux, use `.venv/bin/python` in the last two commands. An existing environment with the listed versions can run `python -B Main_01.py` directly. To preserve the delivered output files when rerunning the analysis, provide an alternate location:

    python -B Main_01.py --output ./reproduced

The complete run estimates all seven horizons and thirteen method–parameter settings for the two universes. The default run regenerates the outputs in `data/`, `tables/` and `figures/`. It leaves the two input workbooks in `data/raw/` intact. No internet connection, absolute path, random seed, or interactive editor is required once Python and the dependencies are available.

## Step 2. Input workbooks and data preparation

- `data/raw/Data_4_newpaper.xlsx`, sheet `Index Levels`, supplies the daily levels. `Data_Preparation_02.py` selects the Nanhua Commodity Index, CSI 300, ChinaBond Treasury and Corporate Bond Net Price indices, and the CSI CN Mainland New Energy Index (000941), stored under the workbook header `CSI Mainland New Energy Theme Index`. The workbook has no separate ticker row. The selected series is the empirical proxy for Chinese new-energy equity exposure. A join to East Money's 000941 daily quotes matched 4,013 dates from launch through the end of the sample, with a maximum absolute discrepancy of 0.016 index points. The analysis uses the workbook values.
- `data/raw/China_Treasury_Yield_1Y.xlsx`, sheet `Treasury Yield 1Y`, supplies daily observations of the annualized one-year Chinese Treasury yield in the `China 1-Year Treasury Yield (%)` column. These values are yield levels in annual percent. The most recent nonzero observation strictly before formation is used. Both portfolios use it to calculate excess returns. The CSV label Renewable identifies the expanded portfolio. Only the asset indices above receive portfolio weights. The source materials do not document vendor contracts or redistribution terms. Confirm the data providers and permissions before publicly posting either workbook or its derived series.
- The input copies use English file, worksheet and column names. Dates, numeric observations and missing entries match the supplied workbooks.
- `Data_Preparation_02.py` selects the last common valid date of each Friday-ending week from the index launch date. Its stand-alone entry point writes only `data/weekly_inputs.csv`. The full run reconstructs this data from the workbooks. The [China Securities Index factsheet](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000941factsheet.pdf) identifies 000941 and gives the launch date of 28 October 2009. [East Money historical quotes](https://quote.eastmoney.com/zs000941.html) were used solely for the identification check.

Raw workbook SHA-256 digests:

| Input | SHA-256 |
| --- | --- |
| `Data_4_newpaper.xlsx` | 4b11b204f50ef6b341f33c9cd19d5b4fcee18f3825d1b7d93c7b7d5c88b185e9 |
| `China_Treasury_Yield_1Y.xlsx` | 5d34ec72c90e16e34f5d18f5cb7f39045d79305739325a55b6b85ee30066d951 |

## Step 3. Main computation

`Main_01.py` runs three labelled regions:

1. Estimate long-only target weights using 52 completed return vectors before each formation. Holding paths are staggered by starting week, with one rebalance every h retained weekly marks. Compute realized gross return, risk-free return, turnover and costs.
2. Compute the matched performance differences, two-sided HAC comparisons with Holm adjustment for all 91 Sharpe differences, and each non-overlapping path's Sharpe and return-loss estimates.
3. Export the analysis tables, checks and numbered figures.

`Portfolio_Methods_03.py` contains 1/N, IV, MV, Fund2, ML and MLS. `Black_Litterman_04.py` provides the iterated VAR(2) view and the posterior moments. `Inference_05.py` contains HAC and Holm calculations, with `Jobson_Korkie_06.py` supplying the independent-normal benchmark for comparison. `DEA_07.py` fits the range directional frontiers. `Figures_08.py` renders all figures from the numerical outputs.

### Match the BL and DEA equations to the scripts

- Section 4.2. In `Black_Litterman_04.py`, `forecast_view` fits the VAR(2), iterates it to the holding endpoint and sums the innovation-covariance contributions using the companion matrix. `posterior_moments` forms the equal-weight prior and updates the mean and predictive covariance. `Main_01.py` sends these moments to the long-only mean–variance solver. `tables/diagnostics/var_status.csv` records the forecast choice.
- Section 4.5. `Main_01.py` pools the holding returns by horizon, universe and rule setting. `DEA_07.py` uses `Turnover` and `CostDrag_50` as inputs and `Sharpe` and `Sortino` as outputs. It solves a variable-returns-to-scale linear program for each unit and reports `theta` and `DEA_efficiency`, computed as one minus the optimal `theta`. `tables/dea_alternative.csv` gives the Sharpe-only results, `tables/dea_matched_differences.csv` subtracts matched benchmark scores from expanded scores, and `tables/dea_diagnostics.csv` reports peer counts and solver status.

### Calendar and units

- For an index level P, the simple holding-period return in percent is 100 × [P(t+h)/P(t) − 1]. Estimation, realized gross portfolio returns and portfolio statistics use these simple returns. Calculations in wealth units divide returns by 100. With decimal gross return r, per-side fee c and fractional turnover TO, the decimal net return is (1+r)(1−c×TO)−1.
- Equation (1) converts the one-year yield level y, observed in annual percent, into the simple reference return in percent for D actual calendar days: 100 × [(1+y/100)^(D/365)−1]. This uses the quoted annual yield as the annual compounding rate. A percentage change in the quoted yield does not measure the reference return. The code evaluates the power using log1p and expm1 to preserve precision. Gross excess return equals gross portfolio return less this common reference return.
- `data/formation_calendar.csv` records each holding's start, end, phase, training endpoints and calendar duration. Phase zero starts at the first available formation of the horizon.
- `data/portfolio_returns.csv` contains paired realized returns per horizon, method, risk-aversion setting and portfolio. Return columns ending in `_pct` are simple holding-period returns in percent, without annualization. Fee-drag columns ending in `_20` or `_50` are also in percentage points. Turnover is a fraction of the portfolio's initial value summing buys and sells. Its first value in each path is blank. The first holding has equal gross and net returns.
- `data/portfolio_weights.csv` reports target weights as fractions. The benchmark's weight in the proxy index is blank because that asset is outside its universe. `NA` in `gamma` indicates a rule without a risk-aversion parameter. When reading with pandas, use `keep_default_na=False` for this string where appropriate.
- The program verifies identical benchmark and expanded dates and reference returns, long-only unit-sum allocations, path boundaries, the net return identity and the fixed-risk gain identity. It saves realized VAR and optimizer diagnostics in `tables/diagnostics/` and records frontier exclusions and solver status in `tables/dea_diagnostics.csv`.

- Sections 5.1 and 5.2 use `tables/risk_adjusted_components.csv` to compare mean excess return, excess-return volatility, downside deviation, the fraction of negative excess returns, and the root mean square conditional on a negative excess return. Downside deviation equals the latter multiplied by the square root of the shortfall share.
- `tables/nonoverlapping_phase_results.csv` and `tables/nonoverlapping_phase_summary.csv` provide starting-week sensitivity checks described in Section 4.4. The latter gives path lengths, sign counts and medians at the 50 bp fee. Both are replication outputs.

## Step 4. Reproduce the tables

| Manuscript item | File under `tables/` | Selection |
| --- | --- | --- |
| Table 1 | `../data/raw/` and Section 3 | Four shared indices, the expanded-only proxy and the common excess-return reference |
| Table 2 | `table_02_sample.csv` | All horizons |
| Table 3 | `table_03_sharpe_inference.csv` | All seven horizons and 13 settings, paired Sharpe differences and two HAC lag choices with Holm adjustment |
| Table 4 | `table_04_economic_comparison.csv` | All seven horizons and 13 settings, gross gain, fee drag, net gain and return loss at both fees |
| Table 5 | `table_05_dea_summary.csv` | Method means and medians pooled over all seven horizons and, where relevant, all γ values |
| Table A1 | `table_A1_asset_statistics.csv` | All seven horizons and all five assets |

Holding-period Sharpe and Sortino ratios, DEA scores, skewness and excess kurtosis are unitless. Table 3 p-values are probabilities. Table 4 changes in returns and fee drag are percentage points per holding, without annualization. Table A1 return statistics are in percent per holding, without annualization. Table 2 reports counts, weeks and dates. The manuscript notes identify authors’ calculations and the inputs where applicable.

Other files report absolute portfolio performance, both sets of frontier scores, all individual starting-date estimates and all asset weight summaries. The main inference CSV reports pointwise intervals, both HAC bandwidths, each set of Holm-adjusted p-values, and the one- and two-sided JK p-values. The 50 bp and 20 bp fee assumptions represent per-side costs. Risk-matched return loss and the fixed-gross-volatility decomposition use different scale factors, as explained in the manuscript.

After running the main script, regenerate the full landscape LaTeX versions of Tables 3, 4 and A1:

    python -B Tables_09.py --output ./latex-tables

From the `CFRI2027` folder in this workspace, use `--output ../../LaTeX/Tables` to write directly to the manuscript project. Use `--tables` to read CSVs from an alternate run.

## Step 5. Reproduce the figures

`Figures_08.py` is called automatically by the main script. The `figures/` folder contains eight image panels and their plotting matrices. Manuscript Figure 1 combines panels 1 and 2 (Sharpe and Sortino). Figure 2 uses panel 3 (turnover). Figure 3 combines panels 4 and 5 (return loss at 50 and 20 bp). Figure 4 uses panel 6 (DEA). Figure 5 combines panels 7 and 8 (mean weight in the proxy index and frequency of positive positions). The plots use γ = 5 where applicable. Panel 3 (`figure_03_data.csv`, manuscript Figure 2) stores turnover differences as fractions. Its image multiplies them by 100 to show percentage points per rebalance. The remaining matrices use the displayed units. Sharpe and Sortino are unitless and use holding-period returns without annualization. Return loss is in percentage points per holding, without annualization. DEA differences are in score units. Weight and positive-position frequency are percentages at formation. The stand-alone panels identify authors’ calculations as their source.

