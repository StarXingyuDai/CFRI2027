"""Export complete manuscript tables from the CSV files written by Main_01.py."""
from argparse import ArgumentParser
import csv
from dataclasses import dataclass
from pathlib import Path


HORIZONS = (1, 2, 4, 7, 13, 26, 52)
SETTINGS = (
    ("1/N", "NA"), ("IV", "NA"),
    *((rule, str(gamma)) for rule in ("MV", "BL", "Fund2") for gamma in (2, 5, 10)),
    ("ML", "NA"), ("MLS", "NA"),
)
ASSETS = {
    "commodity_nanhua": "Commodity",
    "stock_csi300": "Stock",
    "bond_treasury": "Treasury bond",
    "bond_corporate": "Corporate bond",
    "new_energy": "New-energy equity",
}
FEES = (20, 50)


@dataclass(frozen=True)
class ManuscriptTable:
    number: str
    title: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    note: str

    @property
    def filename(self):
        return f"table{self.number}.tex"


def rows_by_key(path, keys, order):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {tuple(row[key] for key in keys): row for row in rows}
    if len(index) != len(rows) or set(index) != set(order):
        raise ValueError(f"{path.name} does not contain the complete, unique manuscript grid.")
    return [index[key] for key in order]


def fixed(value, decimal_places=3):
    number = float(value)
    if abs(number) < .5 * 10 ** -decimal_places:
        number = 0
    return f"{number:.{decimal_places}f}"


def fmt_p(value):
    return "<0.001" if float(value) < .0005 else fixed(value)


def design(row):
    gamma = "—" if row["gamma"] == "NA" else row["gamma"]
    return row["horizon"], row["method"], gamma


def sharpe_row(row):
    return (*design(row), fixed(row["sharpe_difference"]),
            f'[{fixed(row["ci95_low"])}, {fixed(row["ci95_high"])}]',
            *(fmt_p(row[col]) for col in ("hac_p_two_sided", "holm_p", "hac_alt_p_two_sided", "holm_alt_p")))


def cost_row(row):
    fees = tuple(fixed(row[key]) for fee in FEES for key in (
        f"fixed_risk_extra_cost_{fee}_pct", f"fixed_risk_net_gain_{fee}_pct", f"return_loss_{fee}_pct"))
    return (*design(row), fixed(row["fixed_risk_gross_gain_pct"]), *fees)


def asset_row(row):
    return (ASSETS[row["asset"]], row["horizon"], row["n"],
            *(fixed(row[key], 2) for key in (
                "mean_pct", "sd_pct", "min_pct", "max_pct", "skewness", "excess_kurtosis")))


def complete_tables(csv_dir):
    csv_dir = Path(csv_dir)
    grid = [(str(h), rule, gamma) for h in HORIZONS for rule, gamma in SETTINGS]
    assets = [(str(h), asset) for h in HORIZONS for asset in ASSETS]
    keys = ("horizon", "method", "gamma")
    first = ("Horizon (weeks)", "Rule", "γ")
    return (
        ManuscriptTable(
            "3", "Matched Sharpe comparisons across holding horizons",
            (*first, "Δ Sharpe", "95% pointwise interval", "Baseline HAC p", "Baseline Holm p",
             "Longer-lag HAC p", "Longer-lag Holm p"),
            tuple(map(sharpe_row, rows_by_key(csv_dir/"table_03_sharpe_inference.csv", keys, grid))),
            "Sharpe differences and interval bounds are unitless and use holding-period ratios without annualization. "
            "HAC p-values are two-sided probabilities. The interval uses the baseline lag. Holm adjustment uses all "
            "91 comparisons separately for each lag. A dash denotes a rule without a risk-aversion parameter. "
            "Source. Authors’ calculations."
        ),
        ManuscriptTable(
            "4", "Gross return, fee drag and return loss across holding horizons",
            (*first, "Gross gain G", *(f"{label}, {fee} bp" for fee in FEES
                                      for label in ("C", "G−C", "Loss"))),
            tuple(map(cost_row, rows_by_key(csv_dir/"table_04_economic_comparison.csv", keys, grid))),
            "All gain, drag and loss estimates are in percentage points per holding period, without annualization. "
            "G compares gross returns at a fixed "
            "gross-volatility scale. C is incremental fee drag at that scale. Loss denotes return loss at matched net volatility. "
            "Negative return loss favors the expanded portfolio. Fees are in basis points per side. Source. Authors’ calculations."
        ),
        ManuscriptTable(
            "A1", "Simple asset returns across holding horizons",
            ("Index", "Horizon (weeks)", "N", "Mean", "SD", "Min", "Max", "Skewness", "Excess kurtosis"),
            tuple(map(asset_row, rows_by_key(csv_dir/"table_A1_asset_statistics.csv", ("horizon","asset"), assets))),
            "Mean, SD, Min and Max are in percent per holding period, without annualization. N is a count. "
            "Skewness and excess kurtosis are unitless. Returns overlap beyond one week. "
            "Source. Authors’ calculations from the common weekly index levels in Table 1."
        ),
    )


def encode(text):
    substitutions = {"%": r"\%", "γ": r"\(\gamma\)", "Δ": r"\(\Delta\)",
                     "−": r"\ensuremath{-}", "—": "---", "–": "--"}
    return "".join(substitutions.get(c, c) for c in text)


def row_line(values):
    return " & ".join(map(encode, values)) + r" \\"


def to_latex(table):
    cols = "rl" + "r" * (len(table.headers)-2)
    if table.number == "A1":
        cols = "l" + "r" * (len(table.headers)-1)
    head = row_line(tuple(r"\textbf{" + item + "}" for item in table.headers))
    caption = encode(table.title)
    repeat = rf"\multicolumn{{{len(table.headers)}}}{{l}}{{\textbf{{Table \thetable\ (continued)}}}} " + r"\\"
    appendix = table.number == "A1"
    heading = (r"\section*{Appendix. Supplementary descriptive statistics}",) if appendix else ()
    size = (
        r"\fontsize{9.5}{11}\selectfont\renewcommand{\arraystretch}{0.94}",
        r"\setlength{\LTpre}{3pt}\setlength{\LTpost}{0pt}",
    ) if appendix else ()
    return "\n".join((
        "% Generated by Tables_09.py from the full replication CSVs.",
        r"\FloatBarrier",
        r"\begin{landscape}",
        *heading,
        r"\begingroup",
        r"\tablefont",
        *size,
        r"\begin{longtable}{@{}" + cols + "@{}}",
        r"\caption{" + caption + r"}\label{tab:" + table.number + "}" + r" \\",
        r"\toprule", head, r"\midrule", r"\endfirsthead",
        repeat, r"\toprule", head, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot", r"\bottomrule", r"\endlastfoot",
        *map(row_line, table.rows),
        r"\end{longtable}",
        r"\tablenotes{Notes. " + encode(table.note) + "}",
        r"\endgroup", r"\end{landscape}", "",
    ))


def write_tables(csv_dir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for table in complete_tables(csv_dir):
        (output/table.filename).write_text(to_latex(table), encoding="utf-8", newline="\n")
        print(f"{table.filename}: {len(table.rows)} rows")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=Path(__file__).resolve().parent/"tables",
                        help="Directory containing the CSV outputs from Main_01.py.")
    parser.add_argument("--output", required=True, type=Path, help="LaTeX Tables directory to write.")
    opts = parser.parse_args()
    write_tables(opts.tables, opts.output)
