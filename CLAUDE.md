# Project: prediction_intervals_arena

Guidance for Claude Code (and humans) working in this repo. Read before writing code.

---

## Identity

A head-to-head comparison of **five ways to put an honest uncertainty range on a
regression prediction**, judged by the one metric that matters: **coverage** — does the
90% interval actually contain the truth 90% of the time, on held-out data, measured
against known truth.

- **Format:** GitHub repo — 6 Jupyter notebooks + `src/` library + lightweight Streamlit app.
- **Audience:** data scientists who ship regression models that output a single number and
  want to attach an honest range; anyone who reports intervals but never checks they cover.
- **Companion to** the [@Dima806](https://github.com/Dima806) projects (verified to exist):
  [`calibration_arena`](https://github.com/Dima806/calibration_arena) (classification twin),
  [`bootstrap_101`](https://github.com/Dima806/bootstrap_101) (bootstrap is one of the five methods),
  [`forecasting_arena`](https://github.com/Dima806/forecasting_arena) (intervals on the time axis),
  [`threshold_lab`](https://github.com/Dima806/threshold_lab) (decisions that consume the interval),
  [`bayesian_stats_101`](https://github.com/Dima806/bayesian_stats_101) (the Bayesian route to uncertainty).
  Note: the PRD also named `imputation_arena`, which does **not** exist on the account — do not link it.

### The thesis (keep this front of mind; everything serves it)

The textbook normal-theory interval quietly assumes **constant, symmetric, normal** errors.
Real data breaks all three. Under heteroscedasticity (error size grows with the prediction) its
single constant width fails — and the failure is mostly **conditional**, not overall. Measured
(nominal 0.90): overall coverage dips only modestly to **~0.88**, but the conditional-coverage
plot collapses to **~0.66 in the volatile region** while ballooning to **~1.0 in the calm
region** — too narrow and too wide in exactly the wrong places. (Do **not** claim "~70% overall";
the ~66% figure is the *volatile-region* coverage. This was corrected after measurement.)
**Conformal prediction** hits ~0.90 on every structure, wraps **any** model with **no
retraining**, and carries a distribution-free **guarantee** — and almost nobody uses it. The gap
is the difference between false confidence and honest uncertainty.

---

## Stack & constraints

- **Python 3.11+.** Package manager is **`uv`** — never `pip`, never edit `uv.lock` by hand.
- **Runtime target: 2-CPU / 8 GB GitHub Codespace, no GPU.** Everything must run there, each
  notebook in **< 5 minutes** (budget). In practice they are fast: all six run in ~50s total via
  `make notebooks`; the heaviest (the arena, `05`) is ~18s, so no result caching is needed yet.
  If a cell ever approaches the budget, shrink `n_samples`/`n_bootstrap` before adding caching.
- **Runtime deps:** scikit-learn, numpy, scipy, pandas, matplotlib, plotly, streamlit,
  pydantic, pydantic-settings, pyyaml.
- **Dev deps:** pytest, ruff, ty, jupyter, ipykernel, nbconvert, nbclient.
- Datasets are tens of thousands of rows — memory is never a concern.

---

## Repository structure

```
config/settings.yaml            # Coverage levels, noise structures, seeds — single source of truth
src/
  config.py                     # pydantic-settings loader for settings.yaml
  intervals/
    normal.py                   # Normal-theory interval (from scratch) — the cautionary protagonist
    quantile_regression.py      # Pinball-loss quantile regression (from scratch)
    gbm_quantile.py             # Native gradient-boosting quantile regressors (lower/upper models)
    bootstrap.py                # Bootstrap prediction intervals
    conformal.py                # Split + adaptive (Mondrian) conformal (from scratch)
  datasets/
    synthetic.py                # homoscedastic / heteroscedastic / skewed / heavy-tailed; KNOWN TRUTH
                                # (a real dataset e.g. California housing is PLANNED, not yet built)
  evaluation/
    coverage.py                 # empirical coverage vs nominal, overall and by region
    width.py                    # interval width / efficiency (tighter wins at equal coverage)
    conditional.py              # coverage conditioned on the prediction range
    comparison.py               # the arena harness (run_arena, coverage_report)
  visualisation.py              # matplotlib helpers (return figures, never show)
  artifacts.py                  # save_fig -> outputs/figures/*.png, save_json -> outputs/data/*.json
app/streamlit_app.py
notebooks/01..06                # see "Notebooks" below; executed in place with outputs committed
tests/                          # see "Testing" below
outputs/figures/                # PNG figures (tracked)   outputs/data/  # JSON results (tracked)
```

**Build methods from scratch, then compare.** The point of the project is pedagogical: a
reader should see each method constructed, not imported from a conformal library. Keep
implementations readable and NumPy-vectorised.

---

## The five methods

| Method | What it is | Role in the arena |
|---|---|---|
| **Normal-theory** (`normal.py`) | prediction ± multiple of residual std; one constant width for all predictions | The textbook default and the protagonist that the arena breaks |
| **Quantile regression** (`quantile_regression.py`) | predict the 5th & 95th percentiles directly via **pinball (quantile) loss**; width can vary with the input | First method whose interval widens where the model is uncertain |
| **GBM quantiles** (`gbm_quantile.py`) | same idea with a gradient booster's built-in quantile objective; separate models for lower & upper bounds | The practical, powerful version most libraries offer and few use |
| **Bootstrap** (`bootstrap.py`) | resample training data, refit, use the spread of predictions | Captures *estimate* uncertainty more than full *predictive* uncertainty — state this honestly; the arena measures it |
| **Conformal** (`conformal.py`) | **split conformal**: hold out a calibration set, measure errors, take the right error quantile, wrap the interval around any prediction. **adaptive (Mondrian)**: let width vary by region | The payoff. Guaranteed nominal coverage on any model/any data; adaptive fixes plain conformal's constant width |

---

## Key domain concepts (the vocabulary of this repo)

- **Prediction interval** is judged by **coverage**: fraction of held-out truths the interval
  contains. Should equal the nominal level (e.g. 90%). Coverage is to intervals what
  calibration is to classification probabilities — and just as ignored.
- **Heteroscedasticity** = error size varies across the prediction range. This is the realistic
  case that breaks constant-width intervals: too wide where the model is confident, too narrow
  where it is not.
- **Conditional coverage** = coverage *within regions* of the prediction range. Overall coverage
  can look fine while conditional coverage is badly off — this plot is the showpiece of NB02.
- **Efficiency** = interval width. At **equal coverage**, tighter is better. Never compare width
  without first checking coverage is matched.
- **Conformal guarantee** holds under a mild **exchangeability** assumption. It can fail under
  distribution shift / time series — state this plainly (point to `forecasting_arena`).
- **Quantile crossing** (predicted lower bound above upper bound) can occur — handle with
  **monotonic rearrangement** and report it honestly.
- **The rule that outranks all methods:** always measure coverage on held-out data. An interval
  you have not coverage-tested is decoration, not information.

---

## Datasets

Synthetic data with **known truth** is primary — coverage can only be validated when the true
value is known. `synthetic.py` exposes error structure as a knob:

- **homoscedastic** (constant error) — the normal interval's home, where it should work;
- **heteroscedastic** (error grows with prediction) — the realistic case that breaks it;
- **skewed errors** — where symmetric intervals fail;
- **heavy-tailed errors** — where normality fails.

A small **real** dataset (California housing or similar) to ground the arena is **planned but not
yet implemented** — `synthetic.py` is the only source today; don't reference a real dataset as if
it exists. All generators and calibration splits are **seeded** (seeds live in `config/settings.yaml`).

---

## Notebooks (budget < 5 min each; actually seconds on the 2-CPU box)

Each saves its figures to `outputs/figures/*.png` and its numeric/text results to
`outputs/data/*.json` via `src/artifacts.save_fig` / `save_json`. They are executed in place
(`make notebooks` uses `--inplace`), so committed `.ipynb` files carry their outputs.

1. **`01_a_number_is_not_enough`** — the motivating decision a single number can't support;
   introduce the prediction interval and coverage; frame the whole project as a coverage contest.
2. **`02_the_textbook_interval_and_why_it_fails`** *(showpiece)* — build the normal interval,
   show it working on homoscedastic data (~0.90, flat conditional), then break it on
   heteroscedastic data: overall coverage ~0.88 but conditional coverage ~0.66 in the volatile
   region vs ~1.0 in the calm one; skew breaks symmetry.
3. **`03_quantile_methods`** — quantile regression (pinball) from scratch + GBM quantiles; both
   recover varying width with much better conditional coverage; honest limit: needs retraining,
   no guarantee → sets up conformal.
4. **`04_conformal_prediction`** *(payoff)* — split conformal from scratch wrapped around an
   existing GBM with no retraining, hitting ~90% almost exactly; explain the guarantee in plain
   language; then adaptive conformal for region-varying width keeping the guarantee.
5. **`05_arena`** — all six method variants × all (synthetic) error structures, scored on
   **coverage** and **efficiency** (width), plus a 15-seed repeats boxplot showing the result is
   robust; honest verdict on when the simple interval is genuinely fine.
6. **`06_decision_framework`** — the guide: clean/constant/normal → textbook is fine and cheap;
   varying error → quantile/GBM/adaptive conformal; can't retrain → split conformal; need a
   guarantee → conformal; always → coverage-test. Ships a `coverage_report` that builds intervals
   every way and reports actual vs nominal.

---

## Streamlit app (`app/streamlit_app.py`)

1. **Interval viewer** — pick error structure + method, see the band over the data.
2. **Coverage meter** — run any method on held-out data, actual vs promised 90%, shortfall highlighted.
3. **Assumption breaker** — crank the heteroscedasticity slider, watch the normal interval collapse while conformal holds.
4. **Wrap-my-model demo** — pick a trained model, watch split conformal wrap it with no retraining.

---

## Commands (use the Makefile; it wraps `uv`)

```
make setup       # first-time: install uv, uv sync --all-extras, register ipykernel
make sync        # uv sync --all-extras
make lint        # format + check + typecheck
make format      # ruff format src/ tests/ app/
make check       # ruff check --fix src/ tests/ app/
make typecheck   # ty check src/
make test        # pytest
make test-cov    # pytest with coverage
make notebooks   # execute all notebooks (nbconvert, 300s timeout each)
make run         # streamlit app on :8501
make lab         # JupyterLab on :8888
make ci          # sync + lint + test  (what CI runs)
make dev         # lint + test (fast local loop)
```

Run Python through `uv run ...`; don't call a bare `python`/`pytest`.

---

## Coding conventions

- **ruff** formats and lints — line length 99 (`E501` ignored); rule sets
  `E,F,W,I,UP,N,B,A,SIM,PTH` (note `PTH`: use `pathlib`, not `os.path`).
- **ty** type-checks `src/`. Annotate public functions; interval methods take/return arrays with
  documented shapes.
- Config is centralised: coverage levels, noise structures and seeds come from
  `config/settings.yaml` via `src/config.py` (pydantic-settings). **Don't hard-code seeds,
  nominal levels, or noise params** in notebooks or `src/` — read them from config.
- Keep from-scratch methods vectorised and readable; favour explicitness over cleverness since
  the code *is* the teaching material.
- Notebooks persist everything through `src/artifacts`: **every figure → `outputs/figures/*.png`
  via `save_fig(fig, name)`**, **every numeric/text result → `outputs/data/*.json` via
  `save_json(name, payload)`** (DataFrames become record lists; numpy types are coerced). Both
  `outputs/` subdirs are tracked in git.

---

## Testing (invariants that must stay asserted)

`make test` must pass with **zero errors**. The suite encodes the thesis as asserted facts:

- `test_intervals.py` — every method produces **valid** intervals (lower ≤ upper after
  rearrangement; correct shape).
- `test_conformal.py` — **split conformal hits nominal coverage** within ±0.05 on every
  structure and when wrapping a pre-fit model; adaptive keeps coverage and varies width.
- `test_normal_undercovers.py` — the **normal interval undercovers under heteroscedasticity**:
  overall below nominal, volatile-region conditional coverage < 0.75, calm region > 0.95, and
  below split conformal — asserted, not just shown.
- `test_coverage.py` — the coverage evaluators are correct against known truth.

Exact coverage numbers depend on the error structure; assert the **qualitative, robust** result
(normal undercovers; conformal hits nominal) over many seeded repeats, with tolerances, not
brittle point values.

---

## Success criteria (definition of done)

- Normal interval: ~nominal on homoscedastic normal data (flat conditional); under
  heteroscedasticity overall coverage below nominal **and** volatile-region conditional coverage
  **< 0.75** (both asserted), with the calm region over-covering.
- Split conformal: within ±0.05 of nominal on any structure, including when wrapping a pre-fit
  model (asserted).
- Adaptive conformal: ~nominal coverage **with region-varying width** (asserted).
- Quantile methods: better conditional coverage than the normal interval on heteroscedastic data.
- Conformal demonstrably wraps a model **without retraining** (asserted: same fitted object reused).
- Efficiency: at equal coverage, the tighter methods are identified.
- Status now: `make ci` green (29 tests), `make notebooks` green (~50s), 16 PNGs + 6 JSONs produced.

---

## Gotchas / honest caveats to preserve

- **Bootstrap** captures estimate uncertainty more than full predictive uncertainty — say so;
  the arena measures the gap.
- **Conformal exchangeability** can break under distribution shift / time series — state it, point
  to `forecasting_arena` as the fix path.
- **Quantile crossing** — rearrange monotonically and report when it happens.
- Never claim a method "covers better" without matched coverage — only then is width comparable.

---

## Workflow / token rules

- **Caveman mode: ON.** Code-first, minimal prose. (Project skill at `.claude/skills/caveman.md`.)
- `/compact` at notebook boundaries to keep context tight.
- Commit/push only when asked. License is **Apache-2.0**.
