# soc-kalman-filter-comparison

Simulation and real-data validation code for the paper:

> **"State-of-Charge Estimation of a Lithium-Ion Battery under Sensor and Model Bias:
> A Systematic Comparison of EKF, UKF, Adaptive EKF, and Joint EKF"**
>
> Luc Lam Thanh
> Industrial University of Ho Chi Minh City (IUH) & Bosch Global Software Technologies Vietnam
> ORCID: [0009-0005-2382-3696](https://orcid.org/0009-0005-2382-3696)

---

## Repository structure

| File | Description |
|---|---|
| `filters.py` | EKF, UKF, AEKF, JEKF, CC-OCV estimators (vectorised, runs × time) |
| `model.py` | 2RC equivalent-circuit model, OCV-SOC polynomial, nominal parameters |
| `io_cell.py` | Reads LG HG2 `.mat` or `.csv` files, resamples to 1 Hz |
| `build_ocv.py` | Builds OCV curve from C/20 charge or discharge test |
| `fit_hppc.py` | Identifies R0, R1, τ1, R2, τ2 from HPPC pulse tests |
| `run_real.py` | Tune on Mixed cycles, evaluate on UDDS/HWFET/LA92/US06/Mixed |
| `plot_run.py` | Plots reference SOC, estimated SOC and error for one test file |
| `cycles.py` | Drive-cycle helpers |
| `make_fake_data.py` | Generates synthetic data to test the pipeline without the dataset |
| `inspect_file.py` | Prints field names and sign conventions of a `.mat` file |
| `check_cref.py` | Checks Coulomb-counting reference quality |
| `requirements.txt` | Python dependencies |

---

## Dataset

Download the LG HG2 18650 dataset (not included in this repo):

> Kollmeyer, P. (2020). *LG 18650HG2 Li-ion Battery Data*. Mendeley Data, V3.
> https://doi.org/10.17632/cp3473x7xv.3

Place the `.mat` files in a local `data/` directory.

---

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.11, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2.

---

## How to run

```bash
# 1. Inspect dataset file structure
python inspect_file.py <mat_file>

# 2. Build OCV curve from C/20 test
python build_ocv.py <C20_file> --out ocv.csv

# 3. Identify 2RC ECM parameters from HPPC test
python fit_hppc.py <HPPC_file> --ocv ocv.csv --cap <Ah> --out params.json

# 4. Run all filters: tune on Mixed, evaluate on UDDS/HWFET/LA92/US06/Mixed
python run_real.py --ocv ocv.csv --params params.json \
    --tune <Mixed_files...> --test <test_files...> \
    --cap <Ah> --out results.json

# 5. Plot results for one test file
python plot_run.py <test_file> --ocv ocv.csv --params params.json \
    --results results.json --cap <Ah>
```

### Quick test (no dataset needed)

```bash
python make_fake_data.py   # generates synthetic data in fake_data/
```

---

## Reproducing paper results

The results reported in the paper (Table II, Figures 4–7) were obtained by running
`run_real.py` on the LG HG2 dataset at six temperatures: −20, −10, 0, 10, 25, 40 °C.
Pre-identified OCV and ECM parameter files are included (`ocv.csv`, `params.json`,
`params_per_pulse.csv`, `ocv_info.json`) so the identification steps can be skipped.

---

## Citation

If you use this code, please cite our paper (DOI will be updated after acceptance):

```
Luc Lam Thanh. State-of-Charge Estimation of a Lithium-Ion Battery under Sensor
and Model Bias: A Systematic Comparison of EKF, UKF, Adaptive EKF, and Joint EKF.
Journal of Energy Storage, 2026. (under review)
```

---

## License

[MIT License](LICENSE)
