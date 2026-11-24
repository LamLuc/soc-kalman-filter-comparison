# soc-kalman-filter-comparison

Simulation code for the paper:

**"State-of-Charge Estimation of a Lithium-Ion Battery under Sensor and Model Bias: A Systematic Comparison of EKF, UKF, Adaptive EKF, and Joint EKF"**

*Lâm Thành Lực — Industrial University of Ho Chi Minh City (IUH) & Bosch Global Software Technologies Vietnam*

## Dataset

This study uses the publicly available LG HG2 18650 dataset:

> Kollmeyer, P. (2020). LG 18650HG2 Li-ion Battery Data. Mendeley Data.  
> https://doi.org/10.17632/cp3473x7xv.3

## Requirements

- MATLAB R2020b or later (Simulink, Control System Toolbox)

## How to run

1. Download the LG HG2 dataset and place in `data/`
2. Run `main_identify.m` to identify ECM parameters at each temperature
3. Run `main_simulate.m` to run all 120 test scenarios
4. Run `main_plot.m` to reproduce all figures in the paper

## Citation

If you use this code, please cite our paper (DOI will be added after acceptance).

## License

MIT License — see [LICENSE](LICENSE)
