# Macro-Economic Shrinkflation & Supply Chain Predictor

## Project Overview
In highly volatile economic environments, consumer packaged goods (CPG) companies often resort to "shrinkflation"—reducing product size while maintaining the sticker price—to mask rising supply chain costs. 

This project is an end-to-end Python data pipeline that ingests historical macroeconomic data, engineers custom market stress features, and deploys predictive AI to forecast future periods of high shrinkflation risk. 

## Business Value
Instead of looking at raw inflation rates, this tool identifies **market unpredictability**. By combining 12-month rolling volatility with inflation momentum, the model flags the exact "panic windows" where supply chain stress forces manufacturers to alter product sizes. 

## Technical Stack
* **Language:** Python 3.12
* **Data Manipulation:** `pandas`, `numpy` (Handling unpivoting, messy date formats, and missing values)
* **Predictive AI:** Meta `prophet` (Time-series forecasting with multiplicative seasonality)
* **Visualization:** `plotly` (Interactive web dashboards), `matplotlib`

## Core Methodology
1. **Automated Data Ingestion:** Extracts and dynamically cleans 50+ years of raw World Bank Food CPI data, navigating inconsistent column headers and string-based date formats.
2. **Feature Engineering:** * Calculated **Volatility:** 12-month rolling standard deviation to measure market stress.
   * Calculated **Momentum:** Current month vs. 6-month lagged inflation to track acceleration.
3. **Trigger Logic:** Programmatically flagged historical periods where Volatility exceeded 1 Standard Deviation above the mean alongside positive Momentum.
4. **AI Forecasting:** Trained a Prophet time-series model on the cleaned dataset to predict Food CPI trajectories and confidence intervals 24 months into the future.

## How to Run Locally

1. Clone the repository:
    ```bash
    git clone [https://github.com/Thekaran11/shrinkflation-predictor.git](https://github.com/Thekaran11/shrinkflation-predictor.git)
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the analysis pipeline:
    ```bash
    python melts.py
    ```
