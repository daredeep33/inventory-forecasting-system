# Automated Inventory Forecasting & Replenishment System

## Overview

This Python tool automates inventory forecasting and generates replenishment order recommendations based on Tally ERP XML exports...

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place Data Files (Important):**
    *   This repository includes sample files (`data/sample_sales.xml`, `data/godowns/sample_Main_Location.xml`) to demonstrate the expected format.
    *   **For actual use, you MUST place your real Tally export files in the corresponding directories:**
        *   Place your sales export XML files (e.g., `sales2022.xml`, `sales2023.xml`) inside the `./data/` directory.
        *   Place your current stock export XML files for each active godown (e.g., `Main_Location.xml`, `Branch_A.xml`) inside the `./data/godowns/` directory.
    *   These actual data files are **not tracked** by Git due to size and privacy.

## Running the Tool

1.  Ensure your data files are placed correctly (see Setup).
2.  Configure parameters in `config/settings.py` if needed (e.g., `MAX_WORKERS`, thresholds, lead times).
3.  Run the main script from the project root directory:
    ```bash
    python main.py
    ```
4.  The script will:
    *   Update/create the SQLite database (`./data/inventory_data.db`) with sales data.
    *   Load current stock.
    *   Generate forecasts using Prophet.
    *   Calculate individual location needs.
    *   Aggregate needs and calculate the final order for 'Main Location'.
    *   Save the output to `./outputs/orders/replenishment_orders_main_location_aggregated.csv`.
    *   Log detailed information to files in the `./logs/` directory.

## Technologies Used

*   Python
*   Pandas, NumPy
*   Prophet
*   Lxml
*   SQLite3
*   concurrent.futures
