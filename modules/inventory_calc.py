# modules/inventory_calc.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
# from logging.handlers import RotatingFileHandler # Handlers configured elsewhere
from pathlib import Path

# Import necessary settings from the config file
from config.settings import (
    SUPPLIER_LEAD_TIME,
    MINIMUM_ORDER_QUANTITY,
    NON_MOVING_THRESHOLD,
    CATEGORY_MULTIPLIERS,
    ANNUAL_SALES_THRESHOLDS,
    TRENDING_THRESHOLD,       # Ratio for recent vs annual
    SEASONAL_THRESHOLD,       # Ratio for seasonal vs avg monthly
    SEASONAL_YEARS_BACK,      # Desired lookback years
    SEASONAL_PERIOD_MONTHS,   # Duration of seasonal period
    RECENT_SALES_WINDOW_DAYS,
    BASE_SAFETY_DAYS,         # Base days for safety stock calculation
    LOG_DIR                   # Import LOG_DIR
)

# --- Configure Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(module)s - %(message)s')
# Ensure logs directory exists
log_dir_path = Path(LOG_DIR)
log_dir_path.mkdir(parents=True, exist_ok=True)
log_handler = logging.FileHandler(log_dir_path / 'inventory_calc.log', mode='a', encoding='utf-8')
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
if not logger.handlers: # Avoid adding multiple handlers if logger already exists
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG) # Set desired logging level


def calculate_replenishment(
    sales_df: pd.DataFrame,
    forecasts: pd.DataFrame,
    inventory: dict
) -> pd.DataFrame:
    """
    Calculates replenishment metrics for each product/godown based on sales history,
    demand forecasts, current inventory, and defined inventory policies.
    This function calculates the *individual needs* before central aggregation.

    Args:
        sales_df: DataFrame with historical sales ('Date', 'Product', 'Godown', 'Quantity').
        forecasts: DataFrame with demand forecasts ('Product', 'Godown', 'Prophet_Forecast').
        inventory: Dictionary mapping (Product, Godown) tuples to current stock quantity.

    Returns:
        DataFrame with individual replenishment details (including ROP, Ideal_Order_Qty).
    """
    logger.info("Starting individual replenishment calculation...")
    function_start_time = datetime.now()

    # 1. Prepare Inventory Data
    # ========================
    if not inventory:
        logger.error("Inventory data is empty. Cannot calculate replenishment.")
        return pd.DataFrame(columns=[ # Return empty DF with expected columns
            "Product", "Godown", "Category", "Current_Stock", "Avg_Daily_Sales",
            "Lead_Time_Demand", "Safety_Stock", "Reorder_Point", "Inventory_Position",
            "Ideal_Order_Qty", "Multiplier", "Annual_Sales",
            "Recent_Sales", "Annualized_Recent", "Seasonal_Avg_Monthly"
        ])

    inventory_items = []
    for (product, godown), qty in inventory.items():
        if isinstance(product, str) and product.strip() and isinstance(godown, str) and godown.strip():
            inventory_items.append({
                "Product": product.strip().upper(),
                "Godown": godown.strip().title(),
                "Current_Stock": max(0, float(qty)) # Ensure non-negative
            })
        else: logger.warning(f"Skipping invalid inventory key: Product='{product}', Godown='{godown}'")

    if not inventory_items:
        logger.error("No valid inventory items found after cleaning keys. Cannot calculate replenishment.")
        return pd.DataFrame()

    inventory_df = pd.DataFrame(inventory_items)
    logger.debug(f"Prepared inventory_df with {len(inventory_df)} items.")

    # 2. Prepare Sales Data & Dates
    # ============================
    if sales_df.empty or not all(col in sales_df.columns for col in ['Date', 'Product', 'Godown', 'Quantity']):
         logger.warning("Sales data is empty or missing required columns. Classification and safety stock will be based on defaults/fallbacks.")
         sales_df = pd.DataFrame(columns=['Date', 'Product', 'Godown', 'Quantity'])
         sales_df['Date'] = pd.to_datetime(sales_df['Date']); sales_df['Quantity'] = pd.to_numeric(sales_df['Quantity'], errors='coerce')
         sales_df['Product'] = sales_df['Product'].astype(str); sales_df['Godown'] = sales_df['Godown'].astype(str)
    else:
         # Ensure correct types and apply string cleaning using .str accessor
         sales_df['Product'] = sales_df['Product'].astype(str).str.strip().str.upper()
         sales_df['Godown'] = sales_df['Godown'].astype(str).str.strip().str.title()
         sales_df['Date'] = pd.to_datetime(sales_df['Date']).dt.tz_localize(None)
         sales_df['Quantity'] = pd.to_numeric(sales_df['Quantity'], errors='coerce').fillna(0)

    current_date = pd.Timestamp.now().normalize()
    max_sales_date = sales_df["Date"].max() if not sales_df.empty else current_date - timedelta(days=1)
    min_sales_date = sales_df["Date"].min() if not sales_df.empty else current_date
    logger.debug(f"Current Date: {current_date}, Max Sales Date: {max_sales_date}, Min Sales Date: {min_sales_date}")
    analysis_cols = ["Product", "Godown"]

    # 3. Aggregate Sales Data for Classification
    # =========================================
    logger.debug("Aggregating sales data...")
    agg_start_time = datetime.now()
    # --- Annual Sales (last 365 days) ---
    annual_sales_start = max_sales_date - timedelta(days=365)
    annual_sales_mask = (sales_df["Date"] > annual_sales_start) & (sales_df["Date"] <= max_sales_date)
    annual_sales = sales_df[annual_sales_mask].groupby(analysis_cols, as_index=False)["Quantity"].sum().rename(columns={"Quantity": "Annual_Sales"})
    # --- Recent Sales (last X days) ---
    recent_sales_start = max_sales_date - timedelta(days=RECENT_SALES_WINDOW_DAYS)
    recent_sales_mask = (sales_df["Date"] > recent_sales_start) & (sales_df["Date"] <= max_sales_date)
    recent_sales = sales_df[recent_sales_mask].groupby(analysis_cols, as_index=False)["Quantity"].sum().rename(columns={"Quantity": "Recent_Sales"})
    # --- Dynamic Seasonal Analysis ---
    seasonal_sales_list = []; available_years = set(sales_df['Date'].dt.year) if not sales_df.empty else set()
    target_years_from_settings = [max_sales_date.year - offset for offset in SEASONAL_YEARS_BACK]
    actual_years_to_check = sorted([year for year in target_years_from_settings if year in available_years], reverse=True)
    logger.info(f"Desired seasonal years back: {SEASONAL_YEARS_BACK}. Actual years with data: {actual_years_to_check}")
    if not actual_years_to_check: logger.warning("No historical data for desired seasonal range.")
    for check_year in actual_years_to_check:
        year_offset = max_sales_date.year - check_year
        period_end_approx = max_sales_date - relativedelta(years=year_offset)
        period_start_approx = period_end_approx - relativedelta(months=SEASONAL_PERIOD_MONTHS) + timedelta(days=1)
        period_mask = ((sales_df["Date"].dt.year == check_year) & (sales_df["Date"] >= period_start_approx) & (sales_df["Date"] <= period_end_approx))
        period_data = sales_df[period_mask]
        if period_data.empty: continue
        days_in_period = (period_data['Date'].max() - period_data['Date'].min()).days + 1
        if days_in_period <= 0: continue
        year_sales_agg = period_data.groupby(analysis_cols, as_index=False)["Quantity"].sum()
        avg_monthly_sales = year_sales_agg["Quantity"] / max(1, (days_in_period / 30.4375))
        year_sales_agg[f"Sales_Y{year_offset}_AvgMonthly"] = avg_monthly_sales
        seasonal_sales_list.append(year_sales_agg[analysis_cols + [f"Sales_Y{year_offset}_AvgMonthly"]])
        logger.debug(f"Calculated seasonal avg for offset {year_offset} ({check_year}).")
    if seasonal_sales_list:
        seasonal_analysis = seasonal_sales_list[0]
        for i in range(1, len(seasonal_sales_list)): seasonal_analysis = pd.merge(seasonal_analysis, seasonal_sales_list[i], on=analysis_cols, how="outer")
        seasonal_cols = [col for col in seasonal_analysis.columns if "AvgMonthly" in col]
        seasonal_analysis["Seasonal_Avg_Monthly"] = seasonal_analysis[seasonal_cols].mean(axis=1)
        seasonal_analysis = seasonal_analysis[analysis_cols + ["Seasonal_Avg_Monthly"]]
    else: seasonal_analysis = pd.DataFrame(columns=analysis_cols + ["Seasonal_Avg_Monthly"])
    # --- Merge Aggregated Sales ---
    all_items = pd.concat([sales_df[analysis_cols].drop_duplicates(), inventory_df[analysis_cols].drop_duplicates()]).drop_duplicates().reset_index(drop=True)
    analysis_df = all_items
    analysis_df = pd.merge(analysis_df, annual_sales, on=analysis_cols, how="left")
    analysis_df = pd.merge(analysis_df, recent_sales, on=analysis_cols, how="left")
    analysis_df = pd.merge(analysis_df, seasonal_analysis, on=analysis_cols, how="left")
    analysis_df = analysis_df.fillna(0)
    analysis_df["Annualized_Recent"] = analysis_df["Recent_Sales"] * (365.0 / max(1, RECENT_SALES_WINDOW_DAYS))
    analysis_df["Avg_Monthly_Sales"] = analysis_df["Annual_Sales"] / 12.0
    logger.debug(f"Sales aggregation complete in {(datetime.now() - agg_start_time).total_seconds():.2f}s.")

    # 4. Demand Classification
    # ========================
    logger.info("Classifying demand...")
    conditions = [
        (analysis_df["Annual_Sales"] <= NON_MOVING_THRESHOLD) & (analysis_df["Annualized_Recent"] <= NON_MOVING_THRESHOLD),
        (analysis_df["Annual_Sales"] > 0) & ((analysis_df["Annualized_Recent"] / analysis_df["Annual_Sales"]) > TRENDING_THRESHOLD),
        (analysis_df["Avg_Monthly_Sales"] > 0) & ((analysis_df["Seasonal_Avg_Monthly"] / analysis_df["Avg_Monthly_Sales"]) > SEASONAL_THRESHOLD),
        (analysis_df["Annualized_Recent"] >= ANNUAL_SALES_THRESHOLDS["Ultra-Fast-Moving"][0]),
        (analysis_df["Annualized_Recent"] >= ANNUAL_SALES_THRESHOLDS["Fast-Moving"][0]),
        (analysis_df["Annualized_Recent"] >= ANNUAL_SALES_THRESHOLDS["Medium-Demand"][0]),
        (analysis_df["Annualized_Recent"] > NON_MOVING_THRESHOLD) & (analysis_df["Annualized_Recent"] < ANNUAL_SALES_THRESHOLDS["Medium-Demand"][0])
    ]
    choices = ["Non-Moving", "Trending", "Seasonal", "Ultra-Fast-Moving", "Fast-Moving", "Medium-Demand", "Slow-Moving"]
    analysis_df["Category"] = np.select(conditions, choices, default="New-Product")
    has_sales_history = analysis_df['Annual_Sales'] + analysis_df['Recent_Sales'] > 0
    analysis_df.loc[~has_sales_history, 'Category'] = 'New-Product'
    logger.info(f"Category Value Counts:\n{analysis_df['Category'].value_counts().to_string()}")
    analysis_df["Multiplier"] = analysis_df["Category"].apply(lambda cat: CATEGORY_MULTIPLIERS.get(cat, CATEGORY_MULTIPLIERS["Default"]))

    # 5. Prepare Forecast Data
    # =======================
    if forecasts.empty or not all(col in forecasts.columns for col in ['Product', 'Godown', 'Prophet_Forecast']):
        logger.warning("Forecast data is empty/invalid."); forecasts_agg = pd.DataFrame(columns=['Product', 'Godown', 'Prophet_Forecast'])
    else:
        forecasts_agg = forecasts[['Product', 'Godown', 'Prophet_Forecast']].copy()
        forecasts_agg['Product'] = forecasts_agg['Product'].astype(str).str.strip().str.upper()
        forecasts_agg['Godown'] = forecasts_agg['Godown'].astype(str).str.strip().str.title()
        forecasts_agg["Prophet_Forecast"] = forecasts_agg["Prophet_Forecast"].clip(lower=0).astype(float) # Ensure float
        logger.debug(f"Prepared aggregated forecasts for {len(forecasts_agg)} items.")

    # 6. Merge All Data Sources
    # ========================
    logger.info("Merging inventory, analysis, and forecast data...")
    merged = pd.merge(inventory_df, analysis_df, on=analysis_cols, how="left")
    merged = pd.merge(merged, forecasts_agg, on=analysis_cols, how="left")
    default_multiplier = CATEGORY_MULTIPLIERS.get("New-Product", CATEGORY_MULTIPLIERS["Default"])
    merged.fillna({"Annual_Sales": 0, "Recent_Sales": 0, "Seasonal_Avg_Monthly": 0, "Annualized_Recent": 0, "Avg_Monthly_Sales": 0, "Category": "New-Product", "Multiplier": default_multiplier, "Prophet_Forecast": 0}, inplace=True)
    numeric_cols_float = ['Current_Stock', 'Prophet_Forecast', 'Multiplier', 'Annualized_Recent', 'Annual_Sales', 'Recent_Sales', 'Seasonal_Avg_Monthly', 'Avg_Monthly_Sales']
    for col in numeric_cols_float: merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0).astype(float)
    logger.debug("Data merging complete.")

    # 7. Calculate Inventory Metrics (per Location)
    # ============================================
    logger.info("Calculating individual location inventory metrics (LTD, SS, ROP)...")
    metrics_start_time = datetime.now()
    merged["Lead_Time_Demand"] = merged["Prophet_Forecast"]
    merged["Avg_Daily_Sales"] = (merged["Annualized_Recent"] / 365.0).clip(lower=0)
    merged["Safety_Stock"] = (merged["Avg_Daily_Sales"] * BASE_SAFETY_DAYS * merged["Multiplier"]).clip(lower=0)
    merged.loc[merged["Category"] == "Non-Moving", "Safety_Stock"] = 0
    merged["Reorder_Point"] = merged["Lead_Time_Demand"] + merged["Safety_Stock"]
    merged["Inventory_Position"] = merged["Current_Stock"] # Assuming no On Order tracking
    # --- Calculate Ideal Order Quantity (Net Requirement) ---
    merged["Ideal_Order_Qty"] = (merged["Reorder_Point"] - merged["Inventory_Position"]).clip(lower=0)
    logger.debug(f"Individual metrics calculated in {(datetime.now() - metrics_start_time).total_seconds():.2f}s.")
    logger.debug(f"Metrics Sample:\n{merged[['Product', 'Godown', 'Category', 'Lead_Time_Demand', 'Safety_Stock', 'Reorder_Point', 'Inventory_Position', 'Ideal_Order_Qty']].head().to_string()}")

    # 8. Final Output Preparation (Individual Needs)
    # =============================================
    logger.info("Preparing final DataFrame with individual location needs...")
    output_columns = [
        "Product", "Godown", "Category", "Current_Stock", "Inventory_Position",
        "Avg_Daily_Sales", "Lead_Time_Demand", "Safety_Stock", "Reorder_Point",
        "Ideal_Order_Qty", # Key output for aggregation
        "Multiplier", "Annual_Sales", "Recent_Sales", "Annualized_Recent", "Seasonal_Avg_Monthly"
    ]
    final_columns = [col for col in output_columns if col in merged.columns]
    final_df = (
        merged[final_columns]
        .round({ # Apply rounding to specific columns for cleaner output later
            'Current_Stock': 2, 'Inventory_Position': 2, 'Avg_Daily_Sales': 3, 'Lead_Time_Demand': 2,
            'Safety_Stock': 2, 'Reorder_Point': 2, 'Ideal_Order_Qty': 2, 'Multiplier': 2,
            'Annual_Sales': 0, 'Recent_Sales': 0, 'Annualized_Recent': 0, 'Seasonal_Avg_Monthly': 2
         })
        .sort_values(by=["Godown", "Category", "Product"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    total_execution_time = (datetime.now() - function_start_time).total_seconds()
    logger.info(f"Finished individual replenishment calculations for {len(final_df)} items in {total_execution_time:.2f} seconds.")
    return final_df