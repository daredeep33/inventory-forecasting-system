import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from config.settings import (
    SUPPLIER_LEAD_TIME,
    MINIMUM_ORDER_QUANTITY,
    CATEGORY_MULTIPLIERS,
    ANNUAL_SALES_THRESHOLDS,
    TRENDING_THRESHOLD,
    SEASONAL_THRESHOLD,
    SEASONAL_YEARS_BACK,
    SEASONAL_PERIOD_MONTHS,
    RECENT_SALES_WINDOW_DAYS,
    FORECAST_HORIZON_DAYS
)
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler('inventory_calc.log', maxBytes=1024*1024, backupCount=5)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)  # Use a specific logger for this module
logger.addHandler(log_handler)
logger.setLevel(logging.DEBUG) #Set logging level


def calculate_replenishment(
    sales_df: pd.DataFrame,
    forecasts: pd.DataFrame,
    inventory: dict
) -> pd.DataFrame:
    """Calculates replenishment orders based on sales, forecasts, and inventory."""

    inventory_items = []
    for (product, godown), qty in inventory.items():
        if isinstance(product, str) and product.strip():  # Prevent NaN keys
            inventory_items.append({
                "Product": product.strip().upper(), # To make it Product name
                "Godown": godown.strip().title(),
                "Current_Stock": qty
            })
    inventory_df = pd.DataFrame(inventory_items)

    current_date = pd.Timestamp.now().normalize()
    max_sales_date = sales_df["Date"].max()

    # ====== Seasonal Analysis ======
    seasonal_sales = pd.DataFrame()
    for year_offset in SEASONAL_YEARS_BACK:
        period_start = current_date - relativedelta(
            years=year_offset,
            month=current_date.month,
            day=1
        )
        period_end = period_start + relativedelta(months=SEASONAL_PERIOD_MONTHS)

        effective_start = min(period_start, max_sales_date)
        effective_end = min(period_end, max_sales_date)

        period_mask = (
            (sales_df["Date"] >= effective_start) &
            (sales_df["Date"] <= effective_end)
        )

        year_sales = (
            sales_df[period_mask]
            .groupby(["Product","Godown"], as_index=False)["Quantity"]
            .sum()
            .rename(columns={"Quantity": f"Sales_Y-{year_offset}"})
        )

        if not seasonal_sales.empty:
            seasonal_sales = pd.merge(
                seasonal_sales, year_sales,
                on=["Product","Godown"], how="outer", suffixes=("", "_drop")
            ).filter(regex="^(?!.*_drop)")
        else:
            seasonal_sales = year_sales

    seasonal_cols = [c for c in seasonal_sales.columns if "Sales_Y" in c]
    seasonal_sales["Seasonal_Avg"] = seasonal_sales[seasonal_cols].mean(axis=1).fillna(0)

# ====== Recent Sales Velocity ======
    velocity_start = current_date - pd.DateOffset(days=RECENT_SALES_WINDOW_DAYS)

    print(f"Velocity start: {velocity_start}") # Add this line

    recent_sales = (
        sales_df[sales_df["Date"] >= velocity_start]
        .groupby(["Product","Godown"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"Quantity": "Recent_Sales"})
    )

    # ====== Demand Classification ======
    classification_df = pd.merge(
        seasonal_sales,
        recent_sales,
        on=["Product","Godown"],
        how="outer"
    ).fillna(0)

    classification_df["Annualized_Recent"] = classification_df["Recent_Sales"] * (365 / RECENT_SALES_WINDOW_DAYS)

    logger.debug(f"Before classification:\n{classification_df[['Product', 'Seasonal_Avg', 'Recent_Sales', 'Annualized_Recent']].head().to_string()}")

    # Priority-based classification conditions (order-sensitive)
    conditions = [
        # 1. Non-Moving (top priority)
        (classification_df["Seasonal_Avg"] == 0) & (classification_df["Recent_Sales"] == 0),
        
        # 2. Trending (check early to avoid false Seasonal/U-Fast flags)
        classification_df["Recent_Sales"] >= TRENDING_THRESHOLD,
        
        # 3. Ultra-Fast-Moving (annualized, not trending)
        classification_df["Annualized_Recent"] > ANNUAL_SALES_THRESHOLDS["Ultra-Fast-Moving"][0],
        
        # 4. Fast-Moving
        classification_df["Annualized_Recent"] > ANNUAL_SALES_THRESHOLDS["Fast-Moving"][0],
        
        # 5. Seasonal
        classification_df["Seasonal_Avg"] >= SEASONAL_THRESHOLD,
        
        # 6. Medium-Demand
        (classification_df["Annualized_Recent"] > ANNUAL_SALES_THRESHOLDS["Medium-Demand"][0]) & 
        (classification_df["Annualized_Recent"] <= ANNUAL_SALES_THRESHOLDS["Medium-Demand"][1]),
        
        # 7. Slow-Moving (catch-all)
        classification_df["Annualized_Recent"] <= ANNUAL_SALES_THRESHOLDS["Slow-Moving"][1]
    ]

    choices = [
        "Non-Moving",
        "Ultra-Fast-Moving",
        "Fast-Moving",
        "Trending",
        "Seasonal",
        "Medium-Demand",
        "Slow-Moving"
    ]

    classification_df["Category"] = np.select(
        conditions,
        choices,
        default="Default"
    )
    print(f"Categories after selection:\n{classification_df['Category'].value_counts()}")
    logger.debug(f"Category Value Counts:\n{classification_df['Category'].value_counts().to_string()}")
    logger.debug(f"Classification Dataframe with Categories:\n{classification_df.head().to_string()}")

    classification_df["Multiplier"] = classification_df["Category"].apply(
        lambda cat: CATEGORY_MULTIPLIERS.get(cat, CATEGORY_MULTIPLIERS["Default"])
    )

    # ====== Forecast Handling with Prophet ======
    forecasts_agg = (
        forecasts.groupby(["Product", "Godown"], as_index=False)
        ["Prophet_Forecast"].sum()
    )

    forecasts_agg["Prophet_Forecast"] = np.where(
        forecasts_agg["Prophet_Forecast"] <= 0,
        MINIMUM_ORDER_QUANTITY,
        forecasts_agg["Prophet_Forecast"]
    )

    # ====== Data Alignment ======
    merged = (
        pd.merge(
            classification_df,
            forecasts_agg,
            on=["Product","Godown"],
            how="left"
        )
        .merge(
            inventory_df,
            on=["Product","Godown"],
            how="left"
        )
        .fillna({
            "Current_Stock": 0,
            "Multiplier": CATEGORY_MULTIPLIERS["Default"],
            "Seasonal_Avg": 0,
            "Recent_Sales": 0,
            "Annualized_Recent": 0,
            "Category": "Default",
            "Godown": "Amazon Warehouse"
        })
    )

    # ====== Inventory Calculations ======
    merged["Lead_Time_Demand"] = (
        merged["Prophet_Forecast"] * merged["Multiplier"]
    ).round(2)

    # Enhanced Safety Stock Logic
    merged["Safety_Stock"] = np.select(
        [
            merged["Category"] == "Ultra-Fast-Moving",
            merged["Category"] == "Fast-Moving",
            merged["Category"] == "Trending",
            merged["Category"] == "Seasonal"
        ],
        [
        merged["Lead_Time_Demand"] * 0.1,   # Ultra-Fast
        merged["Lead_Time_Demand"] * 0.1,   # Fast-Moving
        merged["Lead_Time_Demand"] * 0.1,  # Trending (with 75% buffer)
        merged["Lead_Time_Demand"] * 0.1    # Seasonal
        ],
        default=merged["Lead_Time_Demand"] * 0.1  # 20% for others
    )
    # Apply MINIMUM_ORDER_QUANTITY before overriding for Non-Moving
    merged["Reorder_Qty"] = np.select(
        [
            merged["Category"] == "Non-Moving",
            merged["Current_Stock"] <= merged["Lead_Time_Demand"] / 2
        ],
        [
            0,  # No reorder for Non-Moving
            MINIMUM_ORDER_QUANTITY  # Force minimal restock even if formula suggests lower
        ],
        default=(merged["Lead_Time_Demand"] + merged["Safety_Stock"] - merged["Current_Stock"])
                .apply(np.ceil)
                .clip(lower=MINIMUM_ORDER_QUANTITY)
    )

    # Final validation and output
    return (
        merged
        .drop_duplicates(subset=["Product", "Godown"])
        .sort_values(["Category", "Product"], ascending=[False, True])
        [[
            "Product",
            "Godown",
            "Current_Stock",
            "Category",
            "Seasonal_Avg",
            "Recent_Sales",
            "Annualized_Recent",
            "Lead_Time_Demand",
            "Safety_Stock",
            "Reorder_Qty",
            "Multiplier"
        ]]
        .reset_index(drop=True)
    )