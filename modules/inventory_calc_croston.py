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

def calculate_replenishment(
    sales_df: pd.DataFrame, 
    forecasts: pd.DataFrame, 
    inventory: dict
) -> pd.DataFrame:
    # Convert inventory to DataFrame
    inventory_items = [
        {"Product": product, "Godown": godown, "Current_Stock": qty}
        for (product, godown), qty in inventory.items()
    ]
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
            .groupby("Product", as_index=False)["Quantity"]
            .sum()
            .rename(columns={"Quantity": f"Sales_Y-{year_offset}"})
        )
        
        seasonal_sales = pd.merge(
            seasonal_sales, year_sales,
            on="Product", how="outer", suffixes=("", "_drop")
        ).filter(regex="^(?!.*_drop)") if not seasonal_sales.empty else year_sales

    seasonal_cols = [c for c in seasonal_sales.columns if "Sales_Y" in c]
    seasonal_sales["Seasonal_Avg"] = (
        seasonal_sales[seasonal_cols].mean(axis=1).fillna(0)
    )
    
    # ====== Recent Sales Velocity ======
    velocity_start = current_date - pd.DateOffset(days=RECENT_SALES_WINDOW_DAYS)
    recent_sales = (
        sales_df[sales_df["Date"] >= velocity_start]
        .groupby("Product", as_index=False)["Quantity"]
        .sum()
        .rename(columns={"Quantity": "Recent_Sales"})
    )
    
    # ====== Demand Classification ======
    # Step 1: Merge seasonal and recent sales data
    classification_df = pd.merge(
        seasonal_sales, 
        recent_sales, 
        on="Product", 
        how="outer"
    ).fillna(0)

    # Step 2: Derived columns
    classification_df["Annualized_Recent"] = classification_df["Recent_Sales"] * (365 / RECENT_SALES_WINDOW_DAYS)


    # Priority 1: Identify Non-Moving Items (0 sales in all periods)
    non_moving_mask = (
        (classification_df["Seasonal_Avg"] == 0) & 
        (classification_df["Recent_Sales"] == 0)
    )

    # Classification conditions (order-sensitive)
    conditions = [
        # 1. Non-Moving Items
        non_moving_mask,
        
        # 2. Seasonal Items
        classification_df["Seasonal_Avg"] >= SEASONAL_THRESHOLD,
        
        # 3. Annual Sales Categories
        classification_df["Annualized_Recent"] > ANNUAL_SALES_THRESHOLDS["Fast-Moving"][0],
        (classification_df["Annualized_Recent"] > ANNUAL_SALES_THRESHOLDS["Medium-Demand"][0]) & 
        (classification_df["Annualized_Recent"] <= ANNUAL_SALES_THRESHOLDS["Medium-Demand"][1]),
        classification_df["Annualized_Recent"] <= ANNUAL_SALES_THRESHOLDS["Slow-Moving"][1],
        
        # 4. Trending (applies to non-seasonal items)
        classification_df["Recent_Sales"] >= TRENDING_THRESHOLD
    ]

    choices = [
        "Non-Moving",
        "Seasonal",
        "Fast-Moving",
        "Medium-Demand",
        "Slow-Moving",
        "Trending"
    ]

    classification_df["Category"] = np.select(conditions, choices, default="Default")
    
    # Clean category names and map multipliers
    classification_df["Category"] = classification_df["Category"].str.strip()
    classification_df["Multiplier"] = (
        classification_df["Category"]
        .map(lambda cat: CATEGORY_MULTIPLIERS.get(cat, CATEGORY_MULTIPLIERS["Default"]))
    )
    
    # ====== Forecast Handling ======
    forecasts_agg = (
        forecasts.groupby(["Product", "Godown"], as_index=False)
        ["CrostonOptimized"].sum()
    )
    
    # Apply forecast floor to prevent zero demand
    forecasts_agg["CrostonOptimized"] = np.where(
        forecasts_agg["CrostonOptimized"] <= 0,
        MINIMUM_ORDER_QUANTITY,
        forecasts_agg["CrostonOptimized"]
    )
    
    # ====== Data Merging ======
    merged = pd.merge(
        classification_df,
        forecasts_agg,
        on="Product",
        how="right"
    ).merge(
        inventory_df,
        on=["Product", "Godown"],
        how="left"
    ).fillna({
        "Current_Stock": 0,
        "Multiplier": CATEGORY_MULTIPLIERS["Default"],
        "Seasonal_Avg": 0,
        "Recent_Sales": 0,
        "Annualized_Recent": 0,
        "Category": "Non-Moving"  # Most conservative default
    })
    
    # ====== Core Calculations ======
    merged["Lead_Time_Demand"] = (
        merged["CrostonOptimized"] * merged["Multiplier"]
    ).round(2)

    # Replace the safety stock calculation with category-specific buffers
    merged["Safety_Stock"] = np.where(
        merged["Category"] == "Fast-Moving",
        merged["Lead_Time_Demand"] * 0.5,  # 50% buffer for fast movers
        merged["Lead_Time_Demand"] * 0.2    # 20% for others
    ).round(2)
    
    # FIRST: Calculate all reorder quantities
    merged["Reorder_Qty"] = (
        (merged["Lead_Time_Demand"] + merged["Safety_Stock"] - merged["Current_Stock"])
        .apply(np.ceil)
        .clip(lower=MINIMUM_ORDER_QUANTITY)
        .astype(int)
    )

    # THEN: Override for Non-Moving AFTER initial calculation
    merged["Reorder_Qty"] = np.where(
        merged["Category"] == "Non-Moving",
        0,
        merged["Reorder_Qty"]
    )
    
    # Final cleanup
    return (
        merged.drop_duplicates(subset=["Product", "Godown"])
        .sort_values(["Product", "Godown"])
        [["Product", "Godown", "Current_Stock", "Category",
          "Lead_Time_Demand", "Reorder_Qty"]]
    )