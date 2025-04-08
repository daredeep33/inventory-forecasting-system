# main.py
import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np # Import numpy
import logging
from datetime import datetime, timedelta
import time

# Adjust import paths if needed
from modules.data_loader import (
    update_sales_from_xml,
    get_sales_data_from_db,
    load_current_stock
)
from modules.forecasting import generate_forecasts
from modules.inventory_calc import calculate_replenishment
from config.settings import (
    OUTPUT_DIR,
    GODOWNS_DIR,
    DATA_DIR,
    DATABASE_FILE,
    SEASONAL_YEARS_BACK,
    MAX_WORKERS,
    MINIMUM_ORDER_QUANTITY, # Import MOQ for central calculation
    LOG_DIR                 # Import LOG_DIR
)

# --- Setup ---
# Ensure console output encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configure logging to file AND stream (console) for high-level messages
log_dir_path = Path(LOG_DIR)
log_dir_path.mkdir(parents=True, exist_ok=True) # Ensure log directory exists
log_file_path = log_dir_path / 'main_process.log'

logging.basicConfig(
    level=logging.INFO, # Set base level
    format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8'), # Log INFO and above to file
        logging.StreamHandler(sys.stdout) # Log INFO and above to console
    ]
)
# Reduce verbosity of libraries if needed by getting their specific loggers
# logging.getLogger('prophet').setLevel(logging.WARNING)
# logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

logger = logging.getLogger(__name__) # Get logger for this module


def main():
    overall_start_time = time.time()
    # Use MAX_WORKERS setting to inform user about execution mode
    exec_mode = "Sequential" if MAX_WORKERS is None or MAX_WORKERS <= 1 else f"Parallel (Workers={MAX_WORKERS})"
    logger.info(f"Starting Inventory Calculation Process (Approach B - Central Warehouse Aggregation) ({exec_mode})...")

    # --- Pre-run Checks ---
    logger.info("Performing pre-run checks...")
    required_paths = {
        "Data Directory": Path(DATA_DIR),
        "Godowns Directory": Path(GODOWNS_DIR),
        "Output Directory": Path(OUTPUT_DIR),
        "Logs Directory": Path(LOG_DIR),
        "Database Directory": Path(DATABASE_FILE).parent
    }
    all_paths_ok = True
    for name, path in required_paths.items():
        if name.endswith("Directory") and not path.is_dir():
             logger.info(f"Creating missing directory: {path}")
             try: path.mkdir(parents=True, exist_ok=True)
             except Exception as e: logger.error(f"Failed to create {name}: {path}. Error: {e}", exc_info=True); all_paths_ok = False
        elif not name.endswith("Directory") and not path.parent.exists():
             logger.info(f"Creating missing parent directory for {name}: {path.parent}")
             try: path.parent.mkdir(parents=True, exist_ok=True)
             except Exception as e: logger.error(f"Failed to create parent directory for {name}: {path.parent}. Error: {e}", exc_info=True); all_paths_ok = False
    if not all_paths_ok: logger.critical("Essential directory creation failed. Check permissions. Exiting."); sys.exit(1)
    logger.info("Pre-run checks passed.")

    try:
        # --- Step 1: Update Database ---
        step_start_time = time.time()
        logger.info("[1/6] Updating sales database from XML files...")
        print("\n[1/6] Updating sales database from XML files...")
        update_sales_from_xml() # Uses tqdm internally
        logger.info(f"Database update process complete in {time.time() - step_start_time:.2f}s.")
        print(f"      Database update complete.")

        # --- Step 2: Load Sales Data ---
        step_start_time = time.time()
        max_hist_years = max(SEASONAL_YEARS_BACK) if SEASONAL_YEARS_BACK else 1
        required_days_history = max(365 * max_hist_years, 365, 90)
        buffer_days = 30 # Define buffer_days
        load_start_date = (datetime.now() - timedelta(days=required_days_history + buffer_days)).replace(hour=0, minute=0, second=0, microsecond=0) # Corrected calculation
        logger.info(f"[2/6] Loading sales data from database (from {load_start_date.date()})...")
        print(f"[2/6] Loading sales data from database (approx last {required_days_history/365:.1f} years)...")
        sales = get_sales_data_from_db(start_date=load_start_date)
        if sales.empty:
             logger.warning("No sales data loaded from database for the required period. Forecasts and calculations will use defaults.")
             print("      WARNING: No sales data loaded.")
        else: logger.info(f"Loaded {len(sales)} sales records. Date range: {sales['Date'].min().date()} to {sales['Date'].max().date()}")
        logger.info(f"Sales data loading complete in {time.time() - step_start_time:.2f}s.")
        print(f"      Loaded {len(sales)} sales records.")

        # --- Step 3: Load Inventory ---
        step_start_time = time.time()
        logger.info("[3/6] Loading current godown inventories...")
        print("[3/6] Loading current godown inventories...")
        inventory = load_current_stock() # Uses tqdm internally
        if not inventory: logger.error("No inventory data loaded."); print("      ERROR: No inventory data loaded. Exiting."); sys.exit(1)
        logger.info(f"Loaded stock for {len(inventory)} items in {time.time() - step_start_time:.2f}s.")
        print(f"      Loaded stock for {len(inventory)} items.")

        # --- Step 4: Generate Forecasts ---
        step_start_time = time.time()
        logger.info("[4/6] Generating demand forecasts...")
        print("[4/6] Generating demand forecasts...")
        try:
            forecasts = generate_forecasts(sales, inventory) # Uses tqdm internally
            logger.info(f"Generated forecasts for {len(forecasts)} items in {time.time() - step_start_time:.2f}s.")
            print(f"      Generated forecasts for {len(forecasts)} items.")
        except Exception as e: logger.exception("Error during forecast generation."); print(f"\nERROR generating forecasts: {e}"); sys.exit(1)

        # --- Step 5: Calculate Individual Location Replenishment Needs ---
        step_start_time = time.time()
        logger.info("[5/6] Calculating individual location replenishment needs...")
        print("[5/6] Calculating individual location replenishment needs...")
        try:
            orders_all_locations = calculate_replenishment(sales, forecasts, inventory)
            logger.info(f"Individual location calculations complete in {time.time() - step_start_time:.2f}s.")
            print(f"      Individual location calculations complete.")
            if orders_all_locations.empty: logger.warning("Individual replenishment calculation returned empty results."); print("      WARNING: No individual replenishment needs calculated."); sys.exit(0)
        except Exception as e: logger.exception("Error during individual replenishment calculation."); print(f"\nERROR calculating individual replenishment: {e}"); sys.exit(1)

        # --- Step 6: Aggregate Needs and Calculate Central Order for Main Location ---
        step_start_time = time.time()
        logger.info("[6/6] Aggregating system needs and calculating Main Location orders...")
        print("[6/6] Aggregating system needs and calculating Main Location orders...")
        try:
            # 1. Aggregate 'Ideal_Order_Qty' per Product
            logger.debug("Aggregating Ideal_Order_Qty per product...")
            system_needs = orders_all_locations.groupby('Product', as_index=False)['Ideal_Order_Qty'].sum()
            system_needs.rename(columns={'Ideal_Order_Qty': 'Total_System_Ideal_Requirement'}, inplace=True)
            logger.debug(f"Calculated Total System Ideal Requirement for {len(system_needs)} products.")

            # 2. Get Main Location's Data
            logger.debug("Extracting data for 'Main Location'...")
            main_loc_data = orders_all_locations[orders_all_locations['Godown'] == 'Main Location'].copy()
            main_loc_columns = ['Product', 'Godown', 'Category', 'Current_Stock', 'Inventory_Position', 'Avg_Daily_Sales', 'Lead_Time_Demand', 'Safety_Stock', 'Reorder_Point', 'Multiplier', 'Annual_Sales', 'Recent_Sales', 'Annualized_Recent', 'Seasonal_Avg_Monthly']
            main_loc_columns = [col for col in main_loc_columns if col in main_loc_data.columns]
            main_loc_data = main_loc_data[main_loc_columns]
            logger.debug(f"Extracted current data for {len(main_loc_data)} products at Main Location.")
            if main_loc_data.empty: logger.warning("'Main Location' not found in results."); print("      WARNING: 'Main Location' not found. No central orders generated."); sys.exit(0)

            # 3. Merge System Needs with Main Location Data
            logger.debug("Merging system needs with Main Location data...")
            central_order_calc = pd.merge(main_loc_data, system_needs, on='Product', how='left')
            central_order_calc['Total_System_Ideal_Requirement'].fillna(0, inplace=True)
            logger.debug("Merge complete.")

            # 4. Calculate Final Order Quantity for Main Location
            logger.debug("Calculating final order quantity for Main Location...")
            central_order_calc['Calc_Order_Qty'] = (central_order_calc['Total_System_Ideal_Requirement'] - central_order_calc['Inventory_Position']).clip(lower=0)
            needs_order_mask = (central_order_calc['Calc_Order_Qty'] > 0) & (central_order_calc['Total_System_Ideal_Requirement'] > 0)
            central_order_calc['Reorder_Qty'] = central_order_calc['Calc_Order_Qty']
            central_order_calc.loc[needs_order_mask, 'Reorder_Qty'] = central_order_calc.loc[needs_order_mask, 'Reorder_Qty'].clip(lower=MINIMUM_ORDER_QUANTITY)
            central_order_calc['Reorder_Qty'] = np.ceil(central_order_calc['Reorder_Qty']).astype(int)
            central_order_calc.loc[central_order_calc['Total_System_Ideal_Requirement'] <= 0, 'Reorder_Qty'] = 0
            central_order_calc.loc[central_order_calc['Category'] == "Non-Moving", 'Reorder_Qty'] = 0
            logger.debug("Final order quantities calculated.")

            # 5. Prepare Final Output DataFrame
            final_output_columns = [ 'Product', 'Godown', 'Category', 'Current_Stock', 'Inventory_Position', 'Total_System_Ideal_Requirement', 'Avg_Daily_Sales', 'Reorder_Point', 'Reorder_Qty', 'Lead_Time_Demand', 'Safety_Stock', 'Multiplier', 'Annual_Sales']
            final_output_columns = [col for col in final_output_columns if col in central_order_calc.columns]
            final_main_orders = central_order_calc[final_output_columns].copy()
            final_main_orders.sort_values(by=['Category', 'Product'], ascending=[True, True], inplace=True)

            # 6. Save Final Output (only rows with Reorder_Qty > 0)
            if final_main_orders.empty: logger.info("No orders generated for Main Location."); print("\nINFO: No replenishment orders required for 'Main Location'.")
            else:
                final_main_orders_to_order = final_main_orders[final_main_orders['Reorder_Qty'] > 0].copy()
                if final_main_orders_to_order.empty: logger.info("Calculations show no items require ordering for 'Main Location'."); print("\nINFO: No items require ordering for 'Main Location'.")
                else:
                    output_path = Path(OUTPUT_DIR) / 'replenishment_orders_main_location_aggregated.csv'
                    final_main_orders_to_order.to_csv(output_path, index=False, encoding='utf-8', float_format='%.2f')
                    items_to_order = len(final_main_orders_to_order); total_units = final_main_orders_to_order['Reorder_Qty'].sum()
                    success_msg = (f"\nSUCCESS: Generated AGGREGATED replenishment orders for Main Location.\nItems requiring restock (Qty > 0): {items_to_order}\nTotal units to order: {total_units:,}\nOutput saved to: {output_path}")
                    logger.info(success_msg); print(success_msg)

            logger.info(f"Aggregation and central order calculation complete in {time.time() - step_start_time:.2f}s.")

        except Exception as e: logger.exception("Error during aggregation or central order calculation."); print(f"\nERROR during aggregation: {e}"); sys.exit(1)

    except Exception as e: logger.exception("An unexpected error occurred in the main process."); print(f"\nFATAL ERROR: {e}"); sys.exit(1)
    finally: logger.info(f"Inventory Calculation Process finished in {time.time() - overall_start_time:.2f}s.")

if __name__ == "__main__":
    main()