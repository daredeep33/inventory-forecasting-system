# modules/forecasting.py
import pandas as pd
from prophet import Prophet
import logging
# from logging.handlers import RotatingFileHandler # Handlers configured elsewhere
import concurrent.futures # Import concurrent futures
from functools import partial # For passing fixed arguments to map function
from tqdm import tqdm # For progress bar
import numpy as np
from pathlib import Path
import sys
import os # For devnull used in MuteProphetOutput

# Import settings
from config.settings import (
    SUPPLIER_LEAD_TIME,
    MINIMUM_ORDER_QUANTITY,
    # UAE_HOLIDAYS_2024, # No longer used directly
    ALL_UAE_HOLIDAYS,   # Import the combined list
    PROPHET_MODEL_CONFIG,
    MAX_WORKERS, # Import MAX_WORKERS
    LOG_DIR      # Import LOG_DIR
)

# --- Configure Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(module)s - %(message)s')
# Ensure logs directory exists
log_dir_path = Path(LOG_DIR)
log_dir_path.mkdir(parents=True, exist_ok=True)
log_handler = logging.FileHandler(log_dir_path / 'inventory_forecast.log', mode='a', encoding='utf-8')
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)
# Silence prophet/cmdstanpy INFO messages if desired, as they can be very verbose
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


# Helper context manager to suppress stdout/stderr during Prophet fitting
# Useful especially in parallel processing to avoid cluttered console output
class MuteProphetOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        # Redirect stdout/stderr to null device
        try:
            self._devnull_out = open(os.devnull, 'w')
            self._devnull_err = open(os.devnull, 'w')
            sys.stdout = self._devnull_out
            sys.stderr = self._devnull_err
        except Exception: # Fallback if os.devnull fails
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()


    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the redirected streams and restore original stdout/stderr
        try:
            if hasattr(self, '_devnull_out') and self._devnull_out:
                 self._devnull_out.close()
            if hasattr(self, '_devnull_err') and self._devnull_err:
                 self._devnull_err.close()
        except Exception as e:
             logger.error(f"Error closing muted streams: {e}")
        finally:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

# --- Helper Function for Parallel Forecasting ---
def _run_prophet_for_item(item_key, sales_data_grouped, holidays_df, config, lead_time, min_qty):
    """
    Fits Prophet model and forecasts for a single (product, godown) item.
    Designed to be called by ProcessPoolExecutor.map. Handles errors gracefully.

    Args:
        item_key (tuple): (product, godown)
        sales_data_grouped (pd.DataFrameGroupBy): Sales data grouped by Product, Godown.
        holidays_df (pd.DataFrame): DataFrame of holidays for Prophet.
        config (dict): Prophet model configuration.
        lead_time (int): Forecast horizon in days.
        min_qty (int): Minimum order quantity (used as fallback/floor).

    Returns:
        dict: {'Product': product, 'Godown': godown, 'Prophet_Forecast': forecast_qty}
    """
    product, godown = item_key
    forecast_qty = -1.0 # Use -1 to clearly see if it gets updated or defaults later
    default_forecast_reason = "Initialization" # Start with a reason

    try:
        # Get data for the specific item using the grouped object
        filtered = sales_data_grouped.get_group(item_key).copy()
        prophet_item_df = filtered[['Date', 'Quantity']].rename(
            columns={'Date': 'ds', 'Quantity': 'y'}
        ).sort_values('ds')

        # Check for sufficient VALID data points (not just rows)
        valid_data_points = len(prophet_item_df.dropna(subset=['y']))
        if valid_data_points < 2:
             default_forecast_reason = f"Insufficient data points ({valid_data_points})"
             logger.warning(f"{default_forecast_reason} for {product} at {godown}.")
             forecast_qty = float(min_qty) # Apply default immediately
        else:
            # Proceed with fitting only if sufficient data
            try:
                model = Prophet(**config, holidays=holidays_df, weekly_seasonality=True, yearly_seasonality=True)
                with MuteProphetOutput():
                    model.fit(prophet_item_df)
                future = model.make_future_dataframe(periods=lead_time, freq='D', include_history=False)
                forecast = model.predict(future)
                forecast_sum = forecast['yhat'].clip(lower=0).sum()

                # --- REVISED Apply floor logic ---
                rounded_sum = np.round(forecast_sum)
                if rounded_sum <= 0:
                    logger.debug(f"Prophet predicted <= 0 ({rounded_sum}) for {product} at {godown}. Setting forecast to 0.")
                    forecast_qty = 0.0
                else:
                    forecast_qty = max(rounded_sum, float(min_qty))
                    logger.debug(f"Prophet forecast successful for {product} at {godown}: {forecast_qty} (Rounded sum: {rounded_sum})")
                # --- END REVISED ---

            except Exception as model_exc:
                default_forecast_reason = f"Prophet model fit/predict failed: {model_exc}"
                logger.error(f"{default_forecast_reason} for {product} at {godown}", exc_info=False)
                forecast_qty = float(min_qty) # Apply default on error

    except KeyError:
        default_forecast_reason = "No sales data found in grouped data"
        logger.info(f"{default_forecast_reason} for inventory item {product} at {godown}.")
        forecast_qty = float(min_qty) # Apply default
    except Exception as outer_exc:
        default_forecast_reason = f"Unexpected error: {outer_exc}"
        logger.error(f"{default_forecast_reason} processing forecast for {product} at {godown}", exc_info=True)
        forecast_qty = float(min_qty) # Apply default

    # --- FINAL CHECK: If forecast_qty is still initial -1, apply default ---
    if forecast_qty == -1.0:
         logger.warning(f"Forecast quantity was not updated for {product} at {godown}, reason: {default_forecast_reason}. Applying default forecast ({min_qty}).")
         forecast_qty = float(min_qty)

    return {
        'Product': product,
        'Godown': godown,
        'Prophet_Forecast': float(forecast_qty)
    }


# --- Main Forecasting Function with Parallelism and Exclusion ---
def generate_forecasts(sales_data: pd.DataFrame, inventory: dict) -> pd.DataFrame:
    """
    Generates forecasts using Prophet for each relevant product-godown pair present in inventory,
    excluding specified item types (e.g., TESTER, BAG) and utilizing parallel processing.
    """
    num_items_total = len(inventory)
    num_workers = MAX_WORKERS if MAX_WORKERS and MAX_WORKERS > 0 else 1
    logger.info(f"Generating forecasts for relevant items among {num_items_total} inventory items using up to {num_workers} processes...")

    forecast_results = [] # List to store results for all items

    # --- Define Keywords for Exclusion ---
    EXCLUSION_KEYWORDS = ["TESTER", " BAG"] # Uppercase, space before BAG
    logger.info(f"Excluding items containing keywords: {EXCLUSION_KEYWORDS} from Prophet forecasting.")
    # ---

    # Prepare shared holiday DataFrame ONCE
    try:
        all_holidays_df = pd.DataFrame(ALL_UAE_HOLIDAYS)
        all_holidays_df['ds'] = pd.to_datetime(all_holidays_df['ds'])
        logger.debug(f"Prepared holiday DataFrame with {len(all_holidays_df)} entries.")
    except Exception as e: logger.error(f"Failed to create holidays DataFrame: {e}. Proceeding without holidays.", exc_info=True); all_holidays_df = None

    # Clean and filter inventory keys
    inventory_items_keys = list(inventory.keys())
    items_to_forecast_keys = []
    excluded_items_keys = []

    for p, g in inventory_items_keys:
        if isinstance(p, str) and p.strip() and isinstance(g, str) and g.strip():
            product_upper = p.strip().upper(); godown_title = g.strip().title()
            item_key = (product_upper, godown_title)
            is_excluded = any(keyword in product_upper for keyword in EXCLUSION_KEYWORDS)
            if is_excluded: excluded_items_keys.append(item_key)
            else: items_to_forecast_keys.append(item_key)
        else: logger.warning(f"Skipping invalid inventory key: Product='{p}', Godown='{g}'")

    items_to_forecast_keys = sorted(list(set(items_to_forecast_keys)))
    excluded_items_keys = sorted(list(set(excluded_items_keys)))
    num_to_forecast = len(items_to_forecast_keys); num_excluded = len(excluded_items_keys)
    logger.info(f"Identified {num_to_forecast} items for Prophet forecasting and {num_excluded} excluded items.")

    # Handle Excluded Items
    for product, godown in excluded_items_keys:
        logger.debug(f"Assigning default forecast ({MINIMUM_ORDER_QUANTITY}) to excluded item: {product} at {godown}")
        forecast_results.append({'Product': product, 'Godown': godown, 'Prophet_Forecast': float(MINIMUM_ORDER_QUANTITY)})

    # Proceed with Forecasting only if needed
    if num_to_forecast > 0:
        if sales_data.empty or not all(col in sales_data.columns for col in ['Date', 'Product', 'Godown', 'Quantity']):
            logger.warning("Sales data is empty or invalid. Forecasting items will receive default minimum quantity.")
            for product, godown in items_to_forecast_keys:
                 forecast_results.append({'Product': product, 'Godown': godown, 'Prophet_Forecast': float(MINIMUM_ORDER_QUANTITY)})
            sales_data_grouped = None # Signal no valid data
        else:
            # Group sales data once
            try:
                sales_data['Product'] = sales_data['Product'].astype(str); sales_data['Godown'] = sales_data['Godown'].astype(str)
                sales_data['Date'] = pd.to_datetime(sales_data['Date']); sales_data['Quantity'] = pd.to_numeric(sales_data['Quantity'])
                sales_data_grouped = sales_data.groupby(['Product', 'Godown'])
                logger.debug(f"Sales data grouped successfully into {sales_data_grouped.ngroups} groups for forecasting.")
            except KeyError as e:
                 logger.error(f"Error grouping sales data. Missing columns? Error: {e}", exc_info=True)
                 logger.warning("Falling back to default forecasts for non-excluded items due to grouping error.")
                 for product, godown in items_to_forecast_keys: forecast_results.append({'Product': product, 'Godown': godown, 'Prophet_Forecast': float(MINIMUM_ORDER_QUANTITY)})
                 sales_data_grouped = None # Signal error

        if sales_data_grouped is not None:
            # Parallel Processing Setup
            task_func = partial(_run_prophet_for_item, sales_data_grouped=sales_data_grouped, holidays_df=all_holidays_df, config=PROPHET_MODEL_CONFIG, lead_time=SUPPLIER_LEAD_TIME, min_qty=MINIMUM_ORDER_QUANTITY)
            prophet_forecasts = []

            if num_workers == 1:
                 logger.info("Running Prophet forecast generation sequentially (MAX_WORKERS=1).")
                 prophet_forecasts = [task_func(item_key) for item_key in tqdm(items_to_forecast_keys, desc="Generating Forecasts (Sequential)", unit="item", leave=True)]
            else:
                 logger.info(f"Using ProcessPoolExecutor with max_workers={num_workers} for Prophet forecasts.")
                 with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                     try:
                         results_iterator = executor.map(task_func, items_to_forecast_keys)
                         prophet_forecasts = list(tqdm(results_iterator, total=len(items_to_forecast_keys), desc="Generating Forecasts (Parallel)", unit="item", leave=True))
                     except Exception as pool_exc:
                          logger.error(f"Error during parallel forecast execution: {pool_exc}", exc_info=True)
                          logger.warning("Parallel execution failed, falling back to default forecasts for remaining items.")
                          prophet_forecasts = [{'Product': p, 'Godown': g, 'Prophet_Forecast': float(MINIMUM_ORDER_QUANTITY)} for p, g in items_to_forecast_keys]
            forecast_results.extend(prophet_forecasts)

    # Combine and Finalize
    logger.info(f"Forecast generation process complete. Total results generated: {len(forecast_results)}.")
    if not forecast_results:
        logger.warning("Forecast results list is empty."); return pd.DataFrame(columns=['Product', 'Godown', 'Prophet_Forecast'])

    forecast_df = pd.DataFrame(forecast_results)

    # Final validation merge to ensure all original inventory items are present
    all_inventory_keys_df = pd.DataFrame(items_to_forecast_keys + excluded_items_keys, columns=['Product', 'Godown']).drop_duplicates()
    final_forecast_df = pd.merge(all_inventory_keys_df, forecast_df, on=['Product', 'Godown'], how='left')
    final_forecast_df['Prophet_Forecast'].fillna(float(MINIMUM_ORDER_QUANTITY), inplace=True) # Fill any potential NaNs
    final_forecast_df['Prophet_Forecast'] = final_forecast_df['Prophet_Forecast'].astype(float)

    logger.debug(f"Final forecast DataFrame shape: {final_forecast_df.shape}")
    logger.debug(f"Sample Forecasts (including defaults):\n{final_forecast_df.head().to_string()}")
    return final_forecast_df