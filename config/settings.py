# config/settings.py Configuration Parameters

from pathlib import Path

# -------------------------
# Path Configurations
# -------------------------
# Define base directory if needed for more complex structures, otherwise relative paths are fine.
# BASE_DIR = Path(__file__).resolve().parent.parent # Example if settings.py is in config/
# DATA_DIR = BASE_DIR / 'data'
# Or simpler relative paths if main script is run from the project root:
DATA_DIR = './data'
GODOWNS_DIR = f'{DATA_DIR}/godowns'
OUTPUT_DIR = './outputs/orders'
LOG_DIR = './logs' # Define Log directory

# --- Database ---
DATABASE_FILE = f'{DATA_DIR}/inventory_data.db' # SQLite database file path

# --- File Patterns ---
SALES_FILE_PATTERN = 'sales*.xml' # Pattern to find sales XML files in DATA_DIR
# Example patterns:
# 'sales_*.xml' -> sales_2023.xml, sales_history.xml etc.
# 'sales_????.xml' -> sales_2023.xml, sales_2024.xml (exactly 4 chars for year)

# -----------------------------------------------------------
# Performance Configuration
# -----------------------------------------------------------
# Number of worker threads/processes for parallel tasks.
# Set to 1 to disable parallelism for debugging (RECOMMENDED INITIALLY).
# Set higher (e.g., 4 or os.cpu_count()) to enable parallelism once sequential run is confirmed working.
MAX_WORKERS = 4 # Adjust based on your machine's cores and I/O capacity

# -------------------------
# Inventory Policy Values
# -------------------------
SUPPLIER_LEAD_TIME = 45             # Days for restocking (Forecast horizon should match this)
MINIMUM_ORDER_QUANTITY = 30         # Minimum units per order (cases/pallets)

# Annualized sales <= this value (units/year) qualifies as Non-Moving (if recent sales are also low)
NON_MOVING_THRESHOLD = 10

# -------------------------
# Forecasting Configuration
# -------------------------
# Prophet holiday configuration for UAE (2023-2025)
# NOTE: Islamic holiday dates are estimates and may vary slightly. Adjust windows as needed.
# Source: Mix of astronomical calendars and past announcements.

# --- 2023 ---
UAE_HOLIDAYS_2023 = [
    {"holiday": "Eid al-Fitr", "ds": "2023-04-21", "lower_window": -2, "upper_window": 2}, # Adjusted window
    {"holiday": "Eid al-Adha", "ds": "2023-06-28", "lower_window": -2, "upper_window": 2}, # Adjusted window (Includes Arafat Day + Eid)
    # {"holiday": "Hijri New Year", "ds": "2023-07-19", "lower_window": 0, "upper_window": 1}, # Removed
    {"holiday": "National Day", "ds": "2023-12-02", "lower_window": -1, "upper_window": 2} # Includes Commemoration Day
]

# --- 2024 ---
UAE_HOLIDAYS_2024 = [
    {"holiday": "Eid al-Fitr", "ds": "2024-04-10", "lower_window": -2, "upper_window": 2}, # Adjusted window
    {"holiday": "Eid al-Adha", "ds": "2024-06-16", "lower_window": -2, "upper_window": 2}, # Adjusted window (Includes Arafat Day + Eid)
    # {"holiday": "Hijri New Year", "ds": "2024-07-07", "lower_window": 0, "upper_window": 1}, # Removed
    {"holiday": "National Day", "ds": "2024-12-02", "lower_window": -1, "upper_window": 2} # Includes Commemoration Day
]

# --- 2025 ---
UAE_HOLIDAYS_2025 = [
    {"holiday": "Eid al-Fitr", "ds": "2025-03-30", "lower_window": -2, "upper_window": 2}, # Approximate, Adjusted window
    {"holiday": "Eid al-Adha", "ds": "2025-06-06", "lower_window": -2, "upper_window": 2}, # Approximate, Adjusted window (Includes Arafat Day + Eid)
    # {"holiday": "Hijri New Year", "ds": "2025-06-26", "lower_window": 0, "upper_window": 1}, # Approximate, Removed
    {"holiday": "National Day", "ds": "2025-12-02", "lower_window": -1, "upper_window": 2} # Includes Commemoration Day
]

# Combine all holidays into one list for Prophet
ALL_UAE_HOLIDAYS = UAE_HOLIDAYS_2023 + UAE_HOLIDAYS_2024 + UAE_HOLIDAYS_2025
# Ensure Prophet uses this combined list (handled in forecasting.py usage)

# Prophet model parameters (tune based on validation)
PROPHET_MODEL_CONFIG = {
    'changepoint_prior_scale': 0.3,      # Flexibility of trend changes
    'seasonality_mode': 'multiplicative',# Assumes seasonality scales with trend
    'holidays_prior_scale': 0.8          # Sensitivity to holidays (Adjust if needed)
    # Add other Prophet parameters if needed (e.g., yearly_seasonality, weekly_seasonality)
}

# Forecast horizon should typically match operational lead time
# Used by Prophet's make_future_dataframe
FORECAST_HORIZON_DAYS = SUPPLIER_LEAD_TIME  # In Days (redundant if always equal to lead time, but explicit)


# -----------------------------------------------------------
# Demand Classification Thresholds & Time Windows
# -----------------------------------------------------------

# --- Time Windows for Analysis ---
# How many years back to look for seasonal patterns.
# Since you have 2022, 2023, 2024 data and it's 2025, looking back 1, 2, 3 years is appropriate.
SEASONAL_YEARS_BACK = [1, 2, 3]  # Look at same period 1, 2, and 3 years ago

# How many months define a "seasonal period" (e.g., 1 for comparing single months, 3 for quarters)
SEASONAL_PERIOD_MONTHS = 3

# How many days of recent sales history to use for velocity and trending checks
RECENT_SALES_WINDOW_DAYS = 365

# --- Classification Thresholds ---
# Ratio: Recent Annualized Sales must be > TRENDING_THRESHOLD * Annual Sales to be "Trending"
TRENDING_THRESHOLD = 1.5 # Example: Recent velocity is 75% higher than long-term average

# Ratio: Avg sales in seasonal period must be > SEASONAL_THRESHOLD * Overall Avg Monthly Sales to be "Seasonal"
SEASONAL_THRESHOLD = 2.0 # Example: Sales in the season are 2.5x the normal monthly average

# Annualized sales thresholds (units/year) for volume categories
# Used primarily if not classified as Non-Moving, Trending, or Seasonal. Based on Annualized Recent sales.
ANNUAL_SALES_THRESHOLDS = {
    # 'Non-Moving' uses NON_MOVING_THRESHOLD
    'Slow-Moving': (NON_MOVING_THRESHOLD + 1, 100), # Items above non-moving but below medium
    'Medium-Demand': (101, 500),
    'Fast-Moving': (501, 1000),
    'Ultra-Fast-Moving': (1001, float('inf'))
}


# -----------------------------------------------------------
# Safety Stock Configuration
# -----------------------------------------------------------

# Base number of days of average sales to hold as safety stock.
# This is scaled by the category multiplier. Tune based on desired service level vs holding cost.
BASE_SAFETY_DAYS = 7

# Multipliers applied to BASE_SAFETY_DAYS for Safety Stock calculation based on category.
# Higher value = more safety stock days = higher service level (usually).
CATEGORY_MULTIPLIERS = {
    'Non-Moving': 0.0,       # No safety stock for non-moving items
    'Slow-Moving': 0.5,      # Less than base safety days
    'Medium-Demand': 0.75,   # Base safety days
    'Fast-Moving': 1.5,      # More safety stock
    'Ultra-Fast-Moving': 2.0,# Even more safety stock for critical high-runners
    'Trending': 1.75,        # Higher safety stock during trend periods
    'Seasonal': 1.25,        # Slightly more safety stock during seasonal peaks (adjust based on forecast accuracy)
    'New-Product': 1.0,      # Default to base safety days for new items
    'Default': 1.0           # Fallback multiplier if category somehow unassigned
}

# -------------------------
# Logging Configuration (Optional - can be done in code)
# -------------------------
# LOG_LEVEL = 'DEBUG' # Or 'INFO', 'WARNING', 'ERROR'
# LOG_FILE = f'{LOG_DIR}/inventory_app.log' # Example using LOG_DIR