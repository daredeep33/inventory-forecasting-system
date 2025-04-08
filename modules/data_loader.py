# modules/data_loader.py

import sqlite3
import pandas as pd
from lxml import etree
import re
from io import BytesIO
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
import concurrent.futures
import threading

# Import settings
from config.settings import (
    DATABASE_FILE, DATA_DIR, SALES_FILE_PATTERN, GODOWNS_DIR, MAX_WORKERS, LOG_DIR
)

# --- Configure Logging ---
# ... (logging setup as before) ...
log_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(module)s - %(message)s')
log_dir_path = Path(LOG_DIR); log_dir_path.mkdir(parents=True, exist_ok=True)
log_handler = logging.FileHandler(log_dir_path / 'data_loader.log', mode='a', encoding='utf-8')
log_handler.setFormatter(log_formatter)
logger = logging.getLogger(__name__)
if not logger.handlers: logger.addHandler(log_handler); logger.setLevel(logging.DEBUG)

# --- Thread Lock ---
stock_dict_lock = threading.Lock()

# --- Cleaning Functions (Ensure these are defined early) ---
def clean_product_name(raw_name: str) -> str:
    if not isinstance(raw_name, str): return "Unknown_Product"
    cleaned = ( re.sub(r'[^\w\s-]', '', raw_name).replace("_", " ").strip().upper() )
    return cleaned or "Unknown_Product"

def clean_godown_name(raw_name: str) -> str:
    if not isinstance(raw_name, str): return "Unknown_Godown"
    cleaned = ( raw_name.replace(".xml", "").replace("_", " ").strip().title() )
    return cleaned if cleaned else "Unknown_Godown"

# --- Database Functions ---
# ... (connect_db, initialize_db - Ensure UNIQUE constraint commented out for now) ...
def connect_db(): # Ensure it returns a connection or raises error
    try:
        db_path_obj = Path(DATABASE_FILE).resolve()
        logger.info(f"Connecting to DB at absolute path: {db_path_obj}")
        conn = sqlite3.connect(db_path_obj, detect_types=sqlite3.PARSE_DECLTYPES, timeout=20.0)
        conn.row_factory = sqlite3.Row
        logger.debug(f"Connected successfully to: {db_path_obj}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error for {DATABASE_FILE}: {e}", exc_info=True)
        raise

def initialize_db():
    conn = None
    try:
        conn = connect_db()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sales_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_date TIMESTAMP NOT NULL, product TEXT NOT NULL,
                    godown TEXT NOT NULL, quantity REAL NOT NULL, source_file TEXT,
                    load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    /* TEMPORARILY REMOVED for debugging:
                    , UNIQUE(transaction_date, product, godown, quantity, source_file)
                    */
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_transactions (transaction_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_product_godown ON sales_transactions (product, godown)")
        logger.info("Database initialized successfully (UNIQUE constraint TEMPORARILY REMOVED).")
    except Exception as e: logger.error(f"Database initialization error: {e}", exc_info=True)
    finally:
        if conn: conn.close()

# ... (get_latest_transaction_date remains unchanged) ...
def get_latest_transaction_date(conn, source_file=None):
    try:
        cursor = conn.cursor(); query = "SELECT MAX(transaction_date) FROM sales_transactions"
        params = []; cursor.execute(query, params); result = cursor.fetchone()
        latest_date = result[0] if result else None
        if latest_date:
            if isinstance(latest_date, str): latest_date = pd.to_datetime(latest_date).tz_localize(None)
            elif isinstance(latest_date, datetime): latest_date = latest_date.replace(tzinfo=None)
            logger.debug(f"Latest transaction date found: {latest_date}"); return latest_date
        else: logger.debug("No previous transactions found."); return None
    except Exception as e: logger.error(f"Error fetching latest transaction date: {e}"); return None


# --- XML Processing Functions ---
# process_sales_xml_file USES the cleaning functions defined above
def process_sales_xml_file(xml_path: Path) -> list:
    """ Parses a single sales XML file (loading ALL data). """
    transactions = []
    file_name = xml_path.name
    logger.debug(f"[{file_name}] Starting processing (loading ALL data)...")
    try:
        with open(xml_path, 'rb') as f: raw_data = f.read()
        tree = etree.parse(BytesIO(raw_data), etree.XMLParser(recover=True))
        vouchers = tree.xpath('//TALLYMESSAGE/VOUCHER')
        logger.debug(f"[{file_name}] Found {len(vouchers)} vouchers.")
        new_transaction_count = 0
        for voucher in vouchers:
            date_str = voucher.xpath('string(DATE)')
            try: date = pd.to_datetime(date_str, format='%Y%m%d').tz_localize(None)
            except: logger.warning(f"[{file_name}] Skipped invalid date '{date_str}'"); continue
            # No incremental check here for now
            for entry in voucher.xpath('.//ALLINVENTORYENTRIES.LIST'):
                # Calls cleaning functions defined above
                product = clean_product_name(entry.xpath('string(STOCKITEMNAME)'))
                godown_nodes = entry.xpath('.//BATCHALLOCATIONS.LIST/GODOWNNAME/text()')
                if godown_nodes: godown = clean_godown_name(godown_nodes[0])
                else: cost_centre_nodes = voucher.xpath('string(COSTCENTRENAME)'); godown = clean_godown_name(cost_centre_nodes) if cost_centre_nodes else "Unknown_Godown"
                qty_raw = entry.xpath('string(ACTUALQTY)').strip()
                if not qty_raw: continue
                try:
                    qty_match = re.search(r'-?(\d+\.?\d*|\.\d+)', qty_raw)
                    if qty_match:
                        qty = float(qty_match.group(1))
                        if qty > 0: transactions.append({'transaction_date': date, 'product': product, 'godown': godown, 'quantity': qty, 'source_file': file_name}); new_transaction_count += 1
                    else: logger.warning(f"[{file_name}] Invalid quantity format '{qty_raw}' for {product}")
                except Exception as e: logger.warning(f"[{file_name}] Error processing quantity '{qty_raw}' for {product}: {e}", exc_info=False)
        logger.info(f"[{file_name}] Finished processing. Found {new_transaction_count} transactions.")
        if transactions: logger.debug(f"[{file_name}] First 5 parsed transactions sample: {transactions[:5]}")
        return transactions
    except Exception as e: logger.error(f"[{file_name}] FAILED to process: {e}", exc_info=True); return []


# --- Helper function for stock processing (MUST BE DEFINED *BEFORE* load_current_stock) ---
def _process_stock_file(xml_path: Path, shared_stock_dict: dict):
    """ Parses one stock file and updates the shared dictionary (using lock). """
    file_name = xml_path.name
    logger.debug(f"[{file_name}] Processing stock file...")
    try:
        with open(xml_path, 'rb') as f: raw_data = f.read()
        tree = etree.parse(BytesIO(raw_data), parser=etree.XMLParser(recover=True))
        godown_raw_name = xml_path.stem
        godown = clean_godown_name(godown_raw_name) # Uses global cleaning func
        products = tree.xpath('//DSPACCNAME/DSPDISPNAME/text()')
        quantities = tree.xpath('//DSPSTKINFO//DSPCLQTY/text()')
        min_len = min(len(products), len(quantities))
        if len(products) != len(quantities): logger.warning(f"[{file_name}] Mismatch product/qty count. Using {min_len}.")
        with stock_dict_lock:
            local_updates = 0
            for product_raw, qty_raw in zip(products[:min_len], quantities[:min_len]):
                product = clean_product_name(product_raw) # Uses global cleaning func
                try:
                    qty_match = re.search(r'-?\d+\.?\d*', qty_raw)
                    qty = max(0, float(qty_match.group())) if qty_match else 0.0
                    key = (product, godown)
                    shared_stock_dict[key] = shared_stock_dict.get(key, 0.0) + qty
                    local_updates += 1
                except Exception as e: logger.warning(f"[{file_name}] Invalid qty '{qty_raw}' for {product}: {str(e)}")
            logger.debug(f"[{file_name}] Updated stock for {local_updates} items into shared dict.")
    except Exception as e: logger.error(f"[{file_name}] Error processing stock file: {e}", exc_info=True)


# --- Main Data Loading Functions ---
def update_sales_from_xml():
    """ Simplified: Scans XMLs, processes ALL transactions, inserts using to_sql, verifies count. """
    # ... (Keep the simplified version from the previous step, using to_sql, no incremental check, verify count) ...
    conn = None
    try:
        initialize_db()
        conn = connect_db()
        all_transactions = []
        files_processed_count = 0
        sales_dir = Path(DATA_DIR); xml_files = list(sales_dir.glob(SALES_FILE_PATTERN))
        logger.info(f"Found {len(xml_files)} sales XML file(s).")
        if not xml_files: logger.warning("No sales XML files found."); return

        num_workers = MAX_WORKERS if MAX_WORKERS and MAX_WORKERS > 0 else 1
        logger.info(f"Processing XML files using up to {num_workers} workers (loading ALL data)...")
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for xml_path in xml_files: futures.append(executor.submit(process_sales_xml_file, xml_path))
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parsing Sales XMLs (All Data)", unit="file", leave=True):
                try:
                    result = future.result()
                    if result: all_transactions.extend(result); files_processed_count += 1
                except Exception as exc: logger.error(f"Exception retrieving result: {exc}", exc_info=True)
            logger.info(f"Finished processing {len(futures)} sales futures.")

        if not all_transactions: logger.warning("No transactions parsed from any XML file."); return
        logger.info(f"Found {len(all_transactions)} total transactions from {files_processed_count} file(s).")
        df_all = pd.DataFrame(all_transactions)
        df_all['transaction_date'] = pd.to_datetime(df_all['transaction_date']); df_all['quantity'] = pd.to_numeric(df_all['quantity'])
        unique_cols = ['transaction_date', 'product', 'godown', 'quantity', 'source_file']
        original_rows = len(df_all)
        df_all.drop_duplicates(subset=unique_cols, keep='first', inplace=True)
        if original_rows > len(df_all): logger.warning(f"Dropped {original_rows - len(df_all)} duplicate rows from XML data before insertion.")
        logger.info(f"DataFrame shape before insert attempt: {df_all.shape}")
        if df_all.empty: logger.warning("DataFrame is empty after processing, skipping DB insert."); return

        insert_success = False; rows_before = 0; rows_after = 0
        try:
            cursor_before = conn.cursor(); cursor_before.execute("SELECT COUNT(*) FROM sales_transactions"); rows_before = cursor_before.fetchone()[0]
            logger.info(f"Rows in DB BEFORE insert attempt: {rows_before}")
            logger.debug("Entering 'with conn' block for to_sql (UNIQUE constraint disabled)...")
            with conn: df_all.to_sql('sales_transactions', conn, if_exists='append', index=False, chunksize=1000)
            insert_success = True; logger.info(f"Successfully COMPLETED 'with conn' block for to_sql.")
        except Exception as e: logger.error(f"Database error during bulk insert: {e}", exc_info=True); insert_success = False

        if insert_success:
             logger.info("Verifying row count after insertion attempt...")
             try:
                 cursor_after = conn.cursor(); cursor_after.execute("SELECT COUNT(*) FROM sales_transactions"); rows_after = cursor_after.fetchone()[0]
                 logger.info(f"Rows in DB AFTER insert attempt: {rows_after}")
                 if rows_after > rows_before: logger.info(f"Verification: {rows_after - rows_before} new rows added.")
                 elif rows_after == rows_before: logger.warning("Verification: Row count unchanged.")
                 else: logger.error("Verification FAILED: Row count DECREASED?")
             except Exception as verify_e: logger.error(f"Error during verification count: {verify_e}", exc_info=True)
        else: logger.error("Insertion attempt FAILED. Verification skipped.")
    except Exception as e: logger.error(f"Error in update_sales_from_xml main block: {e}", exc_info=True)
    finally:
        if conn: logger.debug("Closing DB connection in update_sales_from_xml."); conn.close()


def get_sales_data_from_db(start_date=None, end_date=None) -> pd.DataFrame:
    """ Simplified: Loads ALL sales transaction data from the database. """
    conn = None
    try:
        conn = connect_db()
        if conn is None: logger.error("Failed to get DB connection."); return pd.DataFrame(columns=['Date', 'Product', 'Godown', 'Quantity'])
        # --- Simplified Query: Load ALL data ---
        query = "SELECT transaction_date, product, godown, quantity FROM sales_transactions ORDER BY transaction_date"
        params = []
        logger.info(f"Querying ALL sales data from DB: {query}")
        # --- End Simplification ---
        sales_df = pd.read_sql_query(query, conn, params=params, parse_dates=['transaction_date'])
        logger.info(f"Loaded {len(sales_df)} sales records from database.")
        if not sales_df.empty:
            sales_df = sales_df.rename(columns={'transaction_date': 'Date', 'product': 'Product','godown': 'Godown', 'quantity': 'Quantity'})
            sales_df['Date'] = pd.to_datetime(sales_df['Date']).dt.tz_localize(None)
            sales_df['Product'] = sales_df['Product'].astype(str); sales_df['Godown'] = sales_df['Godown'].astype(str)
            sales_df['Quantity'] = pd.to_numeric(sales_df['Quantity'])
            logger.debug(f"Sales data DataFrame shape after load/rename: {sales_df.shape}")
        else: sales_df = pd.DataFrame(columns=['Date', 'Product', 'Godown', 'Quantity'])
        return sales_df
    except Exception as e: logger.error(f"Failed to load sales data: {e}", exc_info=True); return pd.DataFrame(columns=['Date', 'Product', 'Godown', 'Quantity'])
    finally:
        if conn: logger.debug("Closing DB connection in get_sales_data_from_db."); conn.close()


# --- load_current_stock function MUST come AFTER _process_stock_file ---
def load_current_stock() -> dict:
    """ Loads current stock from XML files in parallel using ThreadPoolExecutor. """
    stock = {}
    godown_dir = Path(GODOWNS_DIR)
    try: xml_stock_files = list(godown_dir.glob('*.xml'))
    except Exception as e: logger.error(f"Error accessing godowns directory {godown_dir}: {e}", exc_info=True); return {}
    num_workers = MAX_WORKERS if MAX_WORKERS and MAX_WORKERS > 0 else 1
    logger.info(f"Loading current stock from {len(xml_stock_files)} files in {godown_dir} using up to {num_workers} workers...")
    if not xml_stock_files: logger.warning("No stock XML files found."); return {}
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for xml_path in xml_stock_files:
            # Calls the helper function defined ABOVE
            futures.append(executor.submit(_process_stock_file, xml_path, stock))
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Loading Stock XMLs", unit="file", leave=True):
            try: future.result()
            except Exception as exc: logger.error(f"Exception from stock loading worker: {exc}", exc_info=True)
    logger.info(f"Finished stock loading. Loaded stock for {len(stock)} unique Product/Godown combinations.")
    return stock