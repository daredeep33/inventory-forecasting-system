import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import CrostonOptimized
from config.settings import FORECAST_HORIZON_DAYS

def generate_forecasts(sales_data: pd.DataFrame, inventory: dict) -> pd.DataFrame:
    # Include 'Godown' in the unique_id and data prep
    sales_clean = (
        sales_data[['Date', 'Product', 'Godown', 'Quantity']]
        .assign(unique_id=lambda x: x['Product'] + '_' + x['Godown'])
        .rename(columns={'Date': 'ds', 'Quantity': 'y'})
    )
    
    # Create unique_id list from inventory keys
    unique_pairs = list(inventory.keys())
    unique_ids = [f"{p}_{g}" for (p, g) in unique_pairs] if unique_pairs else []
    
    # Build date grid for all Product-Godown pairs
    min_date = (sales_clean['ds'].min() - pd.DateOffset(months=1) 
                if not sales_clean.empty else pd.Timestamp.today())
    
    date_range = pd.date_range(
        start=min_date, 
        periods=FORECAST_HORIZON_DAYS + 30, 
        freq='D'
    )
    
    grid = pd.MultiIndex.from_product(
        [date_range, unique_ids],
        names=['ds', 'unique_id']
    ).to_frame(index=False)
    
    # Merge with historical sales
    model_df = (
        pd.merge(grid, sales_clean, on=['ds', 'unique_id'], how='left')
        .fillna({'y': 0})
        .sort_values(['unique_id', 'ds'])
    )
    
    # Forecast and split unique_id back into Product/Godown
    sf = StatsForecast(models=[CrostonOptimized()], freq='D', n_jobs=-1)
    forecasts = sf.forecast(df=model_df, h=FORECAST_HORIZON_DAYS)
    
    forecasts[['Product', 'Godown']] = (
        forecasts['unique_id']
        .str.split('_', n=1, expand=True)
    )
    
    return forecasts[['Product', 'Godown', 'ds', 'CrostonOptimized']]