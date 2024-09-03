# main.py

from fastapi import FastAPI, Query
from typing import List, Optional
import pandas as pd
import re

app = FastAPI()

# Load the CSV data
file_path = '/mnt/data/export_skincare 1.csv'
df = pd.read_csv(file_path)

# Convert the PKR column to a numeric type
def convert_pkr_to_numeric(pkr_value):
    if isinstance(pkr_value, str):
        # Extract numeric part and remove commas
        numeric_value = re.sub(r'[^\d.]', '', pkr_value)
        return float(numeric_value)
    return 0

df['PKR_numeric'] = df['PKR'].apply(convert_pkr_to_numeric)

@app.get("/products/")
def get_products(
    product_type: Optional[str] = Query(None, description="Filter by product type"),
    skintype: Optional[str] = Query(None, description="Filter by skin type"),
    max_price: Optional[float] = Query(None, description="Filter by maximum PKR"),
    min_price: Optional[float] = Query(None, description="Filter by minimum PKR"),
):
    filtered_df = df.copy()

    # Apply filters
    if product_type:
        filtered_df = filtered_df[filtered_df['product_type'].str.contains(product_type, case=False)]

    if skintype:
        filtered_df = filtered_df[filtered_df['skintype'].str.contains(skintype, case=False)]

    if max_price is not None:
        filtered_df = filtered_df[filtered_df['PKR_numeric'] <= max_price]
        
    if min_price is not None:
        filtered_df = filtered_df[filtered_df['PKR_numeric'] >= min_price]

    # Convert to dictionary
    products = filtered_df[['product_name', 'product_type', 'skintype', 'PKR']].to_dict(orient='records')
    return {"products": products}
