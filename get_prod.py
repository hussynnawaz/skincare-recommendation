from flask import Flask, jsonify, request
import pandas as pd
import re

app = Flask(__name__)

# Load the CSV data
file_path = 'products.csv'
df = pd.read_csv(file_path)

# Convert the PKR column to a numeric type
def convert_pkr_to_numeric(pkr_value):
    if isinstance(pkr_value, str):
        # Extract numeric part and remove commas
        numeric_value = re.sub(r'[^\d.]', '', pkr_value)
        return float(numeric_value)
    return 0

df['PKR_numeric'] = df['PKR'].apply(convert_pkr_to_numeric)

@app.route('/products/', methods=['GET'])
def get_products():
    product_type = request.args.get('product_type', default=None, type=str)

    filtered_df = df.copy()

    # Apply filters
    if product_type:
        filtered_df = filtered_df[filtered_df['product_type'].str.contains(product_type, case=False)]

    # Convert to dictionary
    products = filtered_df[['product_name', 'product_type', 'skintype', 'PKR','description','notable_effects','brand']].to_dict(orient='records')
    
    return jsonify({"products": products})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
