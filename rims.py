import streamlit as st
import modules as mod
from modules import pd

# Title for the Streamlit application
st.title("Retail Inventory Management System")

# File uploader for dataset
upload_file = st.file_uploader("Upload sales data (CSV). Necessary Attributes: InvoiceNo, StockCode, Stock Descriptions, Quantity, InvoiceDate", type='csv')

if upload_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(upload_file, low_memory=True)
    
    # Display the first few rows of the dataset
    st.subheader("Data Preview:")
    st.write(data)

    # Data preprocessing using a custom module
    data_filtered = mod.data_preprocessing(data)

    # Sample 50% of the data for further data mining
    data_sample = data_filtered.sample(frac=0.5, random_state=2)

    # Generate association rules and lookup table for product descriptions
    rules = mod.basket(data_sample)
    product_dict = mod.lookup_table(data_sample)

    # Function to map product codes to product descriptions
    def map_product_descriptions(product_set, product_dict):
        return set(product_dict[code] for code in product_set)

    # Adding descriptions to antecedents and consequents in the rules
    rules['Antecedent_descriptions'] = rules['antecedents'].apply(lambda x: map_product_descriptions(x, product_dict))
    rules['Consequent_descriptions'] = rules['consequents'].apply(lambda x: map_product_descriptions(x, product_dict))

    st.subheader("Product Associations:")
    rules = pd.DataFrame(rules)

    # Convert sets of antecedents and consequents into more readable format
    rules['antecedents'] = rules['antecedents'].apply(lambda x: set(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: set(x))

    # Rename columns for clarity
    rules.rename(columns={'antecedents': 'Antecedents', 'consequents': 'Consequents', 'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'}, inplace=True)

    # Function to create a human-readable rule in the format: "A, B -> C"
    def create_rule(row):
        antecedent_str = ', '.join(row['Antecedents'])
        consequent_str = ', '.join(row['Consequents'])
        return f"{antecedent_str} -> {consequent_str}"

    # Apply the create_rule function to generate the rules
    rules['Rules'] = rules.apply(create_rule, axis=1)

    # Display the rules on the Streamlit app, sorted by Lift
    st.write(rules[['Rules', 'Antecedents', 'Antecedent_descriptions', 'Consequents', 'Consequent_descriptions', 'Support', 'Confidence', 'Lift']].sort_values(by='Lift', ascending=False))

    # Chart selection for displaying the rules
    st.subheader("View Charts")
    chart_type = st.selectbox("Select Chart to Display:", ["None", "Barplots"])

    # Display a bar plot for association rules
    if chart_type == "Barplots":
        st.pyplot(mod.view_association_charts(rules))
    
    # Display a heatmap (Placeholder example for heatmap)
    # elif chart_type == "Heatmap":
    #     fig, ax = plt.subplots()
    #     sns.heatmap(rules['Lift'].corr(), annot=True, cmap="coolwarm", ax=ax)  # Placeholder heatmap
    #     st.pyplot(fig)

    # Convert 'InvoiceDate' from object to datetime type
    data_sample['InvoiceDate'] = pd.to_datetime(data_sample['InvoiceDate'])

    # Group stock items by daily sales
    daily_sales = data_sample.groupby(['StockCode', data_sample['InvoiceDate'].dt.date])['Quantity'].sum().reset_index()
    daily_sales.rename(columns={'InvoiceDate': 'Date'}, inplace=True)

    # Identify products involved in association rules (antecedents or consequents)
    items = set(rules['Antecedents'].explode()).union(set(rules['Consequents'].explode()))

    # Filter daily sales to include only items from association rules
    filtered_daily_sales = daily_sales[daily_sales['StockCode'].isin(items)]

    # Forecasting sales for each product
    forecasts = {}
    for item in items:
        forecasts[item] = mod.forecast_demand(item, filtered_daily_sales, days=30)

    # Convert forecasts dictionary into a DataFrame and keep only 30 days of forecasts
    forecasts_df = pd.DataFrame.from_dict(forecasts, orient='index')
    forecasts_df = forecasts_df.iloc[:, :30]  # Restrict to 30 days of forecasts
    forecasts_df.columns = [f'Day_{i+1}' for i in range(forecasts_df.shape[1])]

    # Display the forecasted sales for the next 30 days
    st.subheader("Forecast for next 30 days")
    st.write("Based on Daily Sales")
    st.dataframe(forecasts_df)

    # Calculate reorder points for each item
    reorder_points = {}
    for item in forecasts:
        reorder_points[item] = mod.calculate_reorder_point(forecasts[item], mod.lead_time, mod.safety_stock)

    # Convert reorder points dictionary into a DataFrame
    reorder_df = pd.DataFrame.from_dict(reorder_points, orient='index')

    # Example inventory data
    current_inventory = {item: 100 for item in reorder_points}

    # Check replenishment need for each item
    df = pd.DataFrame()
    for item in items:
        curr_inv_reorder = mod.check_replenishment(item, forecasts[item], current_inventory)
        new_row = pd.DataFrame([curr_inv_reorder])  # Wrapping in a list to convert dict to row
        df = pd.concat([df, new_row], ignore_index=True)

    # Display the reorder status for each product
    st.subheader("Reorder Status")
    st.write(df)

    # Calculate total spend for customer segmentation
    data_sample['TotalSpend'] = data_sample['Quantity'] * data_sample['UnitPrice']

    # Perform customer segmentation and display it
    st.subheader("Customer Segmentation")
    st.write(mod.customer_seg(data_sample))