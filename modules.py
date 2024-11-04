import pandas as pd
# import math
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

def data_preprocessing(data):
    data_filtered = data.dropna(subset=['InvoiceNo','StockCode','Description', 'InvoiceDate'])
    data_filtered = data_filtered.drop_duplicates()
    data_filtered = data_filtered[~data_filtered['InvoiceNo'].str.startswith('C')]
    return data_filtered

def basket(data_sample):
    basket = (data_sample.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().reset_index().fillna(0)
            .set_index('InvoiceNo'))
    basket = basket.map(lambda x: 1 if x > 0 else 0)
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
    return rules

def lookup_table(data_sample):
    lookup = data_sample[['StockCode', 'Description']].drop_duplicates()
    product_dict = pd.Series(lookup.Description.values, index=lookup.StockCode).to_dict()
    return product_dict

def view_association_charts(rules):
    # lift_max = rules['Lift'].max()
    # lift_max = math.ceil(lift_max)
    # print(lift_max)
    fig, axs = plt.subplots(3, 1, figsize=(10,16))

    # Support bar plot
    sns.barplot(x='Rules', y='Support', data=rules, ax=axs[0], palette='Blues_d')
    axs[0].set_title('Support of Association Rules')
    # axs[0].set_ylabel('Support')
    # axs[0].set_ylim(0,1)

    # Confidence bar plot
    sns.barplot(x='Rules', y='Confidence', data=rules, ax=axs[1], palette="Reds_d")
    axs[1].set_title('Confidence of Association Rules')
    # axs[1].set_ylabel('Confidence')
    # axs[1].set_ylim(0,1)

    # Lift bar plot
    sns.barplot(x='Rules', y='Lift',data=rules, ax=axs[2], palette="Greens_d")
    axs[2].set_title('Lift of Association Rules')
    # axs[2].set_ylabel('Lift')
    # axs[2].set_ylim(0,lift_max)

    # plt.xticks(rotation=45)
    plt.tight_layout(pad = 5)
    # plt.subplots_adjust(right=5,left=2)
    return fig

def forecast_demand(stock_code, filtered_daily_sales, days=30):
    # Ignore DeprecationWarnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    product_sales = filtered_daily_sales[filtered_daily_sales['StockCode'] == stock_code]
    product_sales.set_index('Date', inplace=True)

    if len(product_sales) < 2:
        return None

    model = ExponentialSmoothing(product_sales['Quantity'], trend='add', seasonal_periods=None)
    model_fit = model.fit()

    forecast = model_fit.forecast(days)
    print(type(forecast))
    return forecast

def calculate_reorder_point(forecast, lead_time, safety_stock):
    avg_demand_per_day = forecast.mean()
    reorder_point = (avg_demand_per_day * lead_time) + safety_stock
    return reorder_point

lead_time = 5  # days to receive new stock
safety_stock = 50  # buffer stock to prevent stockouts

def check_replenishment(stock_code, forecast, current_inventory):
    reorder_point = calculate_reorder_point(forecast, lead_time, safety_stock)
    current_stock = current_inventory.get(stock_code, 0)
    
    result = {}
    
    if current_stock < reorder_point:
        order_quantity = reorder_point - current_stock
        result['Stock Code'] = stock_code
        result['Order Quantity'] = order_quantity
    else:
        result['Stock Code'] = stock_code
        result['Order Quantity'] = 0  # No need to reorder

    return result

def customer_seg(data_sample):
    # Aggregate data by customer: Total Spend and Frequency of purchases
    customer_data = data_sample.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',  # Frequency
        'TotalSpend': 'sum'      # Total Spend
    }).reset_index()

    customer_data.rename(columns={'InvoiceNo': 'Frequency'}, inplace=True)

    # Step 2: Feature Scaling (optional but recommended)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_data[['Frequency', 'TotalSpend']])

    # Step 3: K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=0)
    customer_data['Cluster'] = kmeans.fit_predict(scaled_features)
    return customer_data

# def clv(data_sample):
#     customer_clv = data_sample.groupby('CustomerID')['TotalSpend'].sum().reset_index()

#     # Step 3: Rename the column to CLV
#     customer_clv.rename(columns={'TotalSpend': 'CLV'}, inplace=True)

#     # Step 4: Merge CLV back into the original data or use it separately for clusterin
#     customer_data = data_sample.merge(customer_clv, on='CustomerID', how='left')

#     X = customer_data[['Frequency', 'TotalSpend']]
#     y = customer_data['LifetimeValue']  # You can create a target column for CLV (e.g., based on total purchases)

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#     # Train Linear Regression model
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Predictions
#     y_pred = model.predict(X_test)

#     # Print the CLV for each customer (predicted values)
#     customer_data['Predicted_CLV'] = model.predict(X)
#     return customer_clv[['Customer_ID', ['Frequency', 'Total_Spend'], 'Predicted_CLV']]