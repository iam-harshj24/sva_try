import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math 
import hashlib

# st.set_page_config(page_title="Inventory Management Dashboard", layout="wide")

def read_sales_data(uploaded_file, sheet_name):
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    id_cols = ['ASIN', 'Product Name']
    date_cols = [col for col in df.columns if col not in id_cols]
    melted_df = df.melt(id_vars=id_cols, value_vars=date_cols,
                        var_name='Date', value_name='Sales')
    melted_df['Date'] = pd.to_datetime(melted_df['Date'], format='%Y%m%d')
    return melted_df

def read_gross_profit(uploaded_file, sheet_name):
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    id_cols = ['ASIN', 'Product Name']
    date_cols = [col for col in df.columns if col not in id_cols]
    melted_df = df.melt(id_vars=id_cols, value_vars=date_cols,
                        var_name='Date', value_name='Gross Profit')
    melted_df['Date'] = pd.to_datetime(melted_df['Date'], format='%Y%m%d')
    return melted_df

def merge_sales_and_profit(sales_data, profit_data):
    df = pd.merge(sales_data, profit_data, on=['ASIN', 'Date', 'Product Name'], how='inner')
    df.to_csv("merged_data.csv")
    return df

def read_inventory_data(uploaded_file, sheet_name):
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    base_cols = ['ASIN', 'Product Name', 'Current inventory']
    date_columns = [col for col in df.columns if col not in base_cols]
    df['Upcoming Inventory'] = df[date_columns].sum(axis=1)
    df['Total Inventory'] = df['Current inventory'] + df['Upcoming Inventory']
    df = df[base_cols + date_columns + ['Upcoming Inventory', 'Total Inventory']]
    df.to_csv('inventory_data.csv')
    return df


def calculate_normal_drr(merged_data):
    df = merged_data.sort_values(['ASIN', 'Date'])
    df['Daily_Run_Rate'] = df.groupby('ASIN')['Sales'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    ).round()


    # Create lists of products if not already defined
    top_products = [
        "B09VPLLPMB", "B071LQFHPY", "B072M2MTK1", "B09W2VSN54", "B09W9SX1W8",
        "B09TXNSQDJ", "B07GNLN5K2", "B07YWWXLJS", "B071ZQ5J4X", "B09WZYCXRQ",
        "B07J14W55P", "B07Q4TD8RS", "B07J1JZJY2", "B09VS77ZDT", "B071ZMQ3X8",
        "B072PYW2VM", "B078Z17WQ9", "B07MCKSNZQ", "B0B8HDYRZQ", "B09W28G7L3",
        "B07895B2VZ", "B07895DHYQ", "B071W92ZRG", "B07GCQDX6M", "B07MV28C29",
        "B07FSYZM6H", "B072Q32KJ2", "B07Y1LCD7T", "B07FSZ921M", "B0BY54K6C3"
    ]

    mid_section_products = [
        "B071J8GQCJ", "B07W8THF1Q", "B0788XRBJ5", "B0BB9QN29D", "B0788WTFPY",
        "B071FNLVF5", "B078Z2KPT5", "B07J1L77D3", "B072M55KZT", "B07FSV4FNK",
        "B07M9QQ8XL", "B086X1WC4N", "B0789JGL72", "B0788X172Q", "B09W2KHTWX",
        "B07M7VQ1N1", "B072Q1NW1T", "B09V1FGQ8Z", "B07J1WMHT2", "B07GCPD2DR",
        "B07M7WL2HD", "B09V7XXJ52", "B07J1DS78L", "B072Q22P7Q", "B079G5H8XP",
        "B081RJL36N", "B07BY4KRKN", "B07QZ4138D", "B07YYCLVGQ", "B07J1K4B3R",
        "B07MT9CY7B", "B07N8XR8C4", "B07MGXBWB"
    ]

    # Create a conditions list for numpy.select
    conditions = [
        df['ASIN'].isin(top_products),
        df['ASIN'].isin(mid_section_products)
    ]

    # Create corresponding multipliers
    multipliers = [1.20, 1.15]  # 20% increase for top, 10% for mid

    # Default multiplier is 1.0 (no change) for bottom products
    df['Target_DRR'] = (df['Daily_Run_Rate'] *
                       np.select(conditions, multipliers, default=1.10)).round()
    df.to_csv('drr_data.csv')
    return df

def shipment_inventory_status(inventory_data, drr_data):
    df1 = inventory_data
    df2 = drr_data

    df = pd.merge(df1, df2, on=['ASIN', 'Product Name'], how='inner')
    df = df.drop(columns=['Upcoming Inventory', 'Total Inventory', 'Sales', 'Gross Profit'], errors='ignore')

    def format_column(column):
        if isinstance(column, datetime):
            return column.strftime("%d-%m-%Y")
        return str(column)

    df.columns = [format_column(col) for col in df.columns]
    shipment_columns = [col for col in df.columns if '-' in col and col != 'Date']

    if not shipment_columns:
        raise ValueError("No shipment date columns detected. Ensure headers are in 'DD-MM-YYYY' format.")

    df[shipment_columns] = df[shipment_columns].fillna(0)

    if "Current inventory" in df.columns:
        df["Current inventory"] = df["Current inventory"].fillna(0)
    else:
        raise ValueError("Missing 'Current inventory' column in the data.")

    if "Daily_Run_Rate" in df.columns:
        df["Daily_Run_Rate"] = df["Daily_Run_Rate"].fillna(0)
    else:
        raise ValueError("Missing 'Daily_Run_Rate' column in the data.")

    def get_inventory_status(days):
        if pd.isna(days) or days == float('inf'):
            return 'No Sales Data'
        elif days <= 20:
            return 'In AIR'
        elif days <= 60:
            return 'Expected to be in air'
        elif days <= 80:
            return 'Sea Shipment'
        elif days <= 100:
            return 'Planned to send in Next Sea Shipment'
        else:
            return 'Sufficient Inventory'

    results = []
    for index, row in df.iterrows():
        shipment_quantities = row[shipment_columns].tolist()
        current_date = datetime.today()
        original_inventory = row["Current inventory"]
        updated_inventory = original_inventory

        for i, shipment_date in enumerate(shipment_columns):
            shipment_date = datetime.strptime(shipment_date, "%d-%m-%Y")
            if shipment_date <= current_date:
                updated_inventory += shipment_quantities[i]
        

        Daily_Run_Rate = row["Daily_Run_Rate"]

        if Daily_Run_Rate == 0:
            out_of_stock_date = "N/A (No consumption rate)"
            expected_date_to_be_in_air = "N/A"
            days_survived = float('inf')
        else:
            out_of_stock_date, expected_date_to_be_in_air = None, None

            for i, shipment_date in enumerate(shipment_columns):
                shipment_date = datetime.strptime(shipment_date, "%d-%m-%Y")
                if shipment_date < current_date:
                    continue
                days_to_next_shipment = (shipment_date - current_date).days
                inventory_needed = days_to_next_shipment * Daily_Run_Rate

                if updated_inventory >= inventory_needed:
                    updated_inventory -= inventory_needed
                    current_date = shipment_date
                    updated_inventory += shipment_quantities[i]
                else:
                    days_survived = updated_inventory // Daily_Run_Rate
                    out_of_stock_date = current_date + timedelta(days=days_survived)
                    break

            if not out_of_stock_date:
                days_survived_after_last_shipment = int(updated_inventory // Daily_Run_Rate)
                out_of_stock_date = current_date + timedelta(days=days_survived_after_last_shipment)

            if out_of_stock_date != "N/A (No consumption rate)":
                expected_date_to_be_in_air = out_of_stock_date - timedelta(days=20)

        inventory_status = get_inventory_status(
            (out_of_stock_date - datetime.today()).days if isinstance(out_of_stock_date, datetime) else float('inf'))

        total_upcoming_shipment = original_inventory + sum(shipment_quantities)

        results.append({
            'Date': row['Date'],
            'ASIN': row['ASIN'],
            'Product Name': row['Product Name'],
            'Current Inventory': row["Current inventory"],
            'Updated Current Inventory': updated_inventory,
            'Daily_Run_Rate': Daily_Run_Rate,
            
            'Date of OOS': out_of_stock_date.strftime("%d-%m-%Y") if isinstance(out_of_stock_date, datetime) else out_of_stock_date,
            'Expected Date to be in Air': expected_date_to_be_in_air.strftime("%d-%m-%Y") if isinstance(expected_date_to_be_in_air, datetime) else expected_date_to_be_in_air,
            'Days of inventory': (out_of_stock_date - datetime.today()).days if isinstance(out_of_stock_date, datetime) else "N/A",
            'Inventory Status': inventory_status,
            'Total Upcoming Shipment': total_upcoming_shipment  # New column added
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('inventory_status_with_updated_inventory_and_total_shipment.csv', index=False)
    print("Results saved to inventory_status_with_updated_inventory_and_total_shipment.csv")

    return results_df



def calculate_shipment_plan(inventory_status, target_date, current_date=None):
    if current_date is None:
        current_date = datetime.now()

    target_date = pd.to_datetime(target_date)
    current_date = pd.to_datetime(current_date)
    result = inventory_status.copy()

    # date_columns = [col for col in result.columns if isinstance(col, datetime)]
    # drr_column = 'Daily_Retail_Rate_Deal' if is_deal_day else 'Daily_Retail_Rate'

    days_until_target = (target_date - current_date).days
    result['Days_To_Target'] = int(days_until_target+1)
    result['Expected_Usage'] = (result['Daily_Run_Rate'] * result['Days_To_Target']).round()

    # future_shipment_cols = [col for col in date_columns if col<= target_date]
    # result['Total Upcoming Shipments'] = result[future_shipment_cols].sum(axis=1) if future_shipment_cols else 0

    result['Total_Available'] = result['Total Upcoming Shipment']
    result['Required Projected Inventory'] = (
        result['Expected_Usage']-result['Total_Available']
    ).round()

    result['Buffer_Stock'] = (result['Daily_Run_Rate'] * 15).round()

    result['Required_Shipment_with_buffer_stock'] = (
        result['Buffer_Stock'] + result['Required Projected Inventory'])
          
    

    # def get_shipment_priority(row):
    #     if row['Status'] == 'No Sales Data':
    #         return 'No Priority - No Sales Data'
    #     elif row['Status'] == 'Urgent Restock':
    #         return 'High - Stockout Expected(IN AIR)'
    #     elif row['Status'] == 'Restock Soon':
    #         return 'Medium - Expected to be in air'
    #     elif row['Status'] == 'Send stocks in Next Shipment':
    #         return 'Medium - Plan for Sea Shipment'
    #     else:
    #         return 'Low - Sufficient Stock'



    # result['Shipment_Priority'] = result.apply(get_shipment_priority, axis=1)
    # result['Estimated_Arrival'] = target_date + timedelta(days=65)


    result.to_csv('shipment_details.csv')
    return result


### day wise loss report ###

import pandas as pd

def calculate_daily_loss_report(file_path, sheet_name, decimal_places=2):
    # Load the data from the specified Excel sheet
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # Ensure the data is a DataFrame
    if isinstance(data, pd.DataFrame):
        # Dictionary to store the results for each day
        daily_report = {}

        # Loop through each product row starting from the second row (index 1)
        for index, row in data.iterrows():
            product_name = row[1]  # assuming the second column is product name
            ASIN = row[0]  # assuming the first column contains the ASIN

            # Extract the profit data excluding ASIN and product name (which are in columns 0 and 1)
            profit_data = row[2:]

            # Loop through each day's profit and calculate losses (if any)
            for date, profit in profit_data.items():
                if profit < 0:  # Loss condition
                    if date not in daily_report:
                        daily_report[date] = {'Total Loss': 0, 'Product Count': 0}
                    daily_report[date]['Total Loss'] += profit
                    daily_report[date]['Product Count'] += 1

        # Round the total losses to the specified number of decimal places
        for date, report in daily_report.items():
            report['Total Loss'] = round(report['Total Loss'], decimal_places)

        # Convert the daily report dictionary to a DataFrame
        report_df = pd.DataFrame.from_dict(daily_report, orient='index')

        # Sort the DataFrame by the index (date) in descending order (latest dates first)
        report_df = report_df.sort_index(ascending=False)

        # Write the report to a CSV file
       

        return report_df
    else:
        raise ValueError("Input data must be a pandas DataFrame.")
    




import pandas as pd

def calculate_averages_and_percentage_change(file_path, sheet_name, decimal_places=2):
    # Load the data from the specified Excel sheet
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # Ensure the data is a DataFrame
    if isinstance(data, pd.DataFrame):
        # List to store the results
        results = []

        # Loop through each product row starting from the second row (index 1)
        for index, row in data.iterrows():
            # The product name is now the second column (index 1) and ASIN is in the first column (index 0)
            ASIN = row[0]
            product_name = row[1]

            # Extract the profit values (skipping the first two columns, ASIN and product name)
            profit_data = row[2:]

            # Ensure there are enough days of data to calculate the averages and percentage change
            if len(profit_data) >= 5:
                # Extract the last 5 days of data excluding today's and the previous day's data
                last_five_days = profit_data.head(5)
                today_profit = last_five_days.iloc[0]  # The second to last day's data (2-day lag)
                last_three_days = profit_data.head(3)  # The first 3 days of the last 5 days excluding the lag

                # Calculate the 3-day average (excluding today's and previous day's data)
                three_day_avg = last_three_days.mean()
                three_day_avg = round(three_day_avg, decimal_places)

                # Calculate the 5-day average (excluding today's and previous day's data)
                five_day_avg = last_five_days.mean()
                five_day_avg = round(five_day_avg, decimal_places)

                # Calculate the percentage change based on the 3-day average
                if three_day_avg != 0:
                    percentage_change_3_day = ((today_profit - three_day_avg) / three_day_avg) * 100
                else:
                    percentage_change_3_day = None

                # Calculate the percentage change based on the 5-day average
                if five_day_avg != 0:
                    percentage_change_5_day = ((today_profit - five_day_avg) / five_day_avg) * 100
                else:
                    percentage_change_5_day = None
            else:
                three_day_avg = None
                five_day_avg = None
                percentage_change_3_day = None
                percentage_change_5_day = None
                today_profit = None

            # Append the data as a row in the results list
            results.append({
                'Product Name': product_name,
                'ASIN': ASIN,
                'Today\'s Data': today_profit,
                '3-Day Average': three_day_avg,
                'Percentage Change (3-day avg)': percentage_change_3_day,
                '5-Day Average': five_day_avg,

                'Percentage Change (5-day avg)': percentage_change_5_day,

            })

        # Convert results list to a DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv('unnatural changes in profit.csv')
        return results_df
    else:
        raise ValueError("Input data must be a pandas DataFrame.")
    

def calculate_normal_drr(merged_data, use_manual_drr=False, manual_drr_value=None):
    df = merged_data.sort_values(['ASIN', 'Date'])
    
    if use_manual_drr and manual_drr_value is not None:
        # Use the single manual DRR value for all ASINs
        df['Daily_Run_Rate'] = manual_drr_value
    else:
        # Calculate DRR normally
        df['Daily_Run_Rate'] = df.groupby('ASIN')['Sales'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
    
    return df.round()

def calculate_max_drr_with_push_drr(inventory_data, target_date, future_date, manual_drr=None):
    df = inventory_data.copy()
    
    # Validate dates
    target_date = pd.to_datetime(target_date)
    future_date = pd.to_datetime(future_date)
    current_date = (datetime.now() - timedelta(days=2)).date()
    
    # Initialize results list
    results = []
    
    for _, row in df.iterrows():
        product_name = row['Product Name']
        current_inventory = row['Current inventory']
        
        # Get shipment columns
        shipment_cols = [col for col in df.columns if isinstance(col, datetime) or 
                       (isinstance(col, str) and col not in ['ASIN', 'Product Name', 'Current inventory'])]
        
        # Create validated shipments list
        shipments = []
        for col in shipment_cols:
            try:
                date = pd.to_datetime(col)
                if pd.notnull(row[col]):
                    shipments.append((date, float(row[col])))
            except:
                continue
        
        # Sort shipments by date
        shipments.sort(key=lambda x: x[0])
        
        # Calculate initial inventory based on future date
        initial_inventory = current_inventory
        calc_start_date = current_date if future_date.date() <= current_date else future_date.date()
        
        # If manual DRR is provided, calculate inventory adjustment up to future date
        if manual_drr is not None:
            days_to_future = (calc_start_date - current_date).days
            initial_inventory -= (days_to_future * manual_drr)
        
        # Add all shipments up to the future date
        for ship_date, ship_qty in shipments:
            if ship_date.date() <= calc_start_date:
                initial_inventory += ship_qty
        
        # Get remaining shipments after future date
        future_shipments = [(d, q) for d, q in shipments if d.date() > calc_start_date and d.date() <= target_date.date()]
        
        # Binary search for maximum DRR
        left, right = 0, 10000  # Reasonable upper bound
        max_drr = 0
        best_ending_inventory = 0
        
        while left <= right:
            mid = (left + right) // 2
            current_inventory = initial_inventory
            valid = True
            temp_date = calc_start_date
            shipment_index = 0
            
            while temp_date <= target_date.date():
                # Process any shipments for this date
                while (shipment_index < len(future_shipments) and 
                       future_shipments[shipment_index][0].date() <= temp_date):
                    current_inventory += future_shipments[shipment_index][1]
                    shipment_index += 1
                
                # Subtract daily consumption
                current_inventory -= mid
                
                if current_inventory < 0:
                    valid = False
                    break
                
                temp_date += timedelta(days=1)
            
            if valid:
                if mid > max_drr:
                    max_drr = mid
                    best_ending_inventory = current_inventory
                left = mid + 1
            else:
                right = mid - 1
        
        # Calculate total incoming shipments for both periods
        shipments_before_future = sum(qty for date, qty in shipments if date.date() <= calc_start_date)
        shipments_after_future = sum(qty for date, qty in shipments if calc_start_date < date.date() <= target_date.date())
        
        results.append({
            'Product Name': product_name,
            'ASIN': row.get('ASIN', 'N/A'),
            'Current Inventory': initial_inventory,
            'Max DRR': max_drr,
            # 'Ending Inventory': best_ending_inventory,
            'Total Shipments Before Future': shipments_before_future,
            'Total Shipments After Future': shipments_after_future,
            # 'Manual DRR Used': manual_drr if manual_drr is not None else 'No',
            # 'Calculation Start Date': calc_start_date
        })
    
    return pd.DataFrame(results)

def calculate_daily_drr(file_path, sheet_name, target_date):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return pd.DataFrame()

    required_cols = ['Product Name', 'Current inventory']
    if not all(col in df.columns for col in required_cols):
        st.error("Missing required columns in the Excel sheet.")
        return pd.DataFrame()

    df['Current inventory'] = pd.to_numeric(df['Current inventory'], errors='coerce').fillna(0).astype(int)

    # Identify shipment columns
    shipment_cols = df.columns[df.columns.get_loc('Current inventory')+1:]
    shipment_dates = pd.to_datetime(shipment_cols, errors='coerce')
    valid_shipments = {date: col for date, col in zip(shipment_dates, shipment_cols) if pd.notnull(date)}

    try:
        target_date = pd.to_datetime(target_date)
    except ValueError:
        st.error("Invalid target date format.")
        return pd.DataFrame()

    date_range = pd.date_range(start=datetime.today(), end=target_date, freq='D')
    results = {}

    for _, row in df.iterrows():
        product_name = row['Product Name']
        initial_inventory = row['Current inventory']
        shipments = [(date, int(pd.to_numeric(row[col], errors='coerce'))) for date, col in valid_shipments.items() if pd.notnull(row[col]) and row[col] > 0]
        shipments.sort()

        left, right = 0, max(initial_inventory + sum(qty for _, qty in shipments), 1000)
        base_drr = 0

        # Binary search for max sustainable DRR
        while left <= right:
            mid = (left + right) // 2
            remaining = initial_inventory
            current_date = datetime.today()
            shipment_idx = 0
            is_sustainable = True

            while current_date <= target_date:
                while shipment_idx < len(shipments) and shipments[shipment_idx][0] <= current_date:
                    remaining += shipments[shipment_idx][1]
                    shipment_idx += 1

                remaining -= mid
                if remaining < 0:
                    is_sustainable = False
                    break

                current_date += timedelta(days=1)

            if is_sustainable:
                base_drr = mid
                left = mid + 1
            else:
                right = mid - 1

        # Adjust DRR for different periods
        total_days = len(date_range)
        period_length = total_days // 3
        drr_data = {}

        for i, date in enumerate(date_range):
            if i < period_length:
                multiplier = 0.9
            elif i < 2 * period_length:
                multiplier = 1.1
            else:
                multiplier = 1.3

            drr_data[date.strftime('%Y-%m-%d')] = int(base_drr * multiplier)

        results[product_name] = {'Current inventory': initial_inventory, **drr_data}

    result_df = pd.DataFrame.from_dict(results, orient='index')
    return result_df

# Update the main tabs section in your main() function:
# def main():
    # ... (previous code remains the same until tabs creation)
    
    # Create tabs with the new DRR Timeline tab
    # tabs = st.tabs(["Overview", "Inventory Status", "Shipment Planning", 
                    # "Loss Analysis", "Profit Analysis", "Maximum DRR Analysis", 
                    # "DRR Timeline"])
    
    # ... (previous tab content remains the same)
    
    # Add the new DRR Timeline tab
    # with tabs[6]:
    #     add_drr_timeline_tab()


    
def init_users():
    if 'users' not in st.session_state:
        st.session_state.users = {
            'Harsh': {
                'password': hashlib.sha256('9838'.encode()).hexdigest(),
                'role': 'Admin',
                'permissions': ['overview', 'inventory_status', 'shipment_planning', 'loss_analysis', 
                 'profit_analysis', 'max_drr', 'drr_timeline', 'labels', 'manage_users']
            }
        }

def get_role_permissions():
    return {
        'Admin': ['overview', 'inventory_status', 'shipment_planning', 'loss_analysis', 
                 'profit_analysis', 'max_drr', 'drr_timeline', 'labels', 'manage_users'],
        'admin': ['overview', 'inventory_status', 'shipment_planning', 'loss_analysis', 
                 'profit_analysis', 'max_drr', 'drr_timeline', 'labels'],
        'inventory': ['overview', 'inventory_status', 'shipment_planning', 'max_drr', 'drr_timeline'],
        'Labels': ['overview', 'labels'],
        'viewer': ['overview']
    }

def add_user(username, password, role):
    if username in st.session_state.users:
        return False, "Username already exists"
    
    st.session_state.users[username] = {
        'password': hashlib.sha256(password.encode()).hexdigest(),
        'role': role,
        'permissions': get_role_permissions()[role]
    }
    return True, "User added successfully"

def remove_user(username):
    if username == 'Harsh':
        return False, "Cannot remove admin user"
    if username in st.session_state.users:
        del st.session_state.users[username]
        return True, "User removed successfully"
    return False, "User not found"

def update_user_role(username, new_role):
    if username == 'Harsh':
        return False, "Cannot modify admin role"
    if username in st.session_state.users:
        st.session_state.users[username]['role'] = new_role
        st.session_state.users[username]['permissions'] = get_role_permissions()[new_role]
        return True, "Role updated successfully"
    return False, "User not found"

def check_password():
    init_users()
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Login")
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username")
        with col2:
            password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in st.session_state.users and \
               st.session_state.users[username]['password'] == hashlib.sha256(password.encode()).hexdigest():
                st.session_state.authenticated = True
                st.session_state.current_user = username
                st.session_state.current_role = st.session_state.users[username]['role']
                st.rerun()
            else:
                st.error("Invalid credentials")
        return False

    return True

def user_management():
    if st.session_state.current_user != 'Harsh':
        st.warning("Only admin can manage users")
        return

    st.subheader("User Management")
    
    tab1, tab2, tab3 = st.tabs(["Add User", "Remove User", "Update Role"])
    
    with tab1:
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        new_role = st.selectbox("Select Role", options=list(get_role_permissions().keys()))
        if st.button("Add User"):
            success, message = add_user(new_username, new_password, new_role)
            if success:
                st.success(message)
            else:
                st.error(message)

    with tab2:
        username_to_remove = st.selectbox("Select User to Remove", 
                                        options=[u for u in st.session_state.users.keys() if u != 'Harsh'])
        if st.button("Remove User"):
            success, message = remove_user(username_to_remove)
            if success:
                st.success(message)
            else:
                st.error(message)

    with tab3:
        username_to_update = st.selectbox("Select User to Update", 
                                        options=[u for u in st.session_state.users.keys() if u != 'Harsh'])
        new_role = st.selectbox("Select New Role", 
                               options=list(get_role_permissions().keys()),
                               key="update_role")
        if st.button("Update Role"):
            success, message = update_user_role(username_to_update, new_role)
            if success:
                st.success(message)
            else:
                st.error(message)

    st.write("Current Users:")
    user_list = pd.DataFrame(
        [(user, data['role'], ', '.join(data['permissions'])) 
         for user, data in st.session_state.users.items()],
        columns=["Username", "Role", "Permissions"]
    )
    st.dataframe(user_list)

def has_permission(permission):
    return permission in st.session_state.users[st.session_state.current_user]['permissions']

def read_us_products_data(uploaded_file, sheet_name="US Products"):
    """Reads US Products sheet from Excel file and returns a DataFrame with ASIN, AWD, Backstock, and Upcoming Orders."""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        df = df[['ASIN','Product Name', 'AWD', 'Backstock', 'Upcoming Orders']].dropna()
        return df
    except Exception as e:
        st.error(f"Error reading {sheet_name} sheet: {e}")
        return pd.DataFrame()
    
def calculate_us_shipment_plan(inventory_status, us_products_data, target_date):
    if target_date is None:
        target_date = datetime.now() + timedelta(days=30)

    target_date = pd.to_datetime(target_date)
    result = inventory_status.copy()

    # Filter inventory status to include only ASINs present in the US Products sheet
    result = result[result['ASIN'].isin(us_products_data['ASIN'])]

    # Calculate Days until Target
    result['Days_To_Target'] = (target_date - datetime.today()).days + 1
    result['Expected_Usage'] = (result['Daily_Run_Rate'] * result['Days_To_Target']).round()

    # Merge inventory data with US Products AWD, Backstock, and Upcoming Order
    result = result.merge(us_products_data, on='ASIN', how='left').fillna(0)
    result['Product Name'] = result['Product Name_x']

    # Display all columns (including AWD, Backstock, and Upcoming Orders)
    result = result[['ASIN', 'Product Name', 'Current Inventory', 'Daily_Run_Rate', 'Expected_Usage',
                     'Total Upcoming Shipment', 'Days_To_Target', 
                     'AWD', 'Backstock', 'Upcoming Orders']]

    # Adjust total available inventory
    result['Total Available Stocks with BS & AWD'] = result['Total Upcoming Shipment'] + result['AWD'] + result['Backstock'] 
    result['Total Available Stocks with BS,AWD & Orders'] = result['Total Upcoming Shipment'] + result['AWD'] + result['Backstock']+result['Upcoming Orders']

    # Calculate Required Projected Inventory
    # result['Required Projected Inventory With AWD & BS'] = (result['Expected_Usage'] - result['Total Available Stocks with BS & AWD']).round()
    result['Required Inventory(AWD+BS+ORDERS)'] = (result['Expected_Usage'] - result['Total Available Stocks with BS,AWD & Orders']).round()
    # Subtract AWD, Backstock, and Upcoming Orders from Required Projected Inventory
    # result['Final Required Shipment'] = result['Required Projected Inventory'] - result['AWD'] - result['Backstock'] - result['Upcoming Orders']

    # Ensure Final Required Shipment is not negative
    # result['Final Required Shipment'] = result['Final Required Shipment'].apply(lambda x: max(x, 0))
  
    result.to_csv('us_shipment_pla.csv', index=False)
    return result
def process_label_planning(uploaded_file, inventory_status, target_date=None):
    """Reads label data, merges with inventory data, and calculates label planning details."""
    try:
        # Read label data
        label_data = pd.read_excel(uploaded_file, sheet_name='labels')
        label_data = label_data[['ASIN', 'Product Name', 'IN Stocks', 'Packed', 'New Orders']].dropna()
    except Exception as e:
        st.error(f"Error reading labels sheet: {e}")
        return pd.DataFrame()
    
    if target_date is None:
        target_date = datetime.now() + timedelta(days=30)
    target_date = pd.to_datetime(target_date)
    
    # Copy inventory status and calculate expected usage
    result = inventory_status.copy()
    result['Days_To_Target'] = (target_date - datetime.today()).days + 1
    result['Expected_Usage'] = (result['Daily_Run_Rate'] * result['Days_To_Target']).round()
    
    # Merge label data
    result = result.merge(label_data, on='ASIN', how='left').fillna(0)
    result['Product Name'] = result['Product Name_x']
    
    # Calculate required inventory
    result['Total Available Label (Stocks+Packed)'] = result['Total Upcoming Shipment'] + result['IN Stocks'] + result['Packed'] 
    result['Total Available Label (Stocks+Packed+New Orders)'] = result['Total Available Label (Stocks+Packed)'] + result['New Orders']
    result['Required Labels (Stocks+Packed+New Orders)'] = (result['Expected_Usage'] - result['Total Available Label (Stocks+Packed+New Orders)']).round()
    
    return result[['Date','ASIN', 'Product Name', 'Current Inventory', 'Daily_Run_Rate', 'Expected_Usage',
                   'Total Upcoming Shipment', 'Days_To_Target', 'IN Stocks', 'Packed', 'New Orders',
                   'Total Available Label (Stocks+Packed)', 'Total Available Label (Stocks+Packed+New Orders)',
                   'Required Labels (Stocks+Packed+New Orders)']]


import pandas as pd
import streamlit as st

import pandas as pd
import streamlit as st

def Performance_Tracker(uploaded_file, sales_sheet="Sales", profit_sheet="Profit", tracker_sheet="Tracker"):
    """
    Function to calculate and display average sales, profit, and DRR based on a selected date range,
    week-over-week analysis, or week-by-year analysis.
    """
    try:
        # Read sales, profit, and tracker data
        sales_data = pd.read_excel(uploaded_file, sheet_name=sales_sheet)
        profit_data = pd.read_excel(uploaded_file, sheet_name=profit_sheet)
        tracker_data = pd.read_excel(uploaded_file, sheet_name=tracker_sheet)

        # Ensure necessary columns exist
        if not {'ASIN', 'Product Name'}.issubset(sales_data.columns) or not {'ASIN', 'Product Name'}.issubset(profit_data.columns):
            st.error("Missing 'ASIN' or 'Product Name' columns in the dataset.")
            return

        # Identify date columns dynamically (all except 'ASIN' & 'Product Name')
        date_columns_sales = [col for col in sales_data.columns if col not in ['ASIN', 'Product Name']]
        date_columns_profit = [col for col in profit_data.columns if col not in ['ASIN', 'Product Name']]

        # Convert column names to datetime for sorting and processing
        date_mapping = {col: pd.to_datetime(col, errors='coerce') for col in date_columns_sales}
        sorted_dates = sorted(date_mapping.items(), key=lambda x: x[1])

        # Group dates into weeks by year
        week_groups = {}
        for col, date in sorted_dates:
            if pd.notna(date):  # Ensure valid dates
                year = date.year
                week_num = date.isocalendar()[1]  # Get ISO week number
                week_label = f"Week {week_num} ({year})"
                
                if week_label not in week_groups:
                    week_groups[week_label] = []
                week_groups[week_label].append(col)

        # Add 'Custom' option
        week_options = list(week_groups.keys()) + ["Custom"]
        selected_option = st.selectbox("📅 Select Date Range", week_options)

        if selected_option == "Custom":
            # Custom Date Selection
            col1, col2 = st.columns(2)
            sorted_dates_str = [str(date.date()) for _, date in sorted_dates if pd.notna(date)]
            start_date = col1.selectbox("📅 Select Start Date", sorted_dates_str)
            end_date = col2.selectbox("📅 Select End Date", sorted_dates_str, index=len(sorted_dates_str) - 1)

            # Convert user-selected dates back to column format
            selected_dates_sales = [col for col, date in sorted_dates if str(date.date()) >= start_date and str(date.date()) <= end_date]
        else:
            # Week-based selection (Week-by-Year)
            selected_dates_sales = week_groups[selected_option]

        selected_dates_profit = selected_dates_sales  # Assume profit has the same date columns

        # Convert selected date columns to numeric (handling missing or non-numeric data)
        sales_data[selected_dates_sales] = sales_data[selected_dates_sales].apply(pd.to_numeric, errors='coerce')
        profit_data[selected_dates_profit] = profit_data[selected_dates_profit].apply(pd.to_numeric, errors='coerce')

        # Calculate average sales & profit for selected date range
        sales_data["Average Sales"] = sales_data[selected_dates_sales].mean(axis=1)
        profit_data["Average Profit"] = profit_data[selected_dates_profit].mean(axis=1)

        # Merge sales and profit on ASIN & Product Name
        Pre_final_data = sales_data[['ASIN', 'Product Name', 'Average Sales']].merge(
            profit_data[['ASIN', 'Product Name', 'Average Profit']],
            on=['ASIN', 'Product Name'],
            how='inner'
        )

        final_data = Pre_final_data[['ASIN', 'Product Name', 'Average Sales', 'Average Profit']].merge(
            tracker_data[['ASIN', 'Product Name', 'Target DRR']],
            on=['ASIN', 'Product Name'],
            how='inner'
        )
        final_data["Performance Status"] = final_data.apply(
            lambda row: "Leading ✅" if row["Target DRR"] < row["Average Sales"] else "Lagging ⚠️",
            axis=1
        )

        # Display results in Streamlit
        num_days = len(selected_dates_sales)
        st.write(f"### 📊 Average Sales & Profit Per Product ({selected_option} - {num_days} Days)")
        st.dataframe(final_data)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)



import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import tempfile
import os

def main():
    if not check_password():
        return

    st.sidebar.title(f"Welcome, {st.session_state.current_user}")
    st.sidebar.write(f"Role: {st.session_state.current_role}")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    if st.session_state.current_user == 'Harsh':
        if st.sidebar.button("Manage Users"):
            st.session_state.show_user_management = True
        
    if st.session_state.get('show_user_management', False):
        user_management()
        if st.button("Back to Dashboard"):
            st.session_state.show_user_management = False
            st.rerun()
        return
    # st.set_page_config(page_title="Inventory Management Dashboard", layout="wide")

    # Custom CSS styling
    st.markdown("""
        <style>
            .main-header {text-align: center; padding: 1rem;}
            .filter-section {background-color: #f0f2f6; padding: 1rem; border-radius: 5px;}
            .stMetricValue {font-size: 24px !important;}
            .status-urgent {color: red;}
            .status-warning {color: orange;}
            .status-good {color: green;}
        </style>
    """, unsafe_allow_html=True)

    # st.title("Inventory Management Dashboard")

    # File upload in Sidebar
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=['xlsx'], key="excel_uploader")

    if uploaded_file:
        try:
            # Save uploaded file temporarily for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name

            # Load all data first
            with st.spinner("Loading data..."):
                sales_data = read_sales_data(uploaded_file, "Sales")
                profit_data = read_gross_profit(uploaded_file, "Profit")
                merged_data = merge_sales_and_profit(sales_data, profit_data)
                inventory_data = read_inventory_data(uploaded_file, "Inventory")
                # label_data = read_labels_data(uploaded_file,'labels')
                us_products_data = read_us_products_data(uploaded_file, "US Products")
                # Sidebar: DRR settings
                st.sidebar.header("DRR Settings")
                use_manual_drr = st.sidebar.checkbox("Use Manual DRR", value=False)
                manual_drr_value = None
                if use_manual_drr:
                    manual_drr_value = st.sidebar.number_input("Enter Manual DRR Value", min_value=0.0, value=100.0, step=0.1)
                
                drr_data = calculate_normal_drr(merged_data, use_manual_drr, manual_drr_value)
                inventory_status = shipment_inventory_status(inventory_data, drr_data)


            # Global Filters in Sidebar
            st.sidebar.header("Global Filters")
            
            # Date filter
            date_options = sorted(merged_data['Date'].unique())
            selected_dates = st.sidebar.multiselect(
                "Select Dates",
                options=date_options,
                # default=date_options[-7:] if len(date_options) > 7 else date_options  # Default to last 7 days
            )

            # ASIN filter
            asin_options = sorted(merged_data['ASIN'].unique())
            selected_asins = st.sidebar.multiselect(
                "Select ASINs",
                options=asin_options
            )

            # Product filter
            product_options = sorted(merged_data['Product Name'].unique())
            selected_products = st.sidebar.multiselect(
                "Select Products",
                options=product_options
            )

            # Function to apply filters to any dataframe
            def apply_filters(df):
                filtered_df = df.copy()
                
                if 'Date' in df.columns and selected_dates:
                    filtered_df = filtered_df[filtered_df['Date'].isin(selected_dates)]
                
                if 'ASIN' in df.columns and selected_asins:
                    filtered_df = filtered_df[filtered_df['ASIN'].isin(selected_asins)]
                
                if 'Product Name' in df.columns and selected_products:
                    filtered_df = filtered_df[filtered_df['Product Name'].isin(selected_products)]
                
                return filtered_df

            # Apply filters to all relevant dataframes
            filtered_merged_data = apply_filters(merged_data)
            filtered_inventory_status = apply_filters(inventory_status)
            filtered_inventory_data = apply_filters(inventory_data)

            # Create tabs
            tabs = st.tabs(["Overview", "Inventory Status", "Shipment Planning", 
                            "Loss Analysis", "Profit Analysis", "Maximum DRR Analysis", 
                            "DRR Timeline","Labels data","Target Sales Management","US Products Shipment Planning","Performance Tracker"])
            # Overview Tab
            with tabs[0]:
                st.header("Overview")
                # Metrics using filtered data
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Products", len(filtered_merged_data['ASIN'].unique()))
                with col2:
                    st.metric("Total Sales", f"{filtered_merged_data['Sales'].sum():,.0f}")
                with col3:
                    st.metric("Average Sales", f"{filtered_merged_data['Sales'].mean():,.2f}")
                with col4:
                    st.metric("Total Profit", f"${filtered_merged_data['Gross Profit'].sum():,.2f}")

                # Sales Trend with filtered data
                st.subheader("Sales Trend")
                sales_trend = filtered_merged_data.groupby('Date')['Sales'].sum().reset_index()
                fig_sales = px.line(sales_trend, x='Date', y='Sales', title='Daily Sales Trend')
                st.plotly_chart(fig_sales, use_container_width=True)

                # Profit Trend with filtered data
                st.subheader("Profit Trend")
                profit_trend = filtered_merged_data.groupby('Date')['Gross Profit'].sum().reset_index()
                fig_profit = px.line(profit_trend, x='Date', y='Gross Profit', title='Daily Profit Trend')
                st.plotly_chart(fig_profit, use_container_width=True)

                st.subheader("Filtered Data View")
                st.dataframe(filtered_merged_data)

            # Inventory Status Tab
            with tabs[1]:
                st.header("Inventory Status")
                # Inventory Status Summary with filtered data
                st.subheader("Inventory Status Summary")
                status_summary = filtered_inventory_status['Inventory Status'].value_counts()
                fig_status = px.pie(values=status_summary.values, 
                                    names=status_summary.index,
                                    title='Distribution of Inventory Status')
                st.plotly_chart(fig_status)

                # Detailed Inventory Status with filtered data
                st.subheader("Detailed Inventory Status")
                display_columns = [
                    'Date', 'ASIN', 'Product Name', 'Current Inventory', 
                    'Daily_Run_Rate', 'Date of OOS', 'Days of inventory', 
                    'Inventory Status'
                ]
                st.dataframe(filtered_inventory_status[display_columns])

                st.subheader("Upcoming Shipments")
                st.dataframe(filtered_inventory_data)

            # Shipment Planning Tab
            with tabs[2]:
                st.header("Shipment Planning")
                target_date = st.date_input(
                    "Select Target Date for Shipment Planning",
                    value=datetime.now() + timedelta(days=30),
                    min_value=datetime.now()
                )
                
                shipment_plan = calculate_shipment_plan(filtered_inventory_status, target_date)
                
                # Shipment Requirements Visualization with filtered data
                st.subheader("Shipment Requirements")
                fig_shipment = px.bar(shipment_plan, 
                                    x='Product Name',
                                    y='Required_Shipment_with_buffer_stock',
                                    title='Required Shipment Quantities by Product')
                fig_shipment.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_shipment, use_container_width=True)

                # Detailed Shipment Plan
                st.subheader("Detailed Shipment Plan")
                st.dataframe(shipment_plan)

            # Loss Analysis Tab
            with tabs[3]:
                st.header("Loss Analysis")
                loss_report = calculate_daily_loss_report(temp_path, "Profit")
                
                if selected_dates:
                    loss_report = loss_report[loss_report.index.isin(selected_dates)]
                
                # Loss Trend Visualization
                st.subheader("Daily Loss Trend")
                fig_loss = px.line(loss_report, 
                                   x=loss_report.index,
                                   y='Total Loss',
                                   title='Daily Loss Trend')
                st.plotly_chart(fig_loss, use_container_width=True)
                
                # Product Count with Losses
                st.subheader("Products with Losses")
                fig_count = px.bar(loss_report,
                                   x=loss_report.index,
                                   y='Product Count',
                                   title='Number of Products with Losses by Date')
                st.plotly_chart(fig_count, use_container_width=True)
                
                st.subheader("Detailed Loss Report")
                st.dataframe(loss_report)

            # Profit Analysis Tab
            with tabs[4]:
                st.header("Profit & Sales Change Analysis")
                st.subheader("Profit Change Analysis")
                profit_analysis = calculate_averages_and_percentage_change(temp_path, "Profit")
                filtered_profit = apply_filters(profit_analysis)
    
                fig_profit_change = px.scatter(filtered_profit,
                                               x='3-Day Average',
                                               y='Percentage Change (3-day avg)',
                                               hover_data=['Product Name'],
                                               title='Profit Change vs 3-Day Average')
                st.plotly_chart(fig_profit_change, use_container_width=True)
    
                st.subheader("Detailed Profit Analysis")
                st.dataframe(filtered_profit)
    
                # Process Sales Sheet
                st.subheader("Sales Change Analysis")
                sales_analysis = calculate_averages_and_percentage_change(temp_path, "Sales")
                filtered_sales = apply_filters(sales_analysis)
    
                fig_sales_change = px.scatter(filtered_sales,
                                              x='3-Day Average',
                                              y='Percentage Change (3-day avg)',
                                              hover_data=['Product Name'],
                                                          title='Sales Change vs 3-Day Average')
                st.plotly_chart(fig_sales_change, use_container_width=True)
    
                st.subheader("Detailed Sales Analysis")
                st.dataframe(filtered_sales)
            

            # Maximum DRR Analysis Tab
            with tabs[5]:
                st.header("Maximum DRR Analysis")

                col1, col2, col3 = st.columns(3)
                with col1:
                    target_date = st.date_input(
                        "Target Date",
                        value=datetime.now() + timedelta(days=30)
                    )
                with col2:
                    future_date = st.date_input(
                        "Starting Date",
                        value=datetime.now()
                    )
                with col3:
                    use_manual_drr_max = st.checkbox("Use Manual DRR for Max DRR")
                    manual_drr_max = None
                    if use_manual_drr_max:
                        manual_drr_max = st.number_input("Enter Manual DRR", min_value=0.0, value=100.0, step=0.1)

                if st.button("Calculate Maximum DRR"):
                    filtered_inventory_data = apply_filters(inventory_data)
                    max_drr_results = calculate_max_drr_with_push_drr(
                        filtered_inventory_data, 
                        target_date, 
                        future_date,
                        manual_drr_max if use_manual_drr_max else None
                    )

                    if not max_drr_results.empty:
                        # Visualization
                        st.subheader("Maximum DRR Distribution")
                        fig = px.bar(
                            max_drr_results,
                            x='Product Name',
                            y='Max DRR',
                            title='Maximum Sustainable DRR by Product'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

                        # Summary Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Max DRR", f"{max_drr_results['Max DRR'].mean():.2f}")
                        with col2:
                            st.metric("Median Max DRR", f"{max_drr_results['Max DRR'].median():.2f}")
                        with col3:
                            st.metric("Total Products", len(max_drr_results))

                        # Detailed Results Table
                        st.subheader("Detailed Results")
                        st.dataframe(max_drr_results)

                        # Download Button
                        st.download_button(
                            label="Download Results",
                            data=max_drr_results.to_csv(index=False),
                            file_name="max_drr_results.csv",
                            mime="text/csv"
                        )
            with tabs[6]:
                st.title("Daily DRR Calculator")
                
                if uploaded_file:
                    sheet_name = "Inventory"  # Use inventory sheet directly
                    target_date = st.date_input("Select Target Date", min_value=datetime.today())

                    if st.button("Calculate DRR"):
                        output = calculate_daily_drr(uploaded_file, sheet_name, target_date)
                        st.success("Calculation Complete!")
                        st.dataframe(output)

                        # Option to download the results
                        csv = output.to_csv().encode('utf-8')
                        st.download_button(
                            label="Download DRR Results as CSV",
                            data=csv,
                            file_name='daily_drr_results.csv',
                            mime='text/csv'
                        )
                else:
                    st.warning("Please upload an Excel file to proceed.")
            
            with tabs[7]:
                st.header("Labels Planning")
                
                if uploaded_file is not None:
                    target_date = st.date_input(
                        "Select Target Date for Label Planning",
                        value=datetime.now() + timedelta(days=30),
                        min_value=datetime.now(),
                        key="label_target_date"
                    )

        # Get the label plan first
                    label_plan = process_label_planning(uploaded_file, inventory_status, target_date)
        
                    if not label_plan.empty:
            # Apply filters to label plan
                        filtered_label_plan = label_plan.copy()
            
            # Apply date filter if selected
                        if selected_dates:
                            filtered_label_plan = filtered_label_plan[
                                filtered_label_plan['Date'].isin(selected_dates)
                ]
            
            # Apply ASIN filter if selected
                        if selected_asins:
                            filtered_label_plan = filtered_label_plan[
                                filtered_label_plan['ASIN'].isin(selected_asins)
                            ]
            
            # Apply product filter if selected
                        if selected_products:
                            filtered_label_plan = filtered_label_plan[
                                filtered_label_plan['Product Name'].isin(selected_products)
                            ]

                        st.subheader("Label Plan")
                        display_columns = [
                'Date', 'ASIN', 'Product Name', 'Current Inventory', 
                'Daily_Run_Rate', 'Expected_Usage', 'IN Stocks', 
                'Packed', 'New Orders', 'Total Available Label (Stocks+Packed)', 
                'Required Labels (Stocks+Packed+New Orders)'
                        ]
            
                    if all(col in filtered_label_plan.columns for col in display_columns):
                # Display metrics for the filtered data
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            total_products = len(filtered_label_plan['Product Name'].unique())
                            st.metric("Total Products", total_products)
                        with col2:
                            total_inventory = filtered_label_plan['Current Inventory'].sum()
                            st.metric("Total Current Inventory", f"{total_inventory:,.0f}")
                        with col3:
                            total_required = filtered_label_plan['Required Labels (Stocks+Packed+New Orders)'].sum()
                            st.metric("Total Required Inventory", f"{total_required:,.0f}")

                # Display the filtered dataframe
                        st.dataframe(filtered_label_plan[display_columns])
                
                # # Add visualizati        
                # Download button for filtered data
                        st.download_button(
                                label="Download Filtered Label Plan",
                            data=filtered_label_plan.to_csv(index=False),
                            file_name="filtered_label_plan.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Some expected columns are missing in the Label Plan data.")
                else:
                    st.warning("No valid label data found in the uploaded file.")

            with tabs[8]:
                st.header("Target Sales Management")
        # Load or initialize data
                if 'data' not in st.session_state:
                    try:
                        st.session_state.data = pd.read_csv('target_sales_data.csv')
                    except FileNotFoundError:
                        st.session_state.data = pd.DataFrame(columns=['Month', 'ASIN', 'Product_Name', 'Units', 'Price', 'Total'])
                data = st.session_state.data
        
        # Define month days
                month_days = {
            'January': 31, 'February': 28, 'March': 31, 'April': 30,
            'May': 31, 'June': 30, 'July': 31, 'August': 31,
            'September': 30, 'October': 31, 'November': 30, 'December': 31
                }
        
        # Predefined products and ASINs
                products = pd.DataFrame([
    {'ASIN': 'B07GNLN5K2', 'Product_Name': 'SVA Peppermint Arvensis 4 Oz'},
    {'ASIN': 'B07GCQDX6M', 'Product_Name': 'SVA Citronella Oil 4 Oz'},
    {'ASIN': 'B072M2MTK1', 'Product_Name': 'SVA Rose Water 4 Oz (US)'},
]
)
        
        # Select month
                month = st.selectbox("Select Month", options=list(month_days.keys()))
        
        # Initialize or load editable data
                if 'editable_data' not in st.session_state or st.session_state.selected_month != month:
                    st.session_state.selected_month = month
                    existing_data = data[data['Month'] == month]
                    if existing_data.empty:
                        editable_data = products.copy()
                        editable_data['Month'] = month
                        editable_data['Units'] = 0
                        editable_data['Price'] = 0.0
                        editable_data['Total'] = 0.0
                    else:
                        editable_data = existing_data[['Month', 'ASIN', 'Product_Name', 'Units', 'Price', 'Total']].copy()
                    st.session_state.editable_data = editable_data
        
                enable_edit = st.toggle("Enable Editing")
        
                st.subheader(f"Set Targets for {month}")
                if enable_edit:
                    edited_data = st.data_editor(
                        st.session_state.editable_data[['ASIN', 'Product_Name', 'Units', 'Price']],
                        num_rows="fixed", key="edit_table"
                    )
                    if st.button("Save Targets"):
                        edited_data['Month'] = month
                        edited_data['Total'] = edited_data['Units'] * edited_data['Price'] * month_days[month]
                
                        data = data[data['Month'] != month]
                        data = pd.concat([data, edited_data], ignore_index=True)
                
                        st.session_state.data = data
                        data.to_csv('target_sales_data.csv', index=False)
                        st.success(f"Targets for {month} saved successfully!")
                else:
                    st.dataframe(st.session_state.editable_data[['ASIN', 'Product_Name', 'Units', 'Price', 'Total']])
     
            with tabs[9]:  # Ensure tab indexing is correct
                st.header("US Products Shipment Planning")

                if not us_products_data.empty:
                    # st.subheader("Fetched AWD, Backstock, and Upcoming Orders for US Products")
                    # st.dataframe(us_products_data)

                    target_date = st.date_input("Select Target Date for Shipment Planning",
                                                value=datetime.now() + timedelta(days=30),
                                                min_value=datetime.now(),
                                                key="us_target_date")

                    us_shipment_plan = calculate_us_shipment_plan(filtered_inventory_status, us_products_data, target_date)

                    st.subheader("Updated Shipment Plan (US Products)")
                    display_columns =[
                    'ASIN', 'Product Name', 'Current Inventory', 
                    'Daily_Run_Rate', 'Expected_Usage', 'Total Upcoming Shipment', 
                    'AWD','Backstock',"Upcoming Orders",'Required Inventory(AWD+BS+ORDERS)']
                    st.dataframe(us_shipment_plan[display_columns])


                    # # Visualization
                    # fig_us_shipment = px.bar(us_shipment_plan, x='Product Name', y='Final Required Shipment',
                    #                          title='Updated Required Shipment Quantities (US Products)')
                    # fig_us_shipment.update_layout(xaxis_tickangle=-45)
                    # st.plotly_chart(fig_us_shipment, use_container_width=True)
                    
                    # Download option
                    st.download_button(label="Download US Shipment Plan",
                                       data=us_shipment_plan.to_csv(index=False),
                                       file_name="us_shipment_plan.csv",
                                       mime="text/csv")
                else:
                    st.warning("No US Products data found in the uploaded file.")

        except Exception as e:
            st.error("Error processing data")
            st.exception(e)


        with tabs[10]:
                st.title("📊 Sales & Profit Analysis Tool")
                st.write("Upload an Excel file containing sales and profit data to analyze the average sales and profit for a selected date range.")

                Performance_Tracker(uploaded_file)


        if has_permission('read'):
             st.title("SVA Analytics")
       
    else:
        pass

if __name__ == "__main__":
    main()
