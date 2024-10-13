import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from dotenv import load_dotenv
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import threading
import sys
import time

# Set default renderer to browser
pio.renderers.default = 'browser'

# Load the API keys from the .env file for Alpha Vantage and FRED
load_dotenv()
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
fred_api_key = os.getenv("FRED_API_KEY")

# Set the base URL for the Alpha Vantage API
base_url = "https://www.alphavantage.co/query"

# Initialize FRED API
fred = Fred(api_key=fred_api_key)

# Directory to save CSV files
CSV_DIR = 'financial_data_csv'


# ------------ CSV Backup Helper Functions ------------
def save_to_csv(dataframe, file_name):
    os.makedirs(CSV_DIR, exist_ok=True)  # Create the directory if it doesn't exist
    file_path = os.path.join(CSV_DIR, file_name)
    dataframe.to_csv(file_path)
    print(f"Data saved to {file_path}")


def load_from_csv(file_name):
    file_path = os.path.join(CSV_DIR, file_name)
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(f"CSV file {file_path} not found.")


# ------------ Data Fetching with CSV Backup ------------
def fetch_data(api_key, function, symbol, report_type):
    file_name = f"{symbol}_{function}_{report_type}.csv"
    try:
        params = {"function": function, "symbol": symbol, "apikey": api_key}
        response = requests.get(base_url, params=params)
        data = response.json()
        if "Note" in data:
            raise RuntimeError("API call frequency exceeded. Please wait and try again later.")
        if "annualReports" in data or "quarterlyReports" in data:
            df = pd.DataFrame(data["annualReports"])
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df.set_index('fiscalDateEnding', inplace=True)
            df = df.sort_index(ascending=True)
            # Keep only the last five years
            df = df.tail(5)
            df = df.apply(pd.to_numeric, errors='coerce')
            save_to_csv(df, file_name)
            return df
        elif function == "OVERVIEW":
            df = pd.DataFrame([data])
            save_to_csv(df, file_name)
            return df
        else:
            raise ValueError(f"Unexpected data format for {report_type}: {data}")
    except Exception as e:
        print(f"Error fetching {report_type} data from API: {e}")
        # Load from CSV backup if API fails
        try:
            return load_from_csv(file_name)
        except FileNotFoundError as fe:
            print(f"Backup CSV not found for {symbol} {report_type}.")
            raise fe


# ------------ FRED API Data ------------
def get_fred_rate(series_id):
    data = fred.get_series(series_id).dropna()
    latest_rate = data.iloc[-1] / 100
    return latest_rate


def get_10_year_treasury():
    return get_fred_rate("DGS10")  # 10-Year Treasury Yield


def get_real_gdp_data():
    return fred.get_series("GDPC1")


def calculate_long_term_gdp_growth(gdp_data):
    gdp_annual_growth = gdp_data.pct_change(4).dropna()
    return gdp_annual_growth.mean()


# ------------ Company-Specific Data ------------
def get_company_beta(symbol):
    # Fetch beta from Alpha Vantage Overview
    overview_data = fetch_data(alpha_vantage_api_key, "OVERVIEW", symbol, "Overview")
    beta = overview_data['Beta'].astype(float).values[0]
    return beta


def calculate_effective_tax_rate(income_stmt_df):
    income_before_tax = income_stmt_df['incomeBeforeTax'].astype(float)
    income_tax_expense = income_stmt_df['incomeTaxExpense'].astype(float)
    # Avoid division by zero
    effective_tax_rate = (income_tax_expense / income_before_tax).replace([np.inf, -np.inf], np.nan).fillna(0)
    # Use the most recent tax rate
    tax_rate = effective_tax_rate.iloc[-1]
    return tax_rate


# ------------ Financial Calculations ------------
def calc_fcff(cash_flow_df, income_stmt_df, tax_rate):
    required_fields = ['operatingCashflow', 'capitalExpenditures']
    for field in required_fields:
        if field not in cash_flow_df.columns:
            raise ValueError(f"Missing required field in Cash Flow Statement: {field}")

    ocf = cash_flow_df['operatingCashflow']
    capex = cash_flow_df['capitalExpenditures']
    int_expense = income_stmt_df.get('interestExpense', pd.Series(0, index=income_stmt_df.index))
    fcff = ocf - capex + (int_expense * (1 - tax_rate))
    return fcff


def calc_wacc(income_stmt_df, balance_sheet_df, risk_free_rate, market_return, beta, tax_rate):
    int_expense = income_stmt_df.get('interestExpense', pd.Series(0)).astype(float).iloc[-1]
    total_debt = balance_sheet_df.get('shortTermDebt', pd.Series(0)).fillna(0).astype(float).iloc[-1] + \
                 balance_sheet_df.get('longTermDebt', pd.Series(0)).fillna(0).astype(float).iloc[-1]
    total_equity = balance_sheet_df.get('totalShareholderEquity', pd.Series(0)).astype(float).iloc[-1]

    # Cost of debt
    if total_debt > 0 and int_expense != 0:
        cost_of_debt = int_expense / total_debt
    else:
        cost_of_debt = 0

    # Cost of equity
    cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)

    # Weights
    total_value = total_equity + total_debt
    weight_equity = total_equity / total_value if total_value != 0 else 0
    weight_debt = total_debt / total_value if total_value != 0 else 0

    # WACC calculation
    wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
    return wacc


def calc_terminal_value(fcff_last, wacc, terminal_growth_rate):
    return (fcff_last * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)


# ------------ Projection and Discounting ------------
def project_future_cash_flows(fcff_current, current_year, growth_rate, years):
    future_fcff = []
    for year in range(1, years + 1):
        projected_fcff = fcff_current * (1 + growth_rate) ** year
        year_end = pd.Timestamp(f'{current_year + year}-12-31')
        future_fcff.append({"Year End": year_end, "Projected FCFF": projected_fcff})
    return pd.DataFrame(future_fcff).set_index("Year End")


def discount_future_fcff(future_fcff_df, wacc):
    future_fcff_df['Discount Factor'] = [(1 / (1 + wacc) ** year) for year in range(1, len(future_fcff_df) + 1)]
    future_fcff_df['PV of FCFF'] = future_fcff_df['Projected FCFF'] * future_fcff_df['Discount Factor']
    return future_fcff_df


def calc_equity_value(future_fcff_df, terminal_value, total_debt, cash_and_equivalents, wacc):
    enterprise_value = future_fcff_df['PV of FCFF'].sum() + (terminal_value / (1 + wacc) ** len(future_fcff_df))
    equity_value = enterprise_value - total_debt + cash_and_equivalents
    return equity_value


# ------------ Plotting with Plotly ------------
def combine_fcff(fcff_df, discounted_fcff_df):
    """
    Combine historical FCFF with PV of projected FCFF into one DataFrame.
    """
    # Rename 'PV of FCFF' in the projected FCFF DataFrame to 'FCFF' for consistency
    projected_fcff_df = discounted_fcff_df[['PV of FCFF']].rename(columns={'PV of FCFF': 'FCFF'})

    # Concatenate historical FCFF and projected PV of FCFF
    combined_fcff_df = pd.concat([fcff_df.to_frame(name='FCFF'), projected_fcff_df])

    return combined_fcff_df

def plot_fcff_combined_with_financials(symbol, fcff_df, combined_fcff_df, income_stmt_df, balance_sheet_df):
    fig = go.Figure()

    # Plot historical FCFF
    fig.add_trace(go.Scatter(
        x=fcff_df.index,
        y=fcff_df.values / 1e9,
        mode='lines+markers',
        name='Historical FCFF',
        line=dict(color='blue')
    ))

    # Add trend line for historical FCFF (Linear Regression)
    X_hist = np.arange(len(fcff_df)).reshape(-1, 1)  # Time indices as independent variable for historical FCFF
    y_hist = fcff_df.values  # Historical FCFF values

    model_hist = LinearRegression()
    model_hist.fit(X_hist, y_hist)
    trend_line_hist = model_hist.predict(X_hist)  # Predicted trend line for historical FCFF

    # Plot the historical FCFF trend line
    fig.add_trace(go.Scatter(
        x=fcff_df.index,
        y=trend_line_hist / 1e9,
        mode='lines',
        name='Trend Line (Historical FCFF)',
        line=dict(color='red', dash='dot')
    ))

    # Plot combined FCFF (Historical + PV of FCFF for projected years)
    combined_fcff_years = combined_fcff_df.index
    combined_fcff_values = combined_fcff_df['FCFF'].values

    fig.add_trace(go.Scatter(
        x=combined_fcff_years,
        y=combined_fcff_values / 1e9,
        mode='lines+markers',
        name='Combined FCFF',
        line=dict(color='green')
    ))

    # Polynomial regression model (degree=6 for more complexity)
    combined_years = np.arange(len(combined_fcff_df)).reshape(-1, 1)  # Time indices for combined FCFF
    combined_fcff = combined_fcff_df['FCFF'].values  # Combined FCFF (historical + projected)

    poly_model = make_pipeline(PolynomialFeatures(degree=6), LinearRegression())
    poly_model.fit(combined_years, combined_fcff)
    trend_line_combined_poly = poly_model.predict(combined_years)  # Predicted trend line for combined FCFF (degree 6)

    # Plot the combined FCFF polynomial trend line (Degree 6)
    fig.add_trace(go.Scatter(
        x=combined_fcff_df.index,
        y=trend_line_combined_poly / 1e9,
        mode='lines',
        name='Polynomial Trend Line (Degree 6 Combined FCFF)',
        line=dict(color='orange', dash='dash')
    ))

    # Exponential regression for combined FCFF (filtering out non-positive FCFF values)
    positive_fcff_mask = combined_fcff > 0  # Filter out non-positive values
    X_exp = combined_years[positive_fcff_mask].reshape(-1, 1)  # Use only the positive values for exponential regression
    log_combined_fcff = np.log(combined_fcff[positive_fcff_mask])  # Take the log of positive FCFF values
    exp_model = LinearRegression()
    exp_model.fit(X_exp, log_combined_fcff)  # Fit the linear model to log-transformed FCFF
    trend_line_exp = np.exp(exp_model.predict(combined_years))  # Revert the log to get exponential predictions

    # Plot the combined FCFF exponential trend line
    fig.add_trace(go.Scatter(
        x=combined_fcff_df.index,
        y=trend_line_exp / 1e9,
        mode='lines',
        name='Exponential Trend Line (Combined FCFF)',
        line=dict(color='purple', dash='dot')
    ))

    # Add linear regression for combined FCFF (historical + projected)
    lin_model_combined = LinearRegression()
    lin_model_combined.fit(combined_years, combined_fcff)
    trend_line_combined_lin = lin_model_combined.predict(combined_years)

    # Plot the linear regression trend line for combined FCFF
    fig.add_trace(go.Scatter(
        x=combined_fcff_df.index,
        y=trend_line_combined_lin / 1e9,
        mode='lines',
        name='Linear Trend Line (Combined FCFF)',
        line=dict(color='brown', dash='solid')
    ))

    # Add bar charts for selected financial statement items (from Income Statement and Balance Sheet)
    fig.add_trace(go.Bar(
        x=income_stmt_df.index,
        y=income_stmt_df['totalRevenue'] / 1e9,
        name='Total Revenue',
        marker=dict(color='orange'),
        opacity=0.6
    ))

    fig.add_trace(go.Bar(
        x=income_stmt_df.index,
        y=income_stmt_df['netIncome'] / 1e9,
        name='Net Income',
        marker=dict(color='purple'),
        opacity=0.6
    ))

    fig.add_trace(go.Bar(
        x=balance_sheet_df.index,
        y=balance_sheet_df['totalAssets'] / 1e9,
        name='Total Assets',
        marker=dict(color='cyan'),
        opacity=0.6
    ))

    fig.add_trace(go.Bar(
        x=balance_sheet_df.index,
        y=balance_sheet_df['totalLiabilities'] / 1e9,
        name='Total Liabilities',
        marker=dict(color='pink'),
        opacity=0.6
    ))

    # Layout adjustments
    fig.update_layout(
        title=f"{symbol} - Historical and Combined FCFF with Polynomial (Degree 6), Exponential, and Linear Trend Lines, PV of FCFF, Financial Statement Items (in Billions USD)",
        xaxis_title="Year",
        yaxis_title="Value (Billions USD)",
        barmode='group',  # Group bars side by side
        showlegend=True
    )

    fig.show()


# ------------ Timed Input Function ------------
def timed_input(prompt, timeout=15, default="AMZN"):
    result = [default]  # Use list for mutable storage to capture input value

    def get_input():
        user_input = input(prompt)  # Capture user input
        result[0] = user_input.strip() if user_input.strip() else default  # Use default if input is empty

    input_thread = threading.Thread(target=get_input)  # Create a thread to wait for input
    input_thread.daemon = True  # Set as daemon so it doesn't block program exit
    input_thread.start()
    input_thread.join(timeout)  # Wait for the specified timeout

    if input_thread.is_alive():  # If the thread is still alive, timeout occurred
        print(f"\nNo input within {timeout} seconds, proceeding with default '{default}'.")
        return default
    else:
        return result[0]  # Return the captured input or default if empty


# ------------ Main Execution ------------
def main():
    try:
        # Prompt for ticker symbol with 5-second timeout, defaulting to AMZN if input is empty or timeout occurs
        symbol = timed_input("Enter the stock ticker symbol (default: AMZN): ", timeout=5, default="AMZN").strip().upper()

        # Prompt for number of years of FCFF analysis with 5-second timeout, defaulting to 5 if input is empty or timeout occurs
        years_input = timed_input("Enter the number of years of FCFF analysis (default: 5): ", timeout=5, default="5").strip()

        try:
            years_of_analysis = int(years_input) if years_input else 5
        except ValueError:
            print("Invalid input. Using default value of 5 years.")
            years_of_analysis = 5

        print(f"\n{'='*60}\nProcessing {symbol} with {years_of_analysis} years of FCFF analysis\n{'='*60}")

        # Fetch financial data from Alpha Vantage
        income_stmt_df = fetch_data(alpha_vantage_api_key, "INCOME_STATEMENT", symbol, "Income Statement")
        balance_sheet_df = fetch_data(alpha_vantage_api_key, "BALANCE_SHEET", symbol, "Balance Sheet")
        cash_flow_df = fetch_data(alpha_vantage_api_key, "CASH_FLOW", symbol, "Cash Flow")

        # Display financial statements in billions
        print("\nIncome Statement (in Billions USD):")
        print(income_stmt_df[['totalRevenue', 'netIncome']] / 1e9)
        print("\nBalance Sheet (in Billions USD):")
        print(balance_sheet_df[['totalAssets', 'totalLiabilities']] / 1e9)
        print("\nCash Flow Statement (in Billions USD):")
        print(cash_flow_df[['operatingCashflow', 'capitalExpenditures']] / 1e9)

        # Get beta
        beta = get_company_beta(symbol)
        print(f"Beta for {symbol}: {beta:.2f}")

        # Calculate tax rate
        tax_rate = calculate_effective_tax_rate(income_stmt_df)
        print(f"Effective Tax Rate for {symbol}: {tax_rate:.2%}")

        # Calculate FCFF
        fcff_df = calc_fcff(cash_flow_df, income_stmt_df, tax_rate)
        print(f"\nFCFF (in Billions USD):\n{fcff_df / 1e9}")

        # Fetch 10-year treasury rate and long-term GDP growth rate
        risk_free_rate = get_10_year_treasury()
        gdp_data = get_real_gdp_data()
        long_term_gdp_growth = calculate_long_term_gdp_growth(gdp_data)

        print(f"\nRisk-Free Rate (10-Year): {risk_free_rate:.4%}")
        print(f"Long-Term GDP Growth Rate (Terminal Growth Rate): {long_term_gdp_growth:.4%}")

        # Get the last FCFF value and current year
        fcff_latest = fcff_df.iloc[-1]
        current_year = fcff_df.index[-1].year

        # Calculate FCFF growth rate
        fcff_growth_rates = fcff_df.pct_change().dropna()

        if (fcff_df <= 0).any() or fcff_growth_rates.mean() <= 0:  # If any FCFF is negative or zero, or FCFF growth rate is <= 0
            print("\nNegative or zero FCFF growth detected, using revenue growth rate instead.")
            # Calculate revenue growth rate
            revenue_growth_rates = income_stmt_df['totalRevenue'].pct_change().dropna()
            if not revenue_growth_rates.empty:
                avg_growth_rate = revenue_growth_rates.mean()  # Use revenue growth rate if available
                print(f"\nUsing Revenue Growth Rate: {avg_growth_rate:.2%}")
            else:
                avg_growth_rate = 0.05  # Default growth rate if no revenue growth data is available
                print("\nNo Revenue Growth data available, using default growth rate of 0.05.")
        else:
            # Use FCFF growth rate
            avg_growth_rate = fcff_growth_rates.mean()
            print(f"\nCalculated FCFF Growth Rate (Average): {avg_growth_rate:.2%}")


        # Project future FCFF for the specified number of years
        projected_fcff_df = project_future_cash_flows(fcff_latest, current_year, avg_growth_rate, years_of_analysis)

        # Assume market return
        market_return = 0.08

        # Calculate WACC
        wacc = calc_wacc(income_stmt_df, balance_sheet_df, risk_free_rate, market_return, beta, tax_rate)
        print(f"\nWACC: {wacc:.4%}")

        # Discount projected FCFF to present value
        discounted_fcff_df = discount_future_fcff(projected_fcff_df, wacc)

        # Calculate terminal value
        terminal_value = calc_terminal_value(projected_fcff_df['Projected FCFF'].iloc[-1], wacc, long_term_gdp_growth)
        print(f"\nTerminal Value (in Billions USD): ${terminal_value / 1e9:.2f}")

        total_debt = balance_sheet_df.get('shortTermDebt', pd.Series(0)).fillna(0).astype(float).iloc[-1] + \
                     balance_sheet_df.get('longTermDebt', pd.Series(0)).fillna(0).astype(float).iloc[-1]

        # Get cash and cash equivalents
        cash_and_equivalents = balance_sheet_df.get('cashAndCashEquivalentsAtCarryingValue', pd.Series(0)).astype(float).iloc[-1]

        # Calculate equity value
        equity_value = calc_equity_value(discounted_fcff_df, terminal_value, total_debt, cash_and_equivalents, wacc)

        # Get shares outstanding
        shares_outstanding = balance_sheet_df.get('commonStockSharesOutstanding', pd.Series(0)).astype(float).iloc[-1]

        # Ensure implied_share_price is a scalar
        implied_share_price = equity_value / shares_outstanding if shares_outstanding != 0 else 0

        # Display the implied share price
        print(f"\nImplied Share Price: ${implied_share_price:.2f}")

        # Combine historical and projected FCFF
        fcff_df_combined = combine_fcff(fcff_df, discounted_fcff_df)

        # Plot the results with ticker symbol in the title
        plot_fcff_combined_with_financials(symbol, fcff_df, fcff_df_combined, income_stmt_df, balance_sheet_df)

    except Exception as e:
        print(f"An error occurred while processing {symbol}: {e}")


if __name__ == "__main__":
    main()