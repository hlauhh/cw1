import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from dotenv import load_dotenv
from fredapi import Fred
from sklearn.linear_model import LinearRegression
import yfinance as yf

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


# ------------ Data Fetching ------------
def fetch_data(api_key, function, symbol):
    params = {"function": function, "symbol": symbol, "apikey": api_key}
    response = requests.get(base_url, params=params)
    data = response.json()
    if "Note" in data:
        raise RuntimeError("API call frequency exceeded. Please wait and try again later.")
    return data


def json_to_dataframe(data, report_type):
    if "annualReports" in data:
        df = pd.DataFrame(data["annualReports"])
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        df.set_index('fiscalDateEnding', inplace=True)
        df = df.sort_index(ascending=True)
        # Keep only the last five years
        df = df.tail(5)
        return df.apply(pd.to_numeric, errors='coerce')
    else:
        raise ValueError(f"Error fetching {report_type} data: {data}")


# ------------ FRED API Data ------------
def get_fred_rate(series_id):
    data = fred.get_series(series_id).dropna()
    latest_rate = data.iloc[-1] / 100
    return latest_rate


def get_5_year_treasury():
    return get_fred_rate("DGS5")


def get_real_gdp_data():
    return fred.get_series("GDPC1")


def calculate_long_term_gdp_growth(gdp_data):
    gdp_annual_growth = gdp_data.pct_change(4).dropna()
    return gdp_annual_growth.mean()


# ------------ Company-Specific Data ------------
def get_company_beta(symbol):
    # Fetch beta from Yahoo Finance
    ticker = yf.Ticker(symbol)
    beta = ticker.info.get('beta', 1)
    return beta


def calculate_effective_tax_rate(income_stmt_df):
    income_before_tax = income_stmt_df['incomeBeforeTax'].astype(float)
    income_tax_expense = income_stmt_df['incomeTaxExpense'].astype(float)
    # Avoid division by zero
    effective_tax_rate = (income_tax_expense / income_before_tax).replace([np.inf, -np.inf], np.nan).fillna(0)
    # Use the most recent tax rate
    return effective_tax_rate.iloc[-1]


def get_shares_outstanding(symbol):
    # Fetch shares outstanding from Yahoo Finance
    ticker = yf.Ticker(symbol)
    shares_outstanding = ticker.info.get('sharesOutstanding', 0)
    return shares_outstanding


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


def calc_wacc(symbol, income_stmt_df, balance_sheet_df, risk_free_rate, market_return, beta, tax_rate):
    # Fetch interest expense
    int_expense = income_stmt_df.get('interestExpense', pd.Series(0)).astype(float).iloc[-1]
    # Total debt
    total_debt = balance_sheet_df.get('shortTermDebt', pd.Series(0)).fillna(0).astype(float).iloc[-1] + \
                 balance_sheet_df.get('longTermDebt', pd.Series(0)).fillna(0).astype(float).iloc[-1]
    # Total equity
    total_equity = balance_sheet_df.get('totalShareholderEquity', pd.Series(0)).astype(float).iloc[-1]
    # Market capitalization
    shares_outstanding = get_shares_outstanding(symbol)
    current_stock_price = yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1]
    market_cap = shares_outstanding * current_stock_price if shares_outstanding and current_stock_price else total_equity

    # Cost of debt
    if total_debt > 0:
        cost_of_debt = int_expense / total_debt
    else:
        cost_of_debt = 0

    # Cost of equity
    cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)

    # Weights
    total_value = market_cap + total_debt
    weight_equity = market_cap / total_value if total_value != 0 else 0
    weight_debt = total_debt / total_value if total_value != 0 else 0

    # WACC calculation
    wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
    return wacc


def calc_terminal_value(fcff_last, wacc, terminal_growth_rate):
    return (fcff_last * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)


def calculate_cagr(fcff_series):
    n = len(fcff_series)
    if n < 2:
        return 0  # Not enough data to calculate CAGR
    if fcff_series.iloc[0] <= 0 or fcff_series.iloc[-1] <= 0:
        return fcff_series.pct_change().dropna().mean()
    cagr = (fcff_series.iloc[-1] / fcff_series.iloc[0]) ** (1 / (n - 1)) - 1
    return cagr


# ------------ Projection and Discounting ------------
def project_future_cash_flows(fcff_current, current_year, growth_rate, years):
    """
    Project future FCFF for a given number of years using the CAGR or average growth rate.
    The index will be set to the year-end date (e.g., 'YYYY-12-31').
    """
    future_fcff = []
    for year in range(1, years + 1):
        projected_fcff = fcff_current * (1 + growth_rate) ** year
        year_end = pd.Timestamp(f'{current_year + year}-12-31')  # Set index to year-end
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


# ------------ Market Debt Valuation ------------
def value_market_debt(total_debt, interest_expense, average_maturity_years=5, company_specific_yield=None):
    """
    Estimate the market value of debt using the present value of future debt payments.
    If company_specific_yield is not provided, use an estimated yield based on credit ratings.
    """
    if company_specific_yield is None:
        # Assume an average corporate bond yield if specific yield is not available
        company_specific_yield = get_fred_rate('BAMLC0A4CBBBEY')  # BofA Merrill Lynch US Corporate BBB Effective Yield

    # Estimate annual debt service (interest payments)
    if total_debt > 0 and interest_expense > 0:
        average_interest_rate = interest_expense / total_debt
        annual_debt_service = total_debt * average_interest_rate
    else:
        annual_debt_service = 0

    # Calculate present value of debt payments
    pv_debt_payments = annual_debt_service * (1 - (1 / (1 + company_specific_yield) ** average_maturity_years)) / company_specific_yield
    pv_debt_payments += total_debt / (1 + company_specific_yield) ** average_maturity_years  # Add PV of principal repayment

    return pv_debt_payments


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
    """
    Plot combined historical FCFF with trend lines and bar charts for selected financial statement items.
    """
    fig = go.Figure()

    # Plot historical FCFF
    fig.add_trace(go.Scatter(
        x=fcff_df.index,
        y=fcff_df.values / 1e9,
        mode='lines+markers',
        name='Historical FCFF',
        line=dict(color='blue')
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

    # Add linear regression for combined FCFF (historical + projected)
    combined_years = np.arange(len(combined_fcff_df)).reshape(-1, 1)
    combined_fcff = combined_fcff_df['FCFF'].values

    lin_model_combined = LinearRegression()
    lin_model_combined.fit(combined_years, combined_fcff)
    trend_line_combined_lin = lin_model_combined.predict(combined_years)

    # Plot the linear regression trend line for combined FCFF
    fig.add_trace(go.Scatter(
        x=combined_fcff_df.index,
        y=trend_line_combined_lin / 1e9,
        mode='lines',
        name='Linear Trend Line (Combined FCFF)',
        line=dict(color='red', dash='solid')
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
        title=f"{symbol} - FCFF and Financial Statements (in Billions USD)",
        xaxis_title="Year",
        yaxis_title="Value (Billions USD)",
        barmode='group',  # Group bars side by side
        showlegend=True
    )

    fig.show()


# ------------ Main Execution ------------
def main(symbol):
    try:
        print(f"\n{'='*60}\nProcessing {symbol}\n{'='*60}")

        # Fetch financial data from Alpha Vantage
        income_stmt = fetch_data(alpha_vantage_api_key, "INCOME_STATEMENT", symbol)
        balance_sheet = fetch_data(alpha_vantage_api_key, "BALANCE_SHEET", symbol)
        cash_flow = fetch_data(alpha_vantage_api_key, "CASH_FLOW", symbol)

        # Convert JSON data to DataFrames and keep only the last five years
        income_stmt_df = json_to_dataframe(income_stmt, "Income Statement")
        balance_sheet_df = json_to_dataframe(balance_sheet, "Balance Sheet")
        cash_flow_df = json_to_dataframe(cash_flow, "Cash Flow")

        # Display financial statements in billions
        print("\nIncome Statement (in Billions USD):")
        print(income_stmt_df[['totalRevenue', 'netIncome']] / 1e9)
        print("\nBalance Sheet (in Billions USD):")
        print(balance_sheet_df[['totalAssets', 'totalLiabilities']] / 1e9)
        print("\nCash Flow Statement (in Billions USD):")
        print(cash_flow_df[['operatingCashflow', 'capitalExpenditures']] / 1e9)

        # Calculate company-specific variables
        beta = get_company_beta(symbol)
        print(f"\nBeta for {symbol}: {beta:.2f}")

        tax_rate = calculate_effective_tax_rate(income_stmt_df)
        print(f"Effective Tax Rate for {symbol}: {tax_rate:.2%}")

        # Calculate FCFF
        fcff_df = calc_fcff(cash_flow_df, income_stmt_df, tax_rate)
        print(f"\nFCFF (in Billions USD):\n{fcff_df / 1e9}")

        # Fetch risk-free rate and long-term GDP growth rate
        risk_free_rate = get_5_year_treasury()
        gdp_data = get_real_gdp_data()
        long_term_gdp_growth = calculate_long_term_gdp_growth(gdp_data)

        print(f"\nRisk-Free Rate (5-Year): {risk_free_rate:.4%}")
        print(f"Long-Term GDP Growth Rate (Terminal Growth Rate): {long_term_gdp_growth:.4%}")

        # Get the last FCFF value and current year
        fcff_latest = fcff_df.iloc[-1]
        current_year = fcff_df.index[-1].year

        # Project future FCFF for 5 years with year-end index
        growth_rate = calculate_cagr(fcff_df)
        print(f"\nCalculated Growth Rate (CAGR): {growth_rate:.2%}")
        projected_fcff_df = project_future_cash_flows(fcff_latest, current_year, growth_rate, 5)

        # Assume market return
        market_return = 0.08

        # Calculate WACC
        wacc = calc_wacc(symbol, income_stmt_df, balance_sheet_df, risk_free_rate, market_return, beta, tax_rate)
        print(f"\nWACC: {wacc:.4%}")

        # Discount projected FCFF to present value
        discounted_fcff_df = discount_future_fcff(projected_fcff_df, wacc)

        # Calculate terminal value using long-term GDP growth rate
        terminal_value = calc_terminal_value(projected_fcff_df['Projected FCFF'].iloc[-1], wacc, long_term_gdp_growth)
        print(f"\nTerminal Value (in Billions USD): ${terminal_value / 1e9:.2f}")

        # Get market value of debt
        total_debt = balance_sheet_df.get('shortTermDebt', pd.Series(0)).fillna(0).astype(float).iloc[-1] + \
                     balance_sheet_df.get('longTermDebt', pd.Series(0)).fillna(0).astype(float).iloc[-1]
        interest_expense = income_stmt_df.get('interestExpense', pd.Series(0)).astype(float).iloc[-1]

        # Value market debt
        market_debt = value_market_debt(total_debt, interest_expense)
        print(f"\nMarket Value of Debt (in Billions USD): ${market_debt / 1e9:.2f}")

        # Get cash and cash equivalents
        cash_and_equivalents = balance_sheet_df.get('cashAndCashEquivalentsAtCarryingValue', pd.Series(0)).astype(float).iloc[-1]

        # Calculate equity value
        equity_value = calc_equity_value(discounted_fcff_df, terminal_value, market_debt, cash_and_equivalents, wacc)

        # Get shares outstanding and calculate implied share price
        shares_outstanding = get_shares_outstanding(symbol)
        if shares_outstanding == 0 or shares_outstanding is None:
            shares_outstanding = balance_sheet_df.get('commonStockSharesOutstanding', pd.Series(0)).astype(float).iloc[-1]

        implied_share_price = equity_value / shares_outstanding if shares_outstanding != 0 else 0

        # Display the implied share price
        print(f"\nImplied Share Price: ${implied_share_price:.2f}")

        # Combine historical and projected FCFF (using PV of FCFF)
        fcff_df_combined = combine_fcff(fcff_df, discounted_fcff_df)

        # Plot historical FCFF with trend lines, PV of FCFF, and financial statement items
        plot_fcff_combined_with_financials(symbol, fcff_df, fcff_df_combined, income_stmt_df, balance_sheet_df)

    except Exception as e:
        print(f"An error occurred while processing {symbol}: {e}")


def run_valuation_for_stocks():
    # List of MAMAA stock symbols
    stock_symbols = ['META', 'AMZN', 'MSFT', 'AAPL', 'GOOG']

    for symbol in stock_symbols:
        main(symbol)


if __name__ == "__main__":
    run_valuation_for_stocks()
