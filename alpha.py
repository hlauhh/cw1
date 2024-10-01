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

# Set default renderer to browser
pio.renderers.default = 'browser'

# Load the API keys from the .env file for Alpha Vantage and FRED
load_dotenv()
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
fred_api_key = os.getenv("FRED_API_KEY")

# Set the base URL for the Alpha Vantage API and stock symbol
base_url = "https://www.alphavantage.co/query"
symbol = "AMZN"  # Amazon stock symbol

# Initialize FRED API
fred = Fred(api_key=fred_api_key)


# ------------ Data Fetching ------------
def fetch_data(api_key, function, symbol):
    params = {"function": function, "symbol": symbol, "apikey": api_key}
    response = requests.get(base_url, params=params)
    return response.json()


def json_to_dataframe(data, report_type):
    if "annualReports" in data:
        df = pd.DataFrame(data["annualReports"])
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        df.set_index('fiscalDateEnding', inplace=True)
        df = df.sort_index(ascending=True)
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


# ------------ Financial Calculations ------------
def calc_fcff(cash_flow_df, income_stmt_df):
    ocf = cash_flow_df['operatingCashflow']
    capex = cash_flow_df['capitalExpenditures']
    int_expense = income_stmt_df.get('interestExpense', 0)
    tax_rate = 0.178
    return ocf - capex + (int_expense * (1 - tax_rate))


def calc_wacc(income_stmt_df, balance_sheet_df, risk_free_rate, market_return=0.08, beta=1.16, tax_rate=0.178):
    int_expense = income_stmt_df['interestExpense'].astype(float).iloc[-1]
    total_debt = balance_sheet_df['shortTermDebt'].fillna(0) + balance_sheet_df['longTermDebt'].fillna(0)
    total_debt = total_debt.iloc[-1]
    total_equity = balance_sheet_df['totalShareholderEquity'].astype(float).iloc[-1]
    cost_of_debt = int_expense / total_debt
    cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
    debt_equity_ratio = total_debt / total_equity
    return ((1 / (1 + debt_equity_ratio)) * cost_of_equity +
            (debt_equity_ratio / (1 + debt_equity_ratio)) * cost_of_debt * (1 - tax_rate))


def calc_terminal_value(fcff_last, wacc, terminal_growth_rate):
    return (fcff_last * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)


# ------------ Projection and Discounting ------------
def project_future_cash_flows(fcff_current, current_year, growth_rate, years):
    """
    Project future FCFF for a given number of years using the historical average growth rate.
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


def calc_equity_value(future_fcff_df, terminal_value, market_debt):
    enterprise_value = future_fcff_df['PV of FCFF'].sum() + terminal_value
    equity_value = enterprise_value - market_debt
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


def plot_fcff_combined_with_financials(fcff_df, combined_fcff_df, income_stmt_df, balance_sheet_df):
    """
    Plot combined historical FCFF with trend lines for both historical FCFF and combined FCFF using Polynomial Regression (Degree 3),
    Exponential Regression (ignoring non-positive FCFF values), Linear Regression, and a bar chart for selected financial statement items.
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

    # Polynomial regression model (degree=3 for more complexity)
    combined_years = np.arange(len(combined_fcff_df)).reshape(-1, 1)  # Time indices for combined FCFF
    combined_fcff = combined_fcff_df['FCFF'].values  # Combined FCFF (historical + projected)

    poly_model = make_pipeline(PolynomialFeatures(degree=6), LinearRegression())
    poly_model.fit(combined_years, combined_fcff)
    trend_line_combined_poly = poly_model.predict(combined_years)  # Predicted trend line for combined FCFF (degree 3)

    # Plot the combined FCFF polynomial trend line (Degree 3)
    fig.add_trace(go.Scatter(
        x=combined_fcff_df.index,
        y=trend_line_combined_poly / 1e9,
        mode='lines',
        name='Polynomial Trend Line (Degree 3 Combined FCFF)',
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
        title="Historical and Combined FCFF with Polynomial (Degree 3), Exponential, and Linear Trend Lines, PV of FCFF, and Financial Statement Items (in Billions USD)",
        xaxis_title="Year",
        yaxis_title="Value (Billions USD)",
        barmode='group',  # Group bars side by side
        showlegend=True
    )

    fig.show()



# ------------ Main Execution ------------
def main():
    # Fetch financial data from Alpha Vantage
    income_stmt = fetch_data(alpha_vantage_api_key, "INCOME_STATEMENT", symbol)
    balance_sheet = fetch_data(alpha_vantage_api_key, "BALANCE_SHEET", symbol)
    cash_flow = fetch_data(alpha_vantage_api_key, "CASH_FLOW", symbol)

    # Convert JSON data to DataFrames
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

    # Calculate FCFF
    fcff_df = calc_fcff(cash_flow_df, income_stmt_df)
    print(f"\nFCFF (in Billions USD): {fcff_df / 1e9}")
    print(f"\nSum of FCFF (in Billions USD): {fcff_df.sum() / 1e9}")

    # Fetch risk-free rate and long-term GDP growth rate
    risk_free_rate = get_5_year_treasury()
    gdp_data = get_real_gdp_data()
    long_term_gdp_growth = calculate_long_term_gdp_growth(gdp_data)

    print(f"\nRisk-Free Rate (5-Year): {risk_free_rate:.4f}")
    print(f"Long-Term GDP Growth Rate (Terminal Growth Rate): {long_term_gdp_growth:.4f}")

    # Get the last FCFF value and current year
    fcff_latest = fcff_df.iloc[-1]
    current_year = fcff_df.index[-1].year

    # Project future FCFF for 5 years with year-end index
    avg_growth_rate = fcff_df.pct_change().dropna().mean()
    projected_fcff_df = project_future_cash_flows(fcff_latest, current_year, avg_growth_rate, 5)

    # Calculate WACC
    wacc = calc_wacc(income_stmt_df, balance_sheet_df, risk_free_rate, beta=1.16)
    print(f"\nWACC: {wacc:.4f}")

    # Discount projected FCFF to present value
    discounted_fcff_df = discount_future_fcff(projected_fcff_df, wacc)

    # Calculate terminal value using long-term GDP growth rate
    terminal_value = calc_terminal_value(discounted_fcff_df['Projected FCFF'].iloc[-1], wacc, long_term_gdp_growth)

    # If terminal_value is a Series, select the last value after sorting
    if isinstance(terminal_value, pd.Series):
        terminal_value_float = terminal_value.sort_index(ascending=True).iloc[-1]
    else:
        terminal_value_float = terminal_value

    print(f"\nTerminal Value: ${terminal_value_float / 1e9:.2f} Billions USD")

    # Get market value of debt
    market_debt = balance_sheet_df['shortTermDebt'].fillna(0) + balance_sheet_df['longTermDebt'].fillna(0)
    market_debt = market_debt.iloc[-1]

    # Calculate equity value
    equity_value = calc_equity_value(discounted_fcff_df, terminal_value_float, market_debt)

    # Get shares outstanding and calculate implied share price
    shares_outstanding = balance_sheet_df['commonStockSharesOutstanding'].astype(float).iloc[-1]

    # Ensure implied_share_price is a scalar
    implied_share_price = equity_value / shares_outstanding if shares_outstanding != 0 else 0

    # Display the implied share price
    print(f"\nImplied Share Price: ${implied_share_price:.2f}")

    # Combine historical and projected FCFF (using PV of FCFF)
    fcff_df_combined = combine_fcff(fcff_df, discounted_fcff_df)

    # Plot historical FCFF with trend lines, PV of FCFF, and financial statement items
    plot_fcff_combined_with_financials(fcff_df, fcff_df_combined, income_stmt_df, balance_sheet_df)


if __name__ == "__main__":
    main()

# TODO
# combine variables like market return, beta, tax rates
# try apply to [MAMAA] stocks with diff variables to achieve similar market prices
