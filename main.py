import yfinance as yf
import pandas as pd

# Retrieve Amazon's financial data
ticker = "AMZN"
amazon = yf.Ticker(ticker)

# Get financials and cash flow data
income_statement = amazon.financials.transpose()  # Income statement
cash_flow = amazon.cashflow.transpose()  # Cash flow
balance_sheet = amazon.balance_sheet.transpose()  # Balance sheet

# Extracting key data for FCFF calculation:
# EBIT = Operating Income
ebit = income_statement['EBIT'].iloc[0]

# Tax rate (using income tax expense / pretax income as a proxy)
tax_expense = income_statement['Income Tax Expense'].iloc[0]
pretax_income = income_statement['Earnings Before Tax'].iloc[0]
tax_rate = tax_expense / pretax_income if pretax_income != 0 else 0

# Depreciation (from cash flow statement)
depreciation = cash_flow['Depreciation'].iloc[0]

# CAPEX (from cash flow statement)
capex = cash_flow['Capital Expenditures'].iloc[0]

# Change in Net Working Capital (NWC)
# NWC = Current Assets - Current Liabilities
current_assets = balance_sheet['Total Current Assets'].iloc[0]
current_liabilities = balance_sheet['Total Current Liabilities'].iloc[0]
nwc_current = current_assets - current_liabilities

# Get previous year NWC to compute delta
nwc_previous = balance_sheet['Total Current Assets'].iloc[1] - balance_sheet['Total Current Liabilities'].iloc[1]
delta_nwc = nwc_current - nwc_previous

# FCFF Calculation
fcff = ebit * (1 - tax_rate) + depreciation - capex - delta_nwc

# Display extracted data and FCFF result
financial_data = pd.DataFrame({
    'EBIT': [ebit],
    'Tax Rate': [tax_rate],
    'Depreciation': [depreciation],
    'CAPEX': [capex],
    'Delta NWC': [delta_nwc],
    'FCFF': [fcff]
})

import ace_tools as tools; tools.display_dataframe_to_user(name="Amazon Financial Data and FCFF", dataframe=financial_data)
