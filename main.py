import yfinance as yf
import numpy as np

# Define constants
ticker = 'AMZN'
tax_rate = 0.178  # Given tax rate
discount_rate = 0.03  # Assumed discount rate (9%)
perpetual_growth_rate = 0.02  # Assumed perpetual growth rate (2.5%)
years = 5  # Projection for 5 years

# Download the financial data for Amazon
amazon = yf.Ticker(ticker)
financials = amazon.financials
cash_flow = amazon.cashflow
balance_sheet = amazon.balance_sheet
info = amazon.info

# Extract financial data from the correct labels
revenue = financials.loc['Total Revenue'].iloc[0]
operating_income = financials.loc['Operating Income'].iloc[0]

# Correct key for Capital Expenditure and Depreciation
capex = -cash_flow.loc['Capital Expenditure'].iloc[0]
depreciation = cash_flow.loc['Depreciation Amortization Depletion'].iloc[0]

# Change in Working Capital
change_in_working_capital = cash_flow.loc['Change In Working Capital'].iloc[0]

# Estimate Free Cash Flow (FCF)
fcf = operating_income * (1 - tax_rate) + depreciation + capex - change_in_working_capital

# Project FCF for 5 years with assumed growth rate (5%)
fcf_growth_rate = 0.05  # Assume 5% growth in FCF
fcf_projections = [(fcf * (1 + fcf_growth_rate) ** i) for i in range(1, years + 1)]

# Calculate Terminal Value (TV) at year 5
terminal_value = (fcf_projections[-1] * (1 + perpetual_growth_rate)) / (discount_rate - perpetual_growth_rate)

# Discount FCF and Terminal Value to Present Value (PV)
discounted_fcf = [fcf / (1 + discount_rate) ** i for i, fcf in enumerate(fcf_projections, 1)]
discounted_terminal_value = terminal_value / (1 + discount_rate) ** years

# Sum of discounted FCFs and Terminal Value gives the Enterprise Value (EV)
enterprise_value = sum(discounted_fcf) + discounted_terminal_value

# Additional data: Cash, Debt, and Shares Outstanding
cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]  # Corrected key for cash
debt = balance_sheet.loc['Long Term Debt'].iloc[0]  # Corrected key for long-term debt
shares_outstanding = info['sharesOutstanding']

# Calculate Equity Value
equity_value = enterprise_value + cash - debt

# Calculate Implied Share Price
implied_share_price = equity_value / shares_outstanding

# Output the results
print(f"Projected Free Cash Flows (Years 1-5): {fcf_projections}")
print(f"Terminal Value (TV): {terminal_value}")
print(f"Discounted FCFs: {discounted_fcf}")
print(f"Discounted Terminal Value: {discounted_terminal_value}")
print(f"Enterprise Value (EV): {enterprise_value}")
print(f"Equity Value: {equity_value}")
print(f"Implied Share Price: ${implied_share_price:.2f}")
