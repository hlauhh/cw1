# README: FCFF Calculations and Analysis

## Introduction

This script provides a comprehensive approach to calculating and analyzing Free Cash Flow to the Firm (FCFF) for publicly traded companies using data from Alpha Vantage and FRED APIs. It covers essential aspects of FCFF-based financial analysis, including the collection of financial data, projections, discounting cash flows, and valuation of a company's equity.

The main objective is to calculate the implied share price of a company based on its future cash flows, applying the Discounted Cash Flow (DCF) methodology, and to visualize the results with historical and projected FCFF data.

## Key Features

1. **Financial Data Fetching**: 
   - The script pulls financial statement data (Income Statement, Balance Sheet, and Cash Flow Statement) using Alpha Vantage API. It also fetches macroeconomic indicators, such as the 10-Year Treasury Yield, from FRED API.
   - CSV backups are saved to a local directory, allowing the script to load data from disk if the API request limit is exceeded.

2. **FCFF Calculation**: 
   - FCFF (Free Cash Flow to the Firm) is a key metric in determining the cash flows available to the company after accounting for operating expenses, capital expenditures, and interest expenses (adjusted for taxes). This script calculates FCFF from the company's cash flow statement data and interest expenses.

3. **Tax Rate Estimation**:
   - The script calculates the effective tax rate from the Income Statement data by dividing income tax expense by income before tax.

4. **Projection and Discounting**:
   - The script projects future FCFF for a user-defined number of years, using an average growth rate. It discounts future FCFF to present value using the Weighted Average Cost of Capital (WACC).

5. **WACC and Terminal Value**:
   - The Weighted Average Cost of Capital (WACC) is computed using the company’s cost of equity (based on its Beta) and cost of debt, as well as its capital structure.
   - The Terminal Value represents the company's value beyond the projection period, calculated using the final year's projected FCFF, WACC, and a terminal growth rate (typically aligned with long-term GDP growth).

6. **Equity Valuation**:
   - The script calculates the total enterprise value by summing the present value of all future FCFF and the terminal value. Subtracting total debt and adding cash equivalents gives the equity value, which is then divided by the number of shares outstanding to derive the implied share price.

7. **Visualization**:
   - The script generates a detailed plot showing historical and projected FCFF data, trend lines (linear, polynomial, and exponential regressions), and selected financial statement items (Revenue, Net Income, Assets, Liabilities). This visual helps investors understand the company’s financial trajectory.

## FCFF Calculation Details

### Formula:
FCFF = 	ext{Operating Cash Flow} - 	ext{Capital Expenditures} + (	ext{Interest Expense} 	imes (1 - 	ext{Tax Rate}))

- **Operating Cash Flow (OCF)**: This represents the cash generated from the company’s core business operations.
- **Capital Expenditures (CapEx)**: These are investments in property, plant, and equipment, required to maintain and expand the company's operational capacity.
- **Interest Expense**: Represents the cost of the company’s debt. It is adjusted for the tax shield since interest payments are tax-deductible.

### Steps:
1. **Fetch Financial Data**: The script fetches the cash flow statement to get OCF and CapEx and the income statement to obtain the interest expense and tax rate.
2. **Calculate FCFF**: Using the formula above, the script computes the historical FCFF for the last five fiscal years.
3. **Project Future FCFF**: Based on the most recent FCFF and an estimated growth rate, the script projects future FCFF for the desired period (e.g., 5 years).
4. **Discount to Present Value**: The projected FCFF values are discounted to present value using the WACC, which takes into account both the cost of equity and cost of debt.
5. **Calculate Terminal Value**: The terminal value is calculated at the end of the projection period, assuming the company continues to grow at a stable, long-term growth rate.

## Usage

To run the script, execute the `main()` function. You will be prompted to enter a stock ticker symbol and the number of years for FCFF analysis. If no input is provided within the specified timeout (5 seconds), the script defaults to analyzing Amazon (AMZN) with a 5-year projection.

The script fetches data, calculates FCFF, projects future cash flows, and plots the results, which are displayed in your browser.
