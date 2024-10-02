import yfinance as yf

# List of ticker symbols
tickers = ["META", "AMZN", "MSFT", "AAPL", "GOOGL"]

# Fetch data
data = yf.download(tickers, period="1d")

# Extract closing prices
close_prices = data["Close"].iloc[-1]

# Print the closing prices
print("Closing Prices:")
for ticker, price in close_prices.items():
    print(f"{ticker}: {price:.2f}")