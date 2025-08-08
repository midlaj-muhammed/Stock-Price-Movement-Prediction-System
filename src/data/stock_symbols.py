"""
Top 50+ stock symbols for user convenience.
"""

# Top 50+ Stock Symbols organized by category
TOP_STOCKS = {
    "Technology": {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc. (Class A)",
        "GOOG": "Alphabet Inc. (Class C)",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "META": "Meta Platforms Inc.",
        "NVDA": "NVIDIA Corporation",
        "NFLX": "Netflix Inc.",
        "ADBE": "Adobe Inc.",
        "CRM": "Salesforce Inc.",
        "ORCL": "Oracle Corporation",
        "INTC": "Intel Corporation",
        "AMD": "Advanced Micro Devices",
        "PYPL": "PayPal Holdings Inc.",
        "UBER": "Uber Technologies Inc.",
        "SPOT": "Spotify Technology S.A.",
        "ZOOM": "Zoom Video Communications",
        "SHOP": "Shopify Inc.",
        "SQ": "Block Inc. (Square)",
        "TWTR": "Twitter Inc.",
        "SNAP": "Snap Inc.",
        "ROKU": "Roku Inc.",
        "DOCU": "DocuSign Inc.",
        "ZM": "Zoom Video Communications"
    },
    
    "Finance": {
        "JPM": "JPMorgan Chase & Co.",
        "BAC": "Bank of America Corp.",
        "WFC": "Wells Fargo & Company",
        "GS": "Goldman Sachs Group Inc.",
        "MS": "Morgan Stanley",
        "C": "Citigroup Inc.",
        "AXP": "American Express Company",
        "V": "Visa Inc.",
        "MA": "Mastercard Inc.",
        "BRK.A": "Berkshire Hathaway Inc. (Class A)",
        "BRK.B": "Berkshire Hathaway Inc. (Class B)",
        "USB": "U.S. Bancorp",
        "PNC": "PNC Financial Services",
        "TFC": "Truist Financial Corp.",
        "COF": "Capital One Financial Corp."
    },
    
    "Healthcare": {
        "JNJ": "Johnson & Johnson",
        "UNH": "UnitedHealth Group Inc.",
        "PFE": "Pfizer Inc.",
        "ABBV": "AbbVie Inc.",
        "TMO": "Thermo Fisher Scientific",
        "ABT": "Abbott Laboratories",
        "LLY": "Eli Lilly and Company",
        "MRK": "Merck & Co. Inc.",
        "BMY": "Bristol Myers Squibb",
        "AMGN": "Amgen Inc.",
        "GILD": "Gilead Sciences Inc.",
        "CVS": "CVS Health Corporation",
        "ANTM": "Anthem Inc.",
        "CI": "Cigna Corporation",
        "HUM": "Humana Inc."
    },
    
    "Consumer": {
        "WMT": "Walmart Inc.",
        "PG": "Procter & Gamble Co.",
        "KO": "The Coca-Cola Company",
        "PEP": "PepsiCo Inc.",
        "COST": "Costco Wholesale Corp.",
        "HD": "The Home Depot Inc.",
        "MCD": "McDonald's Corporation",
        "NKE": "Nike Inc.",
        "SBUX": "Starbucks Corporation",
        "TGT": "Target Corporation",
        "LOW": "Lowe's Companies Inc.",
        "DIS": "The Walt Disney Company",
        "VZ": "Verizon Communications",
        "T": "AT&T Inc.",
        "CMCSA": "Comcast Corporation"
    },
    
    "Energy": {
        "XOM": "Exxon Mobil Corporation",
        "CVX": "Chevron Corporation",
        "COP": "ConocoPhillips",
        "EOG": "EOG Resources Inc.",
        "SLB": "Schlumberger Limited",
        "PSX": "Phillips 66",
        "VLO": "Valero Energy Corporation",
        "MPC": "Marathon Petroleum Corp.",
        "KMI": "Kinder Morgan Inc.",
        "OKE": "ONEOK Inc."
    },
    
    "Industrial": {
        "BA": "The Boeing Company",
        "CAT": "Caterpillar Inc.",
        "GE": "General Electric Company",
        "MMM": "3M Company",
        "HON": "Honeywell International",
        "UPS": "United Parcel Service",
        "FDX": "FedEx Corporation",
        "LMT": "Lockheed Martin Corp.",
        "RTX": "Raytheon Technologies",
        "DE": "Deere & Company"
    },
    
    "Real Estate": {
        "AMT": "American Tower Corporation",
        "PLD": "Prologis Inc.",
        "CCI": "Crown Castle International",
        "EQIX": "Equinix Inc.",
        "SPG": "Simon Property Group",
        "O": "Realty Income Corporation",
        "WELL": "Welltower Inc.",
        "DLR": "Digital Realty Trust",
        "PSA": "Public Storage",
        "EXR": "Extended Stay America"
    },

    "India (NSE/BSE)": {
        "RELIANCE.NS": "Reliance Industries Limited",
        "TCS.NS": "Tata Consultancy Services Limited",
        "INFY.NS": "Infosys Limited",
        "HDFCBANK.NS": "HDFC Bank Limited",
        "ICICIBANK.NS": "ICICI Bank Limited",
        "KOTAKBANK.NS": "Kotak Mahindra Bank Limited",
        "SBIN.NS": "State Bank of India",
        "HINDUNILVR.NS": "Hindustan Unilever Limited",
        "ITC.NS": "ITC Limited",
        "LT.NS": "Larsen & Toubro Limited",
        "ASIANPAINT.NS": "Asian Paints Limited",
        "MARUTI.NS": "Maruti Suzuki India Limited",
        "AXISBANK.NS": "Axis Bank Limited",
        "WIPRO.NS": "Wipro Limited",
        "TECHM.NS": "Tech Mahindra Limited",
        "ULTRACEMCO.NS": "UltraTech Cement Limited",
        "SUNPHARMA.NS": "Sun Pharmaceutical Industries Limited",
        "NESTLEIND.NS": "Nestlé India Limited",
        "TATAMOTORS.NS": "Tata Motors Limited",
        "TATASTEEL.NS": "Tata Steel Limited",
        "ADANIENT.NS": "Adani Enterprises Limited",
        "HDFCLIFE.NS": "HDFC Life Insurance Company Limited",
        "RELIANCE.BO": "Reliance Industries Limited (BSE)",
        "TCS.BO": "Tata Consultancy Services Limited (BSE)",
        "INFY.BO": "Infosys Limited (BSE)",
        "HDFCBANK.BO": "HDFC Bank Limited (BSE)"
    }
}

# Flat list of all symbols for easy access
ALL_SYMBOLS = []
for category, stocks in TOP_STOCKS.items():
    ALL_SYMBOLS.extend(list(stocks.keys()))

# Most popular symbols (top 20)
POPULAR_SYMBOLS = [
    # US
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "NFLX", "JPM", "JNJ",
    "UNH", "PG", "HD", "BAC", "DIS",
    "ADBE", "CRM", "V", "MA", "WMT",
    # India (NSE)
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"
]

# Volatile/High-Growth symbols for active trading
VOLATILE_SYMBOLS = [
    "TSLA", "NVDA", "AMD", "NFLX", "ZOOM",
    "ROKU", "SPOT", "SQ", "UBER", "SNAP",
    "SHOP", "DOCU", "ZM", "PYPL", "TWTR"
]

# Stable/Dividend symbols for conservative investing
STABLE_SYMBOLS = [
    "AAPL", "MSFT", "JNJ", "PG", "KO",
    "PEP", "WMT", "HD", "MCD", "VZ",
    "T", "JPM", "BAC", "XOM", "CVX"
]

def get_stock_info(symbol: str) -> str:
    """Get company name for a stock symbol."""
    for category, stocks in TOP_STOCKS.items():
        if symbol.upper() in stocks:
            return stocks[symbol.upper()]
    return f"Unknown company for {symbol}"

def get_category(symbol: str) -> str:
    """Get category for a stock symbol."""
    for category, stocks in TOP_STOCKS.items():
        if symbol.upper() in stocks:
            return category
    return "Unknown"

def search_stocks(query: str) -> list:
    """Search for stocks by symbol or company name.
    Includes support for Indian markets with .NS (NSE) and .BO (BSE) suffixes.
    """
    query = query.lower().strip()
    results = []

    # 1) Search internal database
    for category, stocks in TOP_STOCKS.items():
        for symbol, name in stocks.items():
            if (query in symbol.lower() or query in name.lower()):
                results.append({'symbol': symbol, 'name': name, 'category': category})

    # 2) If no results and query looks like an Indian symbol without suffix, try adding .NS/.BO
    def _add_if_valid(sym: str, label: str):
        if sym not in [r['symbol'] for r in results]:
            results.append({'symbol': sym, 'name': f"Custom ({label})", 'category': 'Unknown'})

    if not results:
        # Symbol-like queries without market suffix
        if query.isalpha() or ('.' in query and query.endswith(('.ns', '.bo'))):
            base = query.replace('.ns', '').replace('.bo', '').upper()
            if not query.endswith(('.ns', '.bo')):
                _add_if_valid(f"{base}.NS", "NSE candidate")
                _add_if_valid(f"{base}.BO", "BSE candidate")
            else:
                _add_if_valid(base.upper(), "Exact candidate")

    return results

def get_symbols_by_category(category: str) -> dict:
    """Get all symbols in a specific category."""
    return TOP_STOCKS.get(category, {})

def get_random_symbols(count: int = 10) -> list:
    """Get random stock symbols for testing."""
    import random
    return random.sample(ALL_SYMBOLS, min(count, len(ALL_SYMBOLS)))

# Display functions
def display_all_stocks():
    """Display all stocks organized by category."""
    print("📊 TOP 50+ STOCK SYMBOLS")
    print("=" * 60)
    
    for category, stocks in TOP_STOCKS.items():
        print(f"\n🏢 {category.upper()}")
        print("-" * 40)
        for symbol, name in stocks.items():
            print(f"  {symbol:6} - {name}")
    
    print(f"\n📈 Total: {len(ALL_SYMBOLS)} stocks across {len(TOP_STOCKS)} categories")

def display_popular_stocks():
    """Display most popular stocks."""
    print("🌟 MOST POPULAR STOCKS")
    print("=" * 40)
    
    for i, symbol in enumerate(POPULAR_SYMBOLS, 1):
        name = get_stock_info(symbol)
        category = get_category(symbol)
        print(f"{i:2}. {symbol:6} - {name} ({category})")

def display_category_menu():
    """Display category selection menu."""
    print("📂 SELECT CATEGORY")
    print("=" * 30)
    
    categories = list(TOP_STOCKS.keys())
    for i, category in enumerate(categories, 1):
        count = len(TOP_STOCKS[category])
        print(f"{i}. {category} ({count} stocks)")
    
    return categories

if __name__ == "__main__":
    # Demo the stock symbols
    display_all_stocks()
    print("\n")
    display_popular_stocks()
