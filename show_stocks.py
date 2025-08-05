#!/usr/bin/env python3
"""
Display all available stock symbols for the prediction system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Display all stock symbols organized by category."""
    
    try:
        from src.data.stock_symbols import (
            TOP_STOCKS, ALL_SYMBOLS, POPULAR_SYMBOLS, 
            VOLATILE_SYMBOLS, STABLE_SYMBOLS,
            get_stock_info, get_category
        )
        
        print("🚀 Stock Price Prediction System")
        print("📊 Available Stock Symbols")
        print("=" * 80)
        
        # Summary
        print(f"📈 Total Stocks Available: {len(ALL_SYMBOLS)}")
        print(f"🏢 Categories: {len(TOP_STOCKS)}")
        print(f"🌟 Popular Stocks: {len(POPULAR_SYMBOLS)}")
        print(f"⚡ High Volatility: {len(VOLATILE_SYMBOLS)}")
        print(f"🛡️ Stable/Dividend: {len(STABLE_SYMBOLS)}")
        
        # Popular stocks
        print(f"\n🌟 MOST POPULAR STOCKS (Top 20)")
        print("=" * 60)
        for i, symbol in enumerate(POPULAR_SYMBOLS[:20], 1):
            name = get_stock_info(symbol)
            category = get_category(symbol)
            print(f"{i:2}. {symbol:6} - {name[:45]}... ({category})")
        
        # All stocks by category
        print(f"\n📂 ALL STOCKS BY CATEGORY")
        print("=" * 80)
        
        for category, stocks in TOP_STOCKS.items():
            print(f"\n🏢 {category.upper()} ({len(stocks)} stocks)")
            print("-" * 60)
            
            # Sort stocks alphabetically
            sorted_stocks = sorted(stocks.items())
            
            for symbol, name in sorted_stocks:
                # Add indicators for special categories
                indicators = []
                if symbol in POPULAR_SYMBOLS:
                    indicators.append("🌟")
                if symbol in VOLATILE_SYMBOLS:
                    indicators.append("⚡")
                if symbol in STABLE_SYMBOLS:
                    indicators.append("🛡️")
                
                indicator_str = "".join(indicators) + " " if indicators else "   "
                print(f"  {indicator_str}{symbol:6} - {name}")
        
        # Special categories
        print(f"\n⚡ HIGH VOLATILITY STOCKS")
        print("-" * 40)
        for symbol in VOLATILE_SYMBOLS:
            name = get_stock_info(symbol)
            print(f"  {symbol:6} - {name}")
        
        print(f"\n🛡️ STABLE/DIVIDEND STOCKS")
        print("-" * 40)
        for symbol in STABLE_SYMBOLS:
            name = get_stock_info(symbol)
            print(f"  {symbol:6} - {name}")
        
        # Usage examples
        print(f"\n💡 USAGE EXAMPLES")
        print("=" * 50)
        print(f"1. 🌐 Web Interface:")
        print(f"   CUDA_VISIBLE_DEVICES=-1 streamlit run src/web/app.py")
        print(f"   (Use the enhanced stock selector in the sidebar)")
        
        print(f"\n2. 🎯 Fine-tuning:")
        print(f"   python fine_tune_simple.py")
        print(f"   (Interactive stock selection with popular stocks)")
        
        print(f"\n3. 🔤 Interactive Selector:")
        print(f"   python stock_selector.py")
        print(f"   (Full interactive menu with all categories)")
        
        print(f"\n4. 📊 Command Line:")
        print(f"   python src/main.py --symbol AAPL --mode train")
        print(f"   (Replace AAPL with any symbol from the list above)")
        
        print(f"\n5. 🎲 Random Testing:")
        print(f"   Try these popular combinations:")
        print(f"   - Tech: AAPL, MSFT, GOOGL, TSLA, NVDA")
        print(f"   - Finance: JPM, BAC, V, MA, BRK.B")
        print(f"   - Healthcare: JNJ, UNH, PFE, ABBV")
        print(f"   - Consumer: WMT, PG, KO, HD, MCD")
        
        print(f"\n🎯 LEGEND")
        print("-" * 20)
        print(f"🌟 Popular Stock (Most traded)")
        print(f"⚡ High Volatility (Active trading)")
        print(f"🛡️ Stable/Dividend (Conservative)")
        
        print(f"\n✨ All symbols are ready for prediction!")
        print(f"Choose any symbol and start training your model! 🚀")
        
    except ImportError as e:
        print(f"❌ Error importing stock symbols: {e}")
        print(f"💡 Make sure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
