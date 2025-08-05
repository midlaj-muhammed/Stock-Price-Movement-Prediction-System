#!/usr/bin/env python3
"""
Interactive stock selector with 50+ top stocks for user convenience.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def display_stock_menu():
    """Display interactive stock selection menu."""
    
    from src.data.stock_symbols import (
        TOP_STOCKS, POPULAR_SYMBOLS, VOLATILE_SYMBOLS, 
        STABLE_SYMBOLS, get_stock_info, get_category
    )
    
    print("🚀 Stock Price Prediction - Stock Selector")
    print("=" * 60)
    
    while True:
        print("\n📊 SELECT STOCK SYMBOL")
        print("=" * 40)
        print("1. 🌟 Popular Stocks (Top 20)")
        print("2. 📂 Browse by Category")
        print("3. ⚡ High Volatility Stocks")
        print("4. 🛡️ Stable/Dividend Stocks")
        print("5. 🔤 Enter Custom Symbol")
        print("6. 📋 View All Stocks")
        print("7. ❌ Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            # Popular stocks
            print("\n🌟 POPULAR STOCKS")
            print("-" * 40)
            
            for i, symbol in enumerate(POPULAR_SYMBOLS, 1):
                name = get_stock_info(symbol)
                category = get_category(symbol)
                print(f"{i:2}. {symbol:6} - {name[:40]}... ({category})")
            
            try:
                selection = int(input(f"\nSelect stock (1-{len(POPULAR_SYMBOLS)}): ")) - 1
                if 0 <= selection < len(POPULAR_SYMBOLS):
                    return POPULAR_SYMBOLS[selection]
                else:
                    print("❌ Invalid selection!")
            except ValueError:
                print("❌ Please enter a valid number!")
                
        elif choice == "2":
            # Browse by category
            print("\n📂 BROWSE BY CATEGORY")
            print("-" * 40)
            
            categories = list(TOP_STOCKS.keys())
            for i, category in enumerate(categories, 1):
                count = len(TOP_STOCKS[category])
                print(f"{i}. {category} ({count} stocks)")
            
            try:
                cat_choice = int(input(f"\nSelect category (1-{len(categories)}): ")) - 1
                if 0 <= cat_choice < len(categories):
                    selected_category = categories[cat_choice]
                    
                    print(f"\n🏢 {selected_category.upper()} STOCKS")
                    print("-" * 40)
                    
                    stocks = list(TOP_STOCKS[selected_category].items())
                    for i, (symbol, name) in enumerate(stocks, 1):
                        print(f"{i:2}. {symbol:6} - {name}")
                    
                    stock_choice = int(input(f"\nSelect stock (1-{len(stocks)}): ")) - 1
                    if 0 <= stock_choice < len(stocks):
                        return stocks[stock_choice][0]
                    else:
                        print("❌ Invalid selection!")
                else:
                    print("❌ Invalid category!")
            except ValueError:
                print("❌ Please enter a valid number!")
                
        elif choice == "3":
            # Volatile stocks
            print("\n⚡ HIGH VOLATILITY STOCKS")
            print("-" * 40)
            
            for i, symbol in enumerate(VOLATILE_SYMBOLS, 1):
                name = get_stock_info(symbol)
                print(f"{i:2}. {symbol:6} - {name}")
            
            try:
                selection = int(input(f"\nSelect stock (1-{len(VOLATILE_SYMBOLS)}): ")) - 1
                if 0 <= selection < len(VOLATILE_SYMBOLS):
                    return VOLATILE_SYMBOLS[selection]
                else:
                    print("❌ Invalid selection!")
            except ValueError:
                print("❌ Please enter a valid number!")
                
        elif choice == "4":
            # Stable stocks
            print("\n🛡️ STABLE/DIVIDEND STOCKS")
            print("-" * 40)
            
            for i, symbol in enumerate(STABLE_SYMBOLS, 1):
                name = get_stock_info(symbol)
                print(f"{i:2}. {symbol:6} - {name}")
            
            try:
                selection = int(input(f"\nSelect stock (1-{len(STABLE_SYMBOLS)}): ")) - 1
                if 0 <= selection < len(STABLE_SYMBOLS):
                    return STABLE_SYMBOLS[selection]
                else:
                    print("❌ Invalid selection!")
            except ValueError:
                print("❌ Please enter a valid number!")
                
        elif choice == "5":
            # Custom symbol
            symbol = input("\n🔤 Enter stock symbol: ").upper().strip()
            if symbol:
                return symbol
            else:
                print("❌ Please enter a valid symbol!")
                
        elif choice == "6":
            # View all stocks
            print("\n📋 ALL AVAILABLE STOCKS")
            print("=" * 60)
            
            for category, stocks in TOP_STOCKS.items():
                print(f"\n🏢 {category.upper()}")
                print("-" * 40)
                for symbol, name in stocks.items():
                    print(f"  {symbol:6} - {name}")
            
            input("\nPress Enter to continue...")
            
        elif choice == "7":
            print("👋 Goodbye!")
            return None
            
        else:
            print("❌ Invalid choice! Please select 1-7.")

def quick_select_popular():
    """Quick selection from popular stocks."""
    
    from src.data.stock_symbols import POPULAR_SYMBOLS, get_stock_info
    
    print("🌟 QUICK SELECT - POPULAR STOCKS")
    print("=" * 40)
    
    # Show top 10 popular stocks
    for i, symbol in enumerate(POPULAR_SYMBOLS[:10], 1):
        name = get_stock_info(symbol)
        print(f"{i:2}. {symbol}")
    
    try:
        choice = int(input(f"\nQuick select (1-10) or 0 for full menu: "))
        if 1 <= choice <= 10:
            return POPULAR_SYMBOLS[choice - 1]
        elif choice == 0:
            return display_stock_menu()
        else:
            print("❌ Invalid choice!")
            return "AAPL"  # Default
    except ValueError:
        print("❌ Invalid input!")
        return "AAPL"  # Default

def main():
    """Main function for stock selection."""
    
    print("🚀 Stock Price Prediction System")
    print("=" * 50)
    
    # Quick start option
    quick_start = input("🚀 Quick start with popular stocks? (y/n) [y]: ").lower().strip()
    
    if quick_start in ['', 'y', 'yes']:
        symbol = quick_select_popular()
    else:
        symbol = display_stock_menu()
    
    if symbol:
        from src.data.stock_symbols import get_stock_info, get_category
        
        print(f"\n✅ Selected Stock: {symbol}")
        
        try:
            company_name = get_stock_info(symbol)
            category = get_category(symbol)
            print(f"🏢 Company: {company_name}")
            print(f"📂 Category: {category}")
        except:
            print(f"🏢 Company: Custom symbol")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Train model: python fine_tune_simple.py")
        print(f"   2. Web interface: CUDA_VISIBLE_DEVICES=-1 streamlit run src/web/app.py")
        print(f"   3. Command line: python src/main.py --symbol {symbol} --mode train")
        
        # Option to start training immediately
        start_training = input(f"\n🚀 Start training for {symbol} now? (y/n) [n]: ").lower().strip()
        
        if start_training in ['y', 'yes']:
            print(f"\n🧠 Starting training for {symbol}...")
            
            # Force CPU usage
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            try:
                # Import and run training
                from src.training_pipeline import StockPredictionPipeline
                
                pipeline = StockPredictionPipeline(symbol, "classification")
                
                print("📊 Collecting data...")
                pipeline.collect_data(period="2y", source="yahoo")
                
                print("🔧 Engineering features...")
                pipeline.engineer_features()
                
                print("📋 Preparing data...")
                pipeline.prepare_data_for_training(n_features=30)
                
                print("🧠 Training model...")
                pipeline.train_model("lstm", epochs=20, batch_size=32)
                
                print("📊 Evaluating...")
                results = pipeline.evaluate_model("lstm")
                
                print(f"\n🎉 Training completed!")
                print(f"📈 Accuracy: {results['accuracy']:.4f}")
                print(f"📊 F1 Score: {results['f1_score']:.4f}")
                
            except Exception as e:
                print(f"❌ Training failed: {e}")
                print(f"💡 Try the web interface instead: streamlit run src/web/app.py")
        
        return symbol
    else:
        print("❌ No stock selected.")
        return None

if __name__ == "__main__":
    main()
