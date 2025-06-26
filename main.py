import tkinter as tk
from stock_gui import StockPredictorGUI


def main():
    """Main entry point for the Stock Price Predictor application"""
    # Create the main window
    root = tk.Tk()
    
    # Create the GUI application
    app = StockPredictorGUI(root)
    
    # Start the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    main() 