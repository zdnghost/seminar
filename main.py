# main.py
import tkinter as tk
from database import init_db
from ui import create_ui

if __name__ == "__main__":
    init_db()

    root = tk.Tk()
    create_ui(root)
    root.mainloop()
