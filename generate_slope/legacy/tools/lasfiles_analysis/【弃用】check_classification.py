import laspy
import numpy as np
import tkinter as tk
from tkinter import filedialog

def check():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("LAS files", "*.las")])
    
    if file_path:
        las = laspy.read(file_path)
        unique_classes = np.unique(las.classification)
        print(f"Classification values: {unique_classes.tolist()}")

if __name__ == "__main__":
    check()