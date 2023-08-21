import sys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *

import python_gui_as2

_debug = True  # False to eliminate debug printing from callback functions.


def main(*args):
    '''Main entry point for the application.'''
    global root
    root = tk.Tk()
    root.protocol('WM_DELETE_WINDOW', root.destroy)
    # Creates a toplevel widget.
    global _top1, _w1
    _top1 = root
    _w1 = python_gui_as2.UserInterface(_top1)
    root.mainloop()


if __name__ == '__main__':
    python_gui_as2.start_up()
