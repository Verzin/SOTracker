from PIL import Image, ImageTk
import cv2
import tkinter as tk

from tkinter import messagebox

import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker

basic = [True]
trajectory = [False]
all_info = [False]
direction =[False]
TypeInformation = 'basic'

window = tk.Tk()
window.title('SOTracker')
window.geometry('450x300')

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# welcome image
canvas = tk.Canvas(window, height=200, width=500)
image_file = tk.PhotoImage(file='logo.png')
image = canvas.create_image(0,0, anchor='nw', image=image_file)
canvas.pack(side='top')

# input path & output path
tk.Label(window, text='Input path: ').place(x=117, y= 150)
tk.Label(window, text='Output path: ').place(x=110, y= 170)
tk.Label(window, text='Length of Object: ').place(x=75, y= 190)
tk.Label(window, text='Information to observe: ').place(x=33, y=210)

var_Input_path = tk.StringVar()
var_Input_path.set('example:/data2/Verzin/TestVideo/SOTracker/mice_in_box.mp4')
entry_Input_path = tk.Entry(window, textvariable=var_Input_path)
entry_Input_path.place(x=200, y=150)

var_Output_path = tk.StringVar()
var_Output_path.set('example:/data2/Verzin/d3s/pytracking/the_output/')
entry_Output_path = tk.Entry(window, textvariable=var_Output_path)
entry_Output_path.place(x=200, y=170)
var_length = tk.DoubleVar()
var_length.set(5.1111111111)

entry_length = tk.Entry(window, textvariable=var_length)
entry_length.place(x=200, y=190)
tk.Label(window, text='cm').place(x=350, y= 190)

tk.Radiobutton(window,text = 'base',variable=TypeInformation,value='basic').place(x=200, y=210)
tk.Radiobutton(window,text = 'All',variable=TypeInformation,value='all').place(x=300, y=210)
tk.Radiobutton(window,text = 'trajectory',variable=TypeInformation,value='trajectory').place(x=200, y=230)
tk.Radiobutton(window,text = 'direction',variable=TypeInformation,value='direction').place(x=300, y=230)


# default tracker and parama
default_tracker = 'segm'
default_params = 'default_params'

def run_tracker():

    global entry_Input_path
    global entry_Output_path
    global default_tracker
    global default_params
    global screen_width
    global screen_height
    tracker = Tracker(default_tracker, default_params)
    #if os.path.isfile(str(entry_Input_path)) and os.path.exists(str(entry_Output_path)):
    tracker.run_video(Inputvideofilepath = str(entry_Input_path.get()),Outputvideofilepath = str(entry_Output_path.get()),optional_box=None, debug=None,ObjectLength = float(var_length.get()),TypeVideo = TypeInformation)
    #else:
        #tk.messagebox.showerror(title = 'Warning path',message = 'Please check Input Path and Output path!')
        #return None

def train_tracker():
    pass

   
# run and train button
#if isinstance(var_Input_path,str) and isinstance(var_Output_path,str) is True:
btn_run = tk.Button(window, text='Run the tracker', command=run_tracker)
btn_run.place(x=75, y=250)
btn_train = tk.Button(window, text='Train the tracker', command=train_tracker)
btn_train.place(x=270, y=250)

window.mainloop()




