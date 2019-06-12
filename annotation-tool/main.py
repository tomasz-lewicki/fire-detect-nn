import tkinter as tk
from PIL import ImageTk, Image
import os 
import shutil

HEIGHT = 1440
WIDTH = 2560
image_no = 0

def fire(event):
    #global current_image_path
    shutil.move(current_image_path, 'positives/'+current_image_path[4:])
    grab_new_image()

def nonfire(event):
    global current_image_path
    shutil.move(current_image_path, 'negatives/'+current_image_path[4:])
    grab_new_image()

def delete(event):
    global current_image_path
    os.remove(current_image_path)
    grab_new_image()

def grab_new_image():
    global image_no, current_image_path
    image_no+=1
    current_image_path = image_paths[image_no]
    img2 = ImageTk.PhotoImage(Image.open(current_image_path))
    panel.configure(image = img2)
    panel.image = img2

window = tk.Tk()
window.title("annotator")
window.geometry("1440x2560")
window.configure(background='grey')

image_paths = ['all/'+filename for filename in os.listdir('all')]
current_image_path = image_paths[0]
img = ImageTk.PhotoImage(Image.open(current_image_path))
panel = tk.Label(window, image = img)
panel.configure(image=img)
panel.pack(side = "bottom", fill = "both", expand = "yes")

window.bind_all('<KeyPress-1>', fire)
window.bind_all('<KeyPress-0>', nonfire)
window.bind_all('<KeyPress-d>', delete)

window.mainloop()

#GOAL: working tool to move files between folders with 3 key bindings 