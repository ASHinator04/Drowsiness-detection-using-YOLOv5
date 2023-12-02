import tkinter as tk
import customtkinter as ctk 
import dill 
import numpy as np
import cv2
import torch
import random
from PIL import Image, ImageTk
import vlc

app = tk.Tk()
app.geometry("600x600")
app.title("Drowsiness Detection")
ctk.set_appearance_mode('dark')

vidFrame = tk.Frame(height=480, width=600)
vidFrame.pack()
vid = ctk.CTkLabel(vidFrame)
vid.pack()

counter = 0
counterLabel = ctk.CTkLabel(text=counter, height=40, width=120, master=app, text_color="black")
counterLabel.pack(pady = 10)

def reset_counter():
    global counter
    counter = 0
resetButton = ctk.CTkButton(text="Reset counter", command=reset_counter, height=40, width=120, master=app)
resetButton.pack()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True )
cap = cv2.VideoCapture(0)
def detect():
    global counter
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, 
                        cv2.COLOR_BGR2RGB)
    result = model(frame)
    img = np.squeeze(result.render())

    
    if len(result.xywh[0])> 0:
        dconf = result.xywh[0][0][4]
        dclass = result.xywh[0][0][5]

        if dconf.item() > 0.50  and dclass.item() == 16 :
            filechoice  = random.choice([1,2,3])
            counter += 1
            p = vlc.MediaPlayer(f"file:///{filechoice}.mp3")
            p.play()
        
        if (dconf.item() < 0.10  and dclass.item() == 15):  
            filechoice  = random.choice([1,2,3])
            counter += 1
            p = vlc.MediaPlayer(f"file:///{filechoice}.mp3")
            p.play()          

    imgarr = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(imgarr)
    vid.imgtk = imgtk
    vid.configure(image = imgtk)
    vid.after(10,detect)
#    counter.configure(text = counter)
detect()

app.mainloop()
