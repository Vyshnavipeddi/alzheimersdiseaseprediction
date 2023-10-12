from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import keras 
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers

main = tkinter.Tk()
main.title("Alzeimers prediction")
main.geometry("1300x1200")

mapping = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def load():
    global model
    model = load_model('model_final.h5')
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

def upload():
    text.delete('1.0',END)
    global filename
    filename = askopenfilename()
    text.insert(END,"File Uploaded: "+str(filename)+"\n")

def imagepreprocess():
    global img4
    img3 = cv2.imread(filename)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img3 = cv2.resize(img3,(224,224))
    img4 = np.reshape(img3,[1,224,224,3])

def predict():
    disease = model.predict_classes(img4)
    prediction = disease[0]
    prediction_name = mapping[prediction]
    text.insert(END,"Predicted output for uploaded Image: "+str(prediction_name)+"\n")

font = ('times', 16, 'bold')
title = Label(main, text='Alzeimers Prediction From MRI Images')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')

upload = Button(main, text="Upload Image", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)

process = Button(main, text="Image Pre-Processing", command=imagepreprocess)
process.place(x=700,y=150)
process.config(font=font1)

ld = Button(main, text="Model Load", command=load)
ld.place(x=700,y=200)
ld.config(font=font1)

pred = Button(main, text="Prediction", command=predict)
pred.place(x=700,y=250)
pred.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='tan1')
main.mainloop()
