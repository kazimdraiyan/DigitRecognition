# Interface for drawing a digit and converting it to the input layer

from tkinter import *
from PIL import Image, ImageDraw
import numpy as np

from network.model import NeuralNetwork

WIDTH = 280
HEIGHT = 280

root = Tk()
canvas = Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
canvas.pack()  # Show the canvas in a window

image = Image.new("L", (WIDTH, HEIGHT))  # "L" means grayscale image
draw = ImageDraw.Draw(image)


def paint(event):
    x, y = event.x, event.y
    canvas.create_oval(
        x - 8, y - 8, x + 8, y + 8, fill="white", outline="white"
    )  # White dot with 16px diameter
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=255)


# Call paint while dragging pressing the left button
canvas.bind("<B1-Motion>", paint)

# Codes below this statement will run after closing the window
root.mainloop()


# Convert image into input layer
img = image.resize((28, 28))
x = np.array(img, dtype=np.float32)
x /= 255  # Normalize [0, 255] to [0, 1]
x = x.reshape(1, 28 * 28)  # Convert 2D array to 1D array

nn = NeuralNetwork()
prediction = nn.forward(x)
digit = np.argmax(prediction)
print(prediction)
print(digit)
