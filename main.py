# Interface for drawing a digit and converting it to the input layer

from tkinter import * # type: ignore
from PIL import Image, ImageDraw
import numpy as np

from network.model import NeuralNetwork

WIDTH = 280
HEIGHT = 280

root = Tk()

main_frame = Frame(root)
main_frame.pack(padx=8, pady=8)

# Canvas for drawing
left_frame = Frame(main_frame)
left_frame.pack(side=LEFT)
canvas = Canvas(left_frame, width=WIDTH, height=HEIGHT, bg="black")
canvas.pack()

# Output labels
right_frame = Frame(main_frame)
right_frame.pack(side=LEFT, padx=(12, 0))

# Model 1 column
model1_frame = Frame(right_frame)
model1_frame.pack(side=LEFT, padx=(0, 10))
Label(model1_frame, text="Model 1", font=("Arial", 10, "bold")).pack()
probability_texts_m1 = [StringVar(value=f"{i}: 0.0 %") for i in range(10)]
for i, string_var in enumerate(probability_texts_m1):
    Label(model1_frame, textvariable=string_var, anchor="w", width=14).pack()

# Model 2 column
model2_frame = Frame(right_frame)
model2_frame.pack(side=LEFT)
Label(model2_frame, text="Model 2", font=("Arial", 10, "bold")).pack()
probability_texts_m2 = [StringVar(value=f"{i}: 0.0 %") for i in range(10)]
for i, string_var in enumerate(probability_texts_m2):
    Label(model2_frame, textvariable=string_var, anchor="w", width=14).pack()


image = Image.new("L", (WIDTH, HEIGHT))  # "L" means grayscale image
draw = ImageDraw.Draw(image)


def paint(event):
    x, y = event.x, event.y
    canvas.create_oval(
        x - 8, y - 8, x + 8, y + 8, fill="white", outline="white"
    )  # White dot with 16px diameter
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=255)


def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=0)
    # Reset displayed probabilities
    try:
        for i in range(10):
            probability_texts_m1[i].set(f"{i}: 0.0 %")
            probability_texts_m2[i].set(f"{i}: 0.0 %")
    except NameError:
        pass


def predict():
    img = image.resize((28, 28))
    x = np.array(img, dtype=np.float32) / 255.0
    x = x.reshape(1, 28 * 28)

    nn1 = NeuralNetwork(filename="model_v01.npz")
    nn2 = NeuralNetwork(filename="model_v02.npz")
    
    # Get predictions from both models
    prediction1 = nn1.forward(x).squeeze() * 100
    prediction2 = nn2.forward(x).squeeze() * 100
    
    # Sort indices for model 1
    sorted_indices1 = np.argsort(-prediction1)
    # Sort indices for model 2
    sorted_indices2 = np.argsort(-prediction2)

    # Update UI labels for model 1
    for i in range(10):
        probability_texts_m1[i].set(f"{sorted_indices1[i]}: {prediction1[sorted_indices1[i]]:.2f} %")
    
    # Update UI labels for model 2
    for i in range(10):
        probability_texts_m2[i].set(f"{sorted_indices2[i]}: {prediction2[sorted_indices2[i]]:.2f} %")


# Call paint while dragging pressing the left button
canvas.bind("<B1-Motion>", paint)

# Buttons
button_frame = Frame(left_frame)
button_frame.pack(pady=10)

clear_btn = Button(button_frame, text="Clear", command=clear_canvas, width=10)
clear_btn.pack(side=LEFT, padx=5)

done_btn = Button(button_frame, text="Predict", command=predict, width=10)
done_btn.pack(side=LEFT, padx=5)

# Codes below this statement will run after closing the window
root.mainloop()
