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
probability_texts = [StringVar(value=f"{i}: 0.0 %") for i in range(10)]
for i, string_var in enumerate(probability_texts):
    Label(right_frame, textvariable=string_var, anchor="w", width=14).pack()


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
            probability_texts[i].set(f"{i}: 0.0 %")
    except NameError:
        pass


def predict():
    img = image.resize((28, 28))
    x = np.array(img, dtype=np.float32) / 255.0
    x = x.reshape(1, 28 * 28)

    nn = NeuralNetwork(filename="model_v01.npz")
    prediction = nn.forward(x).squeeze() * 100
    sorted_indices = np.argsort(-prediction) # Sort indices in descending order

    # Update UI labels with percentages
    for i in range(10):
        probability_texts[i].set(f"{sorted_indices[i]}: {prediction[sorted_indices[i]]:.2f} %")


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
