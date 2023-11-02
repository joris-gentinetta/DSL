import numpy as np
import tifffile
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import HORIZONTAL, Checkbutton, IntVar
from tkinter import IntVar, Checkbutton, Label


# Load the TIFF file
img = tifffile.imread('/Users/jg/Desktop/DSL/4T1 timepoints FACS.HTD - Well A03 Field #1.tif')
channel_names = ['brightfield', 'green', 'red', 'far red']

# Check the shape of the image data
print(img.shape)  # Should be (timepoints, channels, height, width)

def scale_channel(channel, lower_percentile, upper_percentile):
    p_low = np.percentile(channel, lower_percentile)
    p_high = np.percentile(channel, upper_percentile)
    channel = (channel - p_low) / (p_high - p_low)
    channel = np.clip(channel, 0, 1)
    return channel

class PercentileSlider(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.lower = tk.Scale(self, from_=0, to=100, orient=HORIZONTAL)
        self.lower.grid(row=0, column=0, sticky="ew")
        self.lower.set(0)

        self.upper = tk.Scale(self, from_=0, to=100, orient=HORIZONTAL)
        self.upper.grid(row=1, column=0, sticky="ew")
        self.upper.set(100)

        self.lower.bind("<ButtonRelease-1>", self.sync_sliders)
        self.upper.bind("<ButtonRelease-1>", self.sync_sliders)

    def sync_sliders(self, event):
        lower_val = self.lower.get()
        upper_val = self.upper.get()
        if lower_val > upper_val:
            if event.widget == self.lower:
                self.upper.set(lower_val)
            else:
                self.lower.set(upper_val)

    def get_values(self):
        return self.lower.get(), self.upper.get()


class ChannelFrame(tk.Frame):
    def __init__(self, master, channel, **kwargs):
        super().__init__(master, **kwargs)
        self.channel = channel

        self.var = IntVar()
        self.var.set(1)  # default is selected

        self.title_label = Label(self, text=channel_names[channel])
        self.title_label.grid(row=0, column=1, sticky="ew")

        self.checkbutton = Checkbutton(self, variable=self.var)
        self.checkbutton.grid(row=0, column=0, sticky="ew")

        self.slider = PercentileSlider(self)
        self.slider.grid(row=2, column=0, columnspan=2, sticky="ew")


class ImageViewer(tk.Tk):
    def __init__(self, img):
        tk.Tk.__init__(self)
        self.img = img



        self.channel_frames = []
        for i in range(img.shape[1]):
            frame = ChannelFrame(self, i)
            frame.grid(row=1, column=i, sticky="ew")  # Use grid to arrange frames in a row
            self.channel_frames.append(frame)

        self.fig, self.ax = plt.subplots()
        aspect_ratio = img.shape[2] / img.shape[3]

        # Set the figure size to match the aspect ratio of the image
        self.fig.set_size_inches(3, 3 / aspect_ratio)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=2, column=0, columnspan=img.shape[1], sticky="nsew")

        self.timepoint = tk.Scale(self, from_=0, to=img.shape[0] - 1, orient=HORIZONTAL)
        self.timepoint.grid(row=3, column=0, columnspan=img.shape[1], sticky="ew")

        self.update_image()
        self.timepoint.bind("<ButtonRelease-1>", self.update_image)
        for frame in self.channel_frames:
            frame.var.trace_add("write", lambda *args: self.update_image())
            frame.slider.lower.bind("<ButtonRelease-1>", self.update_image)
            frame.slider.upper.bind("<ButtonRelease-1>", self.update_image)


    def update_image(self, event=None):
        timepoint = self.timepoint.get()

        # Extract the channels for the selected timepoint
        channels = self.img[timepoint]

        # Scale each channel to the range [0, 1] using individual percentiles
        processed_channels = []
        for i in range(channels.shape[0]):
            frame = self.channel_frames[i]
            if frame.var.get():
                lower_percentile, upper_percentile = frame.slider.get_values()
                processed_channel = scale_channel(channels[i, :, :], lower_percentile, upper_percentile)
                processed_channels.append(processed_channel)

        # Convert to grayscale
        gray = np.mean(processed_channels, axis=0) if processed_channels else np.zeros_like(channels[0])

        # Convert back to the range [0, 255]
        gray = (gray * 255).astype(np.uint8)

        # Display the image
        self.ax.clear()
        self.ax.imshow(gray, cmap='gray')
        self.ax.axis('off')
        self.canvas.draw()

viewer = ImageViewer(img)
viewer.mainloop()
