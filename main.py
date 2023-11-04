import os

import numpy as np
import tifffile
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import HORIZONTAL, Checkbutton, IntVar, Label, Entry, Button
import json
import argparse
from os.path import join
from tifffile import TiffFile


def scale_channel(channel, lower_percentile, upper_percentile):
    p_low = lower_percentile
    p_high = upper_percentile
    channel = (channel - p_low) / (p_high - p_low)
    channel = np.clip(channel, 0, 1)
    return channel

class PercentileSlider(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Create the lower percentile slider
        self.lower = tk.Scale(self, from_=0, to=2**16, orient=HORIZONTAL, showvalue=False, command=self.update_entry_from_slider)
        self.lower.grid(row=0, column=0, sticky="ew")
        self.lower.set(0)

        # Create the entry field for the lower percentile
        self.lower_entry = tk.Entry(self, width=10)
        self.lower_entry.grid(row=0, column=1, sticky="ew")
        self.lower_entry.insert(0, "0")

        # Create the upper percentile slider
        self.upper = tk.Scale(self, from_=0, to=2**16, orient=HORIZONTAL, showvalue=False, command=self.update_entry_from_slider)
        self.upper.grid(row=1, column=0, sticky="ew")
        self.upper.set(2**16)

        # Create the entry field for the upper percentile
        self.upper_entry = tk.Entry(self, width=10)
        self.upper_entry.grid(row=1, column=1, sticky="ew")
        self.upper_entry.insert(0, str(2**16))

        # Bind the 'Return' key on entry fields to update the sliders
        # self.lower_entry.bind("<Return>", self.update_slider_from_entry)
        # self.upper_entry.bind("<Return>", self.update_slider_from_entry)

    def update_entry_from_slider(self, value=None):
        # Update entry fields to match the slider values
        print('update_entry_from_slider')
        self.lower_entry.delete(0, tk.END)
        self.lower_entry.insert(0, str(self.lower.get()))
        self.upper_entry.delete(0, tk.END)
        self.upper_entry.insert(0, str(self.upper.get()))

    def update_slider_from_entry(self, event=None):
        # Update sliders to match the entry fields, with validation
        try:
            lower_val = int(self.lower_entry.get())
            upper_val = int(self.upper_entry.get())
            # if lower_val > upper_val:  # Enforce the rule that lower <= upper
            #     if event.widget == self.lower_entry:
            #         self.upper.set(lower_val)
            #         self.upper_entry.delete(0, tk.END)
            #         self.upper_entry.insert(0, str(lower_val))
            #     else:
            #         self.lower.set(upper_val)
            #         self.lower_entry.delete(0, tk.END)
            #         self.lower_entry.insert(0, str(upper_val))
            # else:
            self.lower.set(lower_val)
            self.upper.set(upper_val)

        except ValueError:
            raise ValueError("Invalid percentile value")  # Handle the case where the entry field does not contain a valid integer
            pass  # Handle the case where the entry field does not contain a valid integer

    def get_values(self):
        # Return the current values of the sliders
        return self.lower.get(), self.upper.get()




class ChannelFrame(tk.Frame):
    def __init__(self, master, channel, **kwargs):
        super().__init__(master, **kwargs)
        self.channel = channel
        self.channel_names = master.channel_names

        self.var = IntVar()
        self.var.set(1)  # default is selected

        self.title_label = Label(self, text=self.channel_names[channel])
        self.title_label.grid(row=0, column=1, sticky="ew")

        self.checkbutton = Checkbutton(self, variable=self.var)
        self.checkbutton.grid(row=0, column=0, sticky="ew")

        self.slider = PercentileSlider(self)
        self.slider.grid(row=2, column=0, columnspan=2, sticky="ew")
    def set_params(self, params):
        self.var.set(params['selected'])
        self.slider.lower.set(params['lower_percentile'])
        self.slider.upper.set(params['upper_percentile'])

    def get_params(self):
        return {
            'selected': self.var.get(),
            'lower_percentile': self.slider.lower.get(),
            'upper_percentile': self.slider.upper.get(),
        }

class ImageViewer(tk.Tk):
    def __init__(self, img, channel_names, initial_params=None):
        tk.Tk.__init__(self)
        self.img = img
        self.channel_names = channel_names

        self.channel_frames = []
        for i in range(img.shape[1]):
            frame = ChannelFrame(self, i)
            frame.grid(row=1, column=i, sticky="ew")
            if initial_params is not None:
                frame.set_params(initial_params[i])
            self.channel_frames.append(frame)

        self.fig, self.ax = plt.subplots()
        aspect_ratio = img.shape[2] / img.shape[3]
        self.fig.set_size_inches(3.8, 3.8 / aspect_ratio)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=2, column=0, columnspan=img.shape[1], sticky="nsew")

        self.timepoint = tk.Scale(self, from_=0, to=img.shape[0] - 1, orient=HORIZONTAL)
        self.timepoint.grid(row=3, column=0, columnspan=img.shape[1], sticky="ew")

        self.filename_entry = Entry(self)
        self.filename_entry.grid(row=4, column=0, sticky="ew")

        self.save_button = Button(self, text="Save Params", command=self.save_params)
        self.save_button.grid(row=4, column=1, sticky="ew")

        self.update_image()
        self.timepoint.bind("<ButtonRelease-1>", self.update_image)
        for frame in self.channel_frames:
            frame.var.trace_add("write", lambda *args: self.update_image())
            frame.slider.lower.bind("<ButtonRelease-1>", self.update_image)
            frame.slider.upper.bind("<ButtonRelease-1>", self.update_image)
            frame.slider.lower_entry.bind("<Return>", self.entry_update)
            frame.slider.upper_entry.bind("<Return>", self.entry_update)

    def entry_update(self, event=None):
        print()
        for frame in self.channel_frames:
            frame.slider.update_slider_from_entry()
            self.update_image()

    def save_params(self):
        filename = self.filename_entry.get()
        if not filename:
            print("Please enter a filename")
            return

        params = [frame.get_params() for frame in self.channel_frames]
        os.makedirs('saves', exist_ok=True)
        with open(join('saves', filename), 'w') as f:
            json.dump(params, f)

        print(f"Params saved to saves/{filename}")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='File from which to load processing parameters')
    args = parser.parse_args()
    image_file = '/Users/jg/Desktop/DSL/4T1 timepoints FACS.HTD - Well A03 Field #1.tif'
    with TiffFile(image_file) as tif:
        original_img = tif.asarray()
        axes = tif.series[0].axes
        imagej_metadata = tif.imagej_metadata
        imagej_metadata['axes'] = axes

    img = original_img.astype(np.float32)
    channel_names = ['brightfield', 'green', 'red', 'far red']

    if not args.file:
        viewer = ImageViewer(img, channel_names)
        viewer.mainloop()
    else:
        with open(join('saves', args.file)) as f:
            params = json.load(f)
        delete = []
        for c, p in enumerate(params):
            if p['selected']:
                img[:, c, :, :] = scale_channel(img[:, c, :, :], p['lower_percentile'], p['upper_percentile'])
            else:
                delete.append(c)

        img = np.delete(img, delete, axis=1)


        # Convert to grayscale
        gray = np.mean(img, axis=1, keepdims=True)
        p_low = 0
        p_high = gray.max()
        gray = (gray - p_low) / (p_high - p_low)
        gray = np.clip(gray, 0, 1)
        gray = gray * 2 ** 16
        gray = gray.astype(np.uint16)

        new_image = np.hstack([original_img, gray])
        print()

        imagej_metadata['channels'] = new_image.shape[1]
        imagej_metadata['min'] = np.min(new_image)
        imagej_metadata['max'] = np.max(new_image)

        os.makedirs('processed', exist_ok=True)
        tifffile.imwrite(
            join('processed', os.path.basename(image_file)),
            new_image,
            imagej=True,
            # resolution=(1./2.6755, 1./2.6755),
            metadata=imagej_metadata
        )




if __name__ == "__main__":
    main()
