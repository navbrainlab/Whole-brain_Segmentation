import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import json
import threading

class Parameter_UI(tk.Frame):
    def __init__(self, master):
        super().__init__(master)

        self.selected_file_path = ""

        self.p1_scale = tk.IntVar()
        self.p2_scale = tk.IntVar()
        self.min_contourArea_scale = tk.IntVar()
        self.max_contourArea_scale = tk.IntVar()
        # unit is s
        self.debounce_time = 0.05
        self.width=1500
        self.height=1000

        self.update_timer = None

        self.kernel_close = np.ones((3, 3), np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))



        self.load_parameters()

        self.create_Parameter_UI()


        self.update_image()

    def create_Parameter_UI(self):

        self.p1_scale_widget = tk.Scale(self, label="P1", from_=0, to=200, orient=tk.HORIZONTAL, length=400,variable=self.p1_scale,
                                        command=self.debounced_update)
        self.p1_scale_widget.grid(row=0, column=1, padx=5, pady=5)

        self.p2_scale_widget = tk.Scale(self, label="P2", from_=0, to=200, orient=tk.HORIZONTAL, length=400,variable=self.p2_scale,
                                        command=self.debounced_update)
        self.p2_scale_widget.grid(row=1, column=1, padx=5, pady=5)

        self.min_contourArea_scale_widget = tk.Scale(self, label="min_contourArea", from_=0, to=1000,variable=self.min_contourArea_scale,
                                                     orient=tk.HORIZONTAL, length=400, command=self.debounced_update)
        self.min_contourArea_scale_widget.grid(row=2, column=1, padx=5, pady=5)

        self.max_contourArea_scale_widget = tk.Scale(self, label="max_contourArea", from_=0, to=1000,variable=self.max_contourArea_scale,
                                                     orient=tk.HORIZONTAL, length=400, command=self.debounced_update)
        self.max_contourArea_scale_widget.grid(row=3, column=1, padx=5, pady=5)
        self.back_button = tk.Button(self, text="Return To Main Menu", command=self.show_track_UI, width=60, height=2)
        self.back_button.grid(row=4, columnspan=4, padx=5, pady=5)

        self.choose_file_button = tk.Button(self, text="Choose File", command=self.choose_file, width=20, height=2)
        self.choose_file_button.grid(row=5, column=1, padx=5, pady=5)


        self.canvas = tk.Canvas(self, width=self.width, height=self.height)
        self.canvas.grid(row=6, column=1, padx=5, pady=5)



    def debounced_update(self, event=None):
        if self.update_timer:
            self.update_timer.cancel()

        self.update_timer = threading.Timer(self.debounce_time, self.update_image)
        self.update_timer.start()

    def load_parameters(self):
        try:
            with open("Track_parameters.json", "r") as file:
                data = json.load(file)
                self.p1_scale.set(data.get("P1", "")[0])
                self.p2_scale.set(data.get("P2", "")[0])
                self.min_contourArea_scale.set(data.get("min_wormArea", "")[0])
                self.max_contourArea_scale.set(data.get("max_wormArea", "")[0])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def choose_file(self):

        file_path = filedialog.askopenfilename()
        if file_path:

            self.selected_file_path = file_path

            self.update_image()

    def process_image(self, p1, p2, min_contourArea, max_contourArea):

        video = cv2.VideoCapture(self.selected_file_path)
        ok, frame = video.read()
        image = frame
        frame_gray = image[:, :, 2]
        edges = cv2.Canny(frame_gray, p1, p2)
        image1 = cv2.dilate(edges, self.kernel, iterations=1)
        im_out = cv2.morphologyEx(image1, cv2.MORPH_CLOSE, self.kernel_close)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im_out, connectivity=8)
        filtered_image = np.zeros_like(im_out)
        object_sizes = stats[:, cv2.CC_STAT_AREA]
        selected_labels = np.logical_and(object_sizes > min_contourArea, object_sizes < max_contourArea)
        selected_labels_idx = np.where(selected_labels)[0]
        result = np.isin(labels, selected_labels_idx)
        filtered_image[result] = 255

        img = Image.fromarray(filtered_image)
        img = img.resize((self.width, self.height))
        img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

    def update_image(self,event=None):

        p1 = self.p1_scale_widget.get()
        p2 = self.p2_scale_widget.get()
        min_contourArea = self.min_contourArea_scale_widget.get()
        max_contourArea = self.max_contourArea_scale_widget.get()

        if self.selected_file_path:
            self.process_image(p1, p2, min_contourArea, max_contourArea)

    def show_track_UI(self):
        self.grid_forget()
        self.track_ui_frame.grid(row=0, column=0)

