import tkinter as tk
from tkinter import *
import Track
import single_track
import json
from bar_para import Parameter_UI

class Track_UI(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.Parameter_UI = Parameter_UI(self.master)
        self.P1_var = tk.StringVar()
        self.P2_var = tk.StringVar()
        self.sum_P1_var = tk.StringVar()
        self.sum_P2_var = tk.StringVar()
        self.min_wormArea_var = tk.StringVar()
        self.max_wormArea_var = tk.StringVar()
        self.sum_jump_var = tk.StringVar()
        self.track_jump_var = tk.StringVar()
        self.creat_track_UI()
        self.load_parameters()

    def creat_track_UI(self):

        P1_Label = Label(self, text="P1")
        P1_Label.grid(row=1, column=0, padx=5, pady=5)

        self.P1_value = Entry(self, bd=5, textvariable=self.P1_var)
        self.P1_value.grid(row=1, column=1, padx=5, pady=5)

        P2_Label = Label(self, text="P2")
        P2_Label.grid(row=1, column=2, padx=5, pady=5)
        self.P2_value = Entry(self, bd=5, textvariable=self.P2_var)
        self.P2_value.grid(row=1, column=3, padx=5, pady=5)

        sum_P1_Label = Label(self, text="sum P1")
        sum_P1_Label.grid(row=2, column=0, padx=5, pady=5)

        self.sum_P1_value = Entry(self, bd=5, textvariable=self.sum_P1_var)
        self.sum_P1_value.grid(row=2, column=1, padx=5, pady=5)

        sum_P2_Label = Label(self, text="sum P2")
        sum_P2_Label.grid(row=2, column=2, padx=5, pady=5)
        self.sum_P2_value = Entry(self, bd=5, textvariable=self.sum_P2_var)
        self.sum_P2_value.grid(row=2, column=3, padx=5, pady=5)

        min_wormArea_Label = Label(self, text="min size")
        min_wormArea_Label.grid(row=3, column=0, padx=5, pady=5)
        self.min_wormArea_value = Entry(self, bd=5, textvariable=self.min_wormArea_var)
        self.min_wormArea_value.grid(row=3, column=1, padx=5, pady=5)

        max_wormArea_Label = Label(self, text="max size")
        max_wormArea_Label.grid(row=3, column=2, padx=5, pady=5)
        self.max_wormArea_value = Entry(self, bd=5, textvariable=self.max_wormArea_var)
        self.max_wormArea_value.grid(row=3, column=3, padx=5, pady=5)

        sum_jump_Label = Label(self, text="sum jump")
        sum_jump_Label.grid(row=4, column=0, padx=5, pady=5)
        self.sum_jump_value = Entry(self, bd=5, textvariable=self.sum_jump_var)
        self.sum_jump_value.grid(row=4, column=1, padx=5, pady=5)

        track_jump_Label = Label(self, text="track jump")
        track_jump_Label.grid(row=4, column=2, padx=5, pady=5)
        self.track_jump_value = Entry(self, bd=5, textvariable=self.track_jump_var)
        self.track_jump_value.grid(row=4, column=3, padx=5, pady=5)

        show_boolean = tk.BooleanVar()
        show_boolean.set(False)  # 默认为False
        show_checkbox = tk.Checkbutton(self, text="Show", variable=show_boolean, width=26)
        show_checkbox.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

        save_boolean = tk.BooleanVar()
        save_boolean.set(False)  # 默认为False
        remember_checkbox = tk.Checkbutton(self, text="Save Change", command=self.remember_parameter, width=26)
        remember_checkbox.grid(row=5, column=2, columnspan=2, padx=5, pady=5)

        button0 = tk.Button(self, text="Single Track",
                            command=lambda: single_track.single_track(self, P1=int(self.P1_value.get()),
                                                                                P2=int(self.P2_value.get()),
                                                                                sum_P1=int(self.sum_P1_value.get()),
                                                                                sum_P2=int(self.sum_P2_value.get()),
                                                                                min_wormArea=int(
                                                                                    self.min_wormArea_value.get()),
                                                                                max_wormArea=int(
                                                                                    self.max_wormArea_value.get()),
                                                                                sum_jump=int(self.sum_jump_value.get()),
                                                                                track_jump=int(
                                                                                    self.track_jump_value.get()),
                                                                                show_value=show_boolean.get()),
                            width=26,
                            height=5)
        button0.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

        button1 = tk.Button(self, text="Track",
                            command=lambda: Track.video_file(self, P1=int(self.P1_value.get()),
                                                             P2=int(self.P2_value.get()),
                                                             sum_P1=int(self.sum_P1_value.get()),
                                                             sum_P2=int(self.sum_P2_value.get()),
                                                             min_wormArea=int(
                                                                 self.min_wormArea_value.get()),
                                                             max_wormArea=int(
                                                                 self.max_wormArea_value.get()),
                                                             sum_jump=int(self.sum_jump_value.get()),
                                                             track_jump=int(self.track_jump_value.get()),
                                                             show_value=show_boolean.get()), width=26,
                            height=5)
        button1.grid(row=6, column=2, columnspan=2, padx=5, pady=5)

        back_button = tk.Button(self, text="Return To Main Menu ", command=self.show_main_frame, width=60, height=2)
        back_button.grid(row=7, columnspan=4, padx=5, pady=5)

        back_button = tk.Button(self, text="Parameter_UI ", command=self.show_Parameter_UI, width=60, height=2)
        back_button.grid(row=8, columnspan=4, padx=5, pady=5)

    def test(self):

        self.destroy()

    def show_Parameter_UI(self):
        self.grid_forget()
        self.parameter_ui_frame.grid(row=0, column=0)


    def show_main_frame(self):
        self.grid_forget()
        self.master.show_main_UI()

    def remember_parameter(self):
        P1 = int(self.P1_value.get()),
        P2 = int(self.P2_value.get()),
        sum_P1 = int(self.sum_P1_value.get()),
        sum_P2 = int(self.sum_P2_value.get()),
        min_wormArea = int(
            self.min_wormArea_value.get()),
        max_wormArea = int(
            self.max_wormArea_value.get()),
        sum_jump = int(self.sum_jump_value.get()),
        track_jump = int(self.track_jump_value.get())
        with open("Track_parameters.json", "w") as file:
            json.dump({"P1": P1, "P2": P2, "sum_P1": sum_P1, "sum_P2": sum_P2, "min_wormArea": min_wormArea,
                       "max_wormArea": max_wormArea, 'sum_jump': sum_jump, 'track_jump': track_jump}, file)
            # print(f"记忆参数为:","P1",P1, "P2", P2, "sum_P1",sum_P1, "sum_P2",sum_P2, "min_wormArea",min_wormArea,
            #            "max_wormArea", max_wormArea, 'sum_jump', sum_jump, 'track_jump', track_jump)

    def load_parameters(self):
        try:
            with open("Track_parameters.json", "r") as file:
                data = json.load(file)
                self.P1_var.set(data.get("P1", ""))
                self.P2_var.set(data.get("P2", ""))
                self.sum_P1_var.set(data.get("sum_P1", ""))
                self.sum_P2_var.set(data.get("sum_P2", ""))
                self.min_wormArea_var.set(data.get("min_wormArea", ""))
                self.max_wormArea_var.set(data.get("max_wormArea", ""))
                self.sum_jump_var.set(data.get("sum_jump", ""))
                self.track_jump_var.set(data.get("track_jump", ""))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
            # print("未找到有效的保存参数文件，将使用默认参数")
