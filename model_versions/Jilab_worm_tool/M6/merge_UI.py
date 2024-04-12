import tkinter as tk
from tkinter import *
from Merge import merge
import json

class Merge_UI(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        self.noise_clear_area_var = tk.StringVar()
        self.cluster_number1_var = tk.StringVar()
        self.cluster_number2_var = tk.StringVar()
        self.time_threshold_var = tk.StringVar()
        self.time_number_var = tk.StringVar()
        self.max_time_var = tk.StringVar()
        self.max_distance_var = tk.StringVar()
        self.area_threshold_var = tk.StringVar()
        self.typical_dl_var = tk.StringVar()
        self.noise_clear_area2_var = tk.StringVar()

        self.creat_merge_UI()
        self.load_parameters()

    def creat_merge_UI(self):

        P1_Label = Label(self, text="noise_clear_area")
        P1_Label.grid(row=1, column=0, padx=5, pady=5)

        self.P1_value = Entry(self, bd=5, textvariable=self.noise_clear_area_var)
        self.P1_value.grid(row=1, column=1, padx=5, pady=5)

        P2_Label = Label(self, text="cluster_number1")
        P2_Label.grid(row=1, column=2, padx=5, pady=5)
        self.P2_value = Entry(self, bd=5, textvariable=self.cluster_number1_var)
        self.P2_value.grid(row=1, column=3, padx=5, pady=5)

        sum_P1_Label = Label(self, text="cluster_number2")
        sum_P1_Label.grid(row=2, column=0, padx=5, pady=5)

        self.sum_P1_value = Entry(self, bd=5, textvariable=self.cluster_number2_var)
        self.sum_P1_value.grid(row=2, column=1, padx=5, pady=5)

        sum_P2_Label = Label(self, text="time_threshold")
        sum_P2_Label.grid(row=2, column=2, padx=5, pady=5)
        self.sum_P2_value = Entry(self, bd=5, textvariable=self.time_threshold_var)
        self.sum_P2_value.grid(row=2, column=3, padx=5, pady=5)

        min_wormArea_Label = Label(self, text="time_number")
        min_wormArea_Label.grid(row=3, column=0, padx=5, pady=5)
        self.min_wormArea_value = Entry(self, bd=5, textvariable=self.time_number_var)
        self.min_wormArea_value.grid(row=3, column=1, padx=5, pady=5)

        max_wormArea_Label = Label(self, text="max_time")
        max_wormArea_Label.grid(row=3, column=2, padx=5, pady=5)
        self.max_wormArea_value = Entry(self, bd=5, textvariable=self.max_time_var)
        self.max_wormArea_value.grid(row=3, column=3, padx=5, pady=5)

        sum_jump_Label = Label(self, text="max_distance")
        sum_jump_Label.grid(row=4, column=0, padx=5, pady=5)
        self.sum_jump_value = Entry(self, bd=5, textvariable=self.max_distance_var)
        self.sum_jump_value.grid(row=4, column=1, padx=5, pady=5)

        area_threshold_Label = Label(self, text="area_threshold")
        area_threshold_Label.grid(row=4, column=2, padx=5, pady=5)
        self.area_threshold_value = Entry(self, bd=5, textvariable=self.area_threshold_var)
        self.area_threshold_value.grid(row=4, column=3, padx=5, pady=5)

        typical_dl_Label = Label(self, text="typical_dl")
        typical_dl_Label.grid(row=5, column=0, padx=5, pady=5)
        self.typical_dl_value = Entry(self, bd=5, textvariable=self.typical_dl_var)
        self.typical_dl_value.grid(row=5, column=1, padx=5, pady=5)


        noise_clear_area2_Label = Label(self, text="noise_clear_area2")
        noise_clear_area2_Label.grid(row=5, column=2, padx=5, pady=5)
        self.noise_clear_area2_Label_value = Entry(self, bd=5, textvariable=self.noise_clear_area2_var)
        self.noise_clear_area2_Label_value.grid(row=5, column=3, padx=5, pady=5)

        show_boolean = tk.BooleanVar()
        show_boolean.set(False)  # 默认为False
        show_checkbox = tk.Checkbutton(self, text="Show", variable=show_boolean, width=26)
        show_checkbox.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

        save_boolean = tk.BooleanVar()
        save_boolean.set(False)  # 默认为False
        remember_checkbox = tk.Checkbutton(self, text="Save Change", command=self.remember_parameter, width=26)
        remember_checkbox.grid(row=6, column=2, columnspan=2, padx=5, pady=5)

        button0 = tk.Button(self, text="Start merge",
                            command=lambda: merge(self, noise_clear_area=int(self.P1_value.get()),
                                                                                cluster_number_1=int(self.P2_value.get()),
                                                                                cluster_number_2=int(self.sum_P1_value.get()),
                                                                                time_threshold=int(self.sum_P2_value.get()),
                                                                                time_number=int(
                                                                                    self.min_wormArea_value.get()),
                                                                                max_time=int(
                                                                                    self.max_wormArea_value.get()),
                                                                                max_distance=(self.sum_jump_value.get()),
                                                                                area_threshold=int(
                                                                                    self.area_threshold_value.get()),
                                                                                typical_diagonal_Length_threshold=int(
                                                                                    self.typical_dl_value.get()),
                                                                                noise_clear_area_2=int(
                                                                                    self.noise_clear_area2_Label_value.get()),
                                                                                show_value=show_boolean.get()),
                            width=26,
                            height=5)
        button0.grid(row=7, column=0, columnspan=4, padx=5, pady=5)


        back_button = tk.Button(self, text="Return To Main Menu ", command=self.show_main_frame, width=60, height=2)
        back_button.grid(row=8, columnspan=4, padx=5, pady=5)


    def show_main_frame(self):
        self.grid_forget()
        self.master.show_main_UI()

    def remember_parameter(self):
        noise_clear_area_var = int(self.P1_value.get()),
        cluster_number1_var = int(self.P2_value.get()),
        cluster_number2_var = int(self.sum_P1_value.get()),
        time_threshold_var = int(self.sum_P2_value.get()),
        time_number_var = int(
            self.min_wormArea_value.get()),
        max_time_var = int(
            self.max_wormArea_value.get()),
        max_distance_var = (self.sum_jump_value.get()),
        area_threshold_var = int(self.area_threshold_value.get())

        typical_dl_var = int(self.typical_dl_value.get()),
        noise_clear_area2_var = int(self.noise_clear_area2_Label_value.get())

        with open("merge_parameters.json", "w") as file:
            json.dump({"noise_clear_area_var": noise_clear_area_var, "cluster_number1_var": cluster_number1_var, "cluster_number2_var": cluster_number2_var, "time_threshold_var": time_threshold_var, "time_number_var": time_number_var,
                       "max_time_var": max_time_var, 'max_distance_var': max_distance_var, 'area_threshold_var': typical_dl_var,'typical_dl_var': area_threshold_var,'noise_clear_area2_var': noise_clear_area2_var}, file)


    def load_parameters(self):
        try:
            with open("merge_parameters.json", "r") as file:
                data = json.load(file)
                self.noise_clear_area_var.set(data.get("noise_clear_area_var", ""))
                self.cluster_number1_var.set(data.get("cluster_number1_var", ""))
                self.cluster_number2_var.set(data.get("cluster_number2_var", ""))
                self.time_threshold_var.set(data.get("time_threshold_var", ""))
                self.time_number_var.set(data.get("time_number_var", ""))
                self.max_time_var.set(data.get("max_time_var", ""))
                self.max_distance_var.set(data.get("max_distance_var", ""))
                self.area_threshold_var.set(data.get("area_threshold_var", ""))
                self.typical_dl_var.set(data.get("typical_dl_var", ""))
                self.noise_clear_area2_var.set(data.get("noise_clear_area2_var", ""))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
            # print("未找到有效的保存参数文件，将使用默认参数")
