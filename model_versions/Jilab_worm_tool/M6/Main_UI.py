import tkinter as tk
from track_UI import Track_UI
from merge_UI import Merge_UI
from bar_para import Parameter_UI
from Visualisation_compression import compress_multi_vid
from Visualisation_compression import Vis_multi_vid
import multiprocessing
from threading import Lock
import concurrent.futures

class SampleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ji-Lab Worm Tool")

        self.Parameter_UI = Parameter_UI(self)
        self.main_UI = tk.Frame(self)
        self.Track_UI = Track_UI(self)
        self.Merge_UI = Merge_UI(self)
        self.Track_UI.parameter_ui_frame = self.Parameter_UI
        self.Parameter_UI.track_ui_frame = self.Track_UI
        self.create_main_UI()
        self.show_main_UI()

    def create_main_UI(self):
        button1 = tk.Button(self.main_UI, text="Track",command=self.show_Track_UI,width=30, height=5)
        button1.grid(row=0, column=0)  # 使用 grid 布局

        button2 = tk.Button(self.main_UI, text="Merge", command=self.show_Merge_UI,width=30, height=5)
        button2.grid(row=1, column=0)

        button3 = tk.Button(self.main_UI, text="Compression", command=lambda: compress_multi_vid(self.main_UI),
                            width=30, height=5)
        button3.grid(row=2, column=0)

        button4 = tk.Button(self.main_UI, text="Visualisation", command=lambda: Vis_multi_vid(self.main_UI),
                            width=30, height=5)
        button4.grid(row=3, column=0)

        # self.main_UI.grid(row=0, column=0)
    #
    def show_main_UI(self):
        self.Track_UI.grid_forget()
        self.Merge_UI.grid_forget()
        self.main_UI.grid(row=0, column=0)

    def show_Track_UI(self):
        self.main_UI.grid_forget()
        self.Track_UI.grid(row=0, column=0)

    def show_Merge_UI(self):
        self.main_UI.grid_forget()
        self.Merge_UI.grid(row=0, column=0)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = SampleApp()
    app.mainloop()