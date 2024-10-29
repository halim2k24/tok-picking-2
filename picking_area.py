import json
import os
from tkinter import messagebox

def set_picking_area(self):
    self.camera_canvas.bind("<ButtonPress-1>", self.on_button_press)
    self.camera_canvas.bind("<B1-Motion>", self.on_mouse_drag)
    self.camera_canvas.bind("<ButtonRelease-1>", self.on_button_release)

def on_button_press(self, event):
    self.start_x = event.x
    self.start_y = event.y
    self.bbox = self.camera_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                    outline="red")

def on_mouse_drag(self, event):
    cur_x, cur_y = (event.x, event.y)
    self.camera_canvas.coords(self.bbox, self.start_x, self.start_y, cur_x, cur_y)

def on_button_release(self, event):
    self.end_x, self.end_y = (event.x, event.y)
    self.camera_canvas.unbind("<ButtonPress-1>")
    self.camera_canvas.unbind("<B1-Motion>")
    self.camera_canvas.unbind("<ButtonRelease-1>")
    messagebox.showinfo("Picking Area",
                        f"Picking area set from ({self.start_x}, {self.start_y}) to ({self.end_x}, {self.end_y})")
    save_picking_area(self)
    self.start_picking()

def save_picking_area(self):
    picking_area = {
        "start_x": self.start_x,
        "start_y": self.start_y,
        "end_x": self.end_x,
        "end_y": self.end_y
    }
    with open("picking_area.json", "w") as f:
        json.dump(picking_area, f)

def load_picking_area(self):
    if os.path.exists("picking_area.json"):
        with open("picking_area.json", "r") as f:
            picking_area = json.load(f)
            self.start_x = picking_area["start_x"]
            self.start_y = picking_area["start_y"]
            self.end_x = picking_area["end_x"]
            self.end_y = picking_area["end_y"]