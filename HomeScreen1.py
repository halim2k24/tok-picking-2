import os
import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from pypylon import pylon
from datetime import datetime
import json  # Import the json module
from plcsetting import open_plc_settings
from picking_area import set_picking_area, load_picking_area
from top_menu import create_top_menu


def save_picking_area(self):
    # Create a list of bounding box coordinates
    bboxes_coords = []
    for bbox in self.bboxes:
        coords = self.camera_canvas.coords(bbox)
        bboxes_coords.append({
            'start_x': coords[0],
            'start_y': coords[1],
            'end_x': coords[2],
            'end_y': coords[3]
        })

    # Define the path to the JSON file
    json_file_path = os.path.join(os.getcwd(), 'picking_areas.json')

    # Save the bounding box coordinates to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(bboxes_coords, json_file, indent=4)


class HomeScreen:
    def __init__(self, root):
        self.bbox = None
        self.root = root
        self.root.title('Home Screen with Menu and Sections')
        self.root.geometry("1200x800")
        self.root.configure(bg="white")

        self.language = 'English'

        self.record_video_button, self.stop_recording_button = create_top_menu(
            self.root, self.go_home, self.upload_action, self.open_camera, self.start_recording, self.stop_recording,
            self.toggle_language, self.exit_application, self.open_picking_settings
        )

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "model/best2.pt")
        self.model = YOLO(model_path)

        self.main_frame = tk.Frame(self.root, bg="white")
        self.main_frame.pack(fill="both", expand=True)

        self.main_frame.columnconfigure(0, weight=2, minsize=800)
        self.main_frame.columnconfigure(1, weight=1, minsize=200)
        self.main_frame.rowconfigure(0, weight=1)

        self.image_view_frame = tk.Frame(self.main_frame, bg="white")
        self.image_view_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.results_frame = tk.Frame(self.main_frame, bg="lightgray")
        self.results_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.image_view_label = tk.Label(self.image_view_frame, text="Image View", font=("Helvetica", 16, "bold"),
                                         bg="white")
        self.image_view_label.pack(pady=20)

        self.results_label = tk.Label(self.results_frame, text="Results", font=("Helvetica", 16, "bold"),
                                      bg="lightgray")
        self.results_label.pack(pady=20)

        self.camera_canvas = tk.Canvas(self.image_view_frame, width=1400, height=800, bg="gray")
        self.camera_canvas.pack(pady=10)

        self.button_frame = tk.Frame(self.image_view_frame, bg="white")
        self.button_frame.pack(pady=10)

        self.image_picking_button = tk.Button(self.button_frame, text="Image Picking", command=self.image_picking,
                                              bg="blue", fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.image_picking_button.pack(side="left", padx=10)

        self.start_button = tk.Button(self.button_frame, text="Start Picking", command=self.start_picking, bg="green",
                                      fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.start_button.pack(side="left", padx=10)

        self.set_area_button = tk.Button(self.button_frame, text="Set Picking Area",
                                         command=lambda: set_picking_area(self), bg="orange", fg="white",
                                         font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.set_area_button.pack(side="left", padx=10)

        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_picking, bg="red", fg="white",
                                     font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.stop_button.pack(side="left", padx=10)

        self.bboxes = []
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

        self.recording = False
        self.save_folder = None
        self.total_ok = 0
        self.total_ng = 0

        self.date_time_label = tk.Label(self.results_frame, text="", font=("Helvetica", 12, "bold"), bg="lightgray",
                                        fg="black", anchor="w")
        self.date_time_label.pack(pady=5, anchor="w", padx=10)

        self.total_ok_label = tk.Label(self.results_frame, text="Total OK: 0", font=("Helvetica", 12, "bold"),
                                       bg="lightgray", fg="green", anchor="w")
        self.total_ok_label.pack(pady=5, anchor="w", padx=10)

        self.total_ng_label = tk.Label(self.results_frame, text="Total NG: 0", font=("Helvetica", 12, "bold"),
                                       bg="lightgray", fg="red", anchor="w")
        self.total_ng_label.pack(pady=5, anchor="w", padx=10)

        self.count_label = tk.Label(self.results_frame, text="Count: 0", font=("Helvetica", 12, "bold"), bg="lightgray",
                                    fg="blue", anchor="w")
        self.count_label.pack(pady=5, anchor="w", padx=10)

        self.update_date_time()
        load_picking_area(self)

    def open_picking_settings(self):
        messagebox.showinfo("Picking Settings", "Open Picking Settings")

    def open_settings(self):
        open_plc_settings(self.root)

    def update_date_time(self):
        now = datetime.now().strftime("Today: %Y-%m-%d %H:%M:%S")
        self.date_time_label.config(text=now)
        self.root.after(1000, self.update_date_time)

    def toggle_language(self):
        if self.language == 'English':
            self.language = 'Japanese'
        else:
            self.language = 'English'


    def go_home(self):
        self.stop_camera()
        messagebox.showinfo("Home", "You are on the Home screen!")


    def stop_camera(self):
        if hasattr(self, 'camera') and self.camera.IsGrabbing():
            self.camera.StopGrabbing()

    def open_camera(self):
        self.stop_camera()
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_RGB8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            def grab_and_display():
                if self.camera.IsGrabbing():
                    grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    if grab_result.GrabSucceeded():
                        image = self.converter.Convert(grab_result)  # Convert the image to RGB8
                        frame = image.GetArray()  # Get the RGB frame

                        # Apply object detection
                        results = self.model.predict(source=frame, task="detect", show=False)
                        result_image = results[0].plot()
                        result_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGRA2BGR))

                        # Resize the image while maintaining aspect ratio
                        canvas_width = self.camera_canvas.winfo_width()
                        canvas_height = self.camera_canvas.winfo_height()
                        img_width, img_height = result_image.size
                        width_ratio = canvas_width / img_width
                        height_ratio = canvas_height / img_height
                        new_ratio = min(width_ratio, height_ratio)
                        new_width = int(img_width * new_ratio)
                        new_height = int(img_height * new_ratio)
                        result_image = result_image.resize((new_width, new_height), Image.LANCZOS)

                        imgtk = ImageTk.PhotoImage(image=result_image)
                        self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                        self.camera_canvas.imgtk = imgtk

                        # Adjust coordinates based on the resizing ratio
                        for result in results:
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0]  # Ensure correct unpacking
                                center_x = (x1 + x2) / 2 * new_ratio
                                center_y = (y1 + y2) / 2 * new_ratio
                                radius = ((x2 - x1) + (y2 - y1)) / 4 * new_ratio  # Approximate radius
                                # Convert tensor values to integers
                                center_x = int(center_x.item())
                                center_y = int(center_y.item())
                                radius = int(radius.item())
                                self.camera_canvas.create_oval(center_x - radius, center_y - radius, center_x + radius,
                                                               center_y + radius, outline='red')
                                self.camera_canvas.create_text(center_x, center_y, text=f"({center_x}, {center_y})",
                                                               fill='red')
                                # Draw the red center point
                                self.camera_canvas.create_oval(center_x - 2, center_y - 2, center_x + 2, center_y + 2,
                                                               fill='red')

                        # Ensure all circles are on top of the uploaded image
                        for bbox in self.bboxes:
                            self.camera_canvas.tag_raise(bbox)

                    grab_result.Release()
                    self.camera_canvas.after(10, grab_and_display)

            grab_and_display()

        except Exception as e:
            messagebox.showerror("Error", str(e))




    def update_camera_frame(self):
        if self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = self.converter.Convert(grab_result)  # Convert the image to RGB8
                frame = image.GetArray()  # Get the RGB frame

                # Resize the frame to fit the canvas
                canvas_width = self.camera_canvas.winfo_width()
                canvas_height = self.camera_canvas.winfo_height()
                frame = cv2.resize(frame, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)

                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.camera_canvas.imgtk = imgtk
            grab_result.Release()
        self.root.after(10, self.update_camera_frame)


    def start_recording(self):
        self.recording = True
        current_date = datetime.now().strftime("%Y%m%d")
        self.save_folder = os.path.join(os.getcwd(), current_date)
        os.makedirs(self.save_folder, exist_ok=True)
        self.record_video_button.config(state=tk.DISABLED)
        self.stop_recording_button.config(state=tk.NORMAL)

    def stop_recording(self):
        self.recording = False
        self.save_folder = None
        self.record_video_button.config(state=tk.NORMAL)
        self.stop_recording_button.config(state=tk.DISABLED)
        messagebox.showinfo("Recording", "Recording stopped and images saved.")

    def record_video(self):
        self.stop_camera()
        self.open_camera()

    def start_picking(self):
        self.stop_camera()
        self.open_camera()
        self.record_video_button.config(state=tk.NORMAL)

    def stop_picking(self):
        self.stop_camera()
        messagebox.showinfo("Picking", "Stopping Picking...")

    def image_picking(self):
        self.stop_camera()
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
            if file_path:
                [...]
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def upload_action(self):
        self.stop_camera()
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            img = Image.open(file_path)
            max_width, max_height = 1400, 800
            width_ratio = max_width / img.width
            height_ratio = max_height / img.height
            new_ratio = min(width_ratio, height_ratio)
            new_width = int(img.width * new_ratio)
            new_height = int(img.height * new_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            results = self.model.predict(source=frame, task="segment", show=False)
            result_image = results[0].plot()
            result_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=result_image)
            self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.camera_canvas.imgtk = imgtk

            # Draw circles for each detected object and show x, y values
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]  # Ensure correct unpacking
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    radius = ((x2 - x1) + (y2 - y1)) / 4  # Approximate radius
                    # Convert tensor values to integers
                    center_x = int(center_x.item())
                    center_y = int(center_y.item())
                    radius = int(radius.item())
                    self.camera_canvas.create_oval(center_x - radius, center_y - radius, center_x + radius,
                                                   center_y + radius, outline='red')
                    self.camera_canvas.create_text(center_x, center_y, text=f"({center_x}, {center_y})", fill='red',
                                                   anchor=tk.NW)
                    # Draw the red center point
                    self.camera_canvas.create_oval(center_x - 2, center_y - 2, center_x + 2, center_y + 2, fill='red')

            # Ensure all circles are on top of the uploaded image
            for bbox in self.bboxes:
                self.camera_canvas.tag_raise(bbox)



    def exit_application(self):
        self.root.quit()
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        bbox = self.camera_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                   outline='red')
        self.bboxes.append(bbox)
        self.camera_canvas.tag_raise(bbox)  # Ensure the bounding box is on top

    def on_mouse_drag(self, event):
        self.end_x = event.x
        self.end_y = event.y
        if self.bboxes:
            self.camera_canvas.coords(self.bboxes[-1], self.start_x, self.start_y, self.end_x, self.end_y)
            self.camera_canvas.tag_raise(self.bboxes[-1])  # Ensure the bounding box is on top

    def on_button_release(self, event):
        self.end_x = event.x
        self.end_y = event.y
        if self.bboxes:
            self.camera_canvas.coords(self.bboxes[-1], self.start_x, self.start_y, self.end_x, self.end_y)
            self.camera_canvas.tag_raise(self.bboxes[-1])  # Ensure the bounding box is on top
        save_picking_area(self)