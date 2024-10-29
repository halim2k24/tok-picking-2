import tkinter as tk
from tkinter import messagebox
from pymodbus.client import ModbusTcpClient  # Correct import
import json
import os

class PLCCommunicator:
    def __init__(self, ip_address, port=502):  # Default Modbus port is 502
        self.ip_address = ip_address
        self.port = port
        self.client = None

    def connect(self):
        try:
            self.client = ModbusTcpClient(self.ip_address, port=self.port)
            if self.client.connect():
                messagebox.showinfo("Connection Status", "Connected to PLC successfully.")
            else:
                messagebox.showerror("Connection Error", "Failed to connect to PLC.")
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))

    def disconnect(self):
        if self.client:
            self.client.close()
            messagebox.showinfo("Connection Status", "Disconnected from PLC.")

    def read_register(self, address, count=1):
        try:
            result = self.client.read_holding_registers(address, count)
            if result.isError():
                messagebox.showerror("Read Error", f"Failed to read register at {address}")
            else:
                return result.registers[0]  # Returning the first register value
        except Exception as e:
            messagebox.showerror("Read Error", str(e))
            return None

    def write_register(self, address, value):
        try:
            result = self.client.write_register(address, value)
            if result.isError():
                messagebox.showerror("Write Error", f"Failed to write value to register {address}")
            else:
                messagebox.showinfo("Write Status", f"Successfully wrote {value} to register {address}")
        except Exception as e:
            messagebox.showerror("Write Error", str(e))

    def test_connection(self):
        try:
            self.client = ModbusTcpClient(self.ip_address, port=self.port)
            if self.client.connect():
                self.client.close()
                return True
            else:
                return False
        except Exception as e:
            messagebox.showerror("Connection Test Error", str(e))
            return False

def save_plc_settings(ip_address, port):
    settings = {
        "ip_address": ip_address,
        "port": port
    }
    with open("plc_settings.json", "w") as f:
        json.dump(settings, f)

def load_plc_settings():
    if os.path.exists("plc_settings.json"):
        with open("plc_settings.json", "r") as f:
            settings = json.load(f)
            return settings["ip_address"], settings["port"]
    return "", 502  # Default values

def open_plc_settings(root):
    plc_window = tk.Toplevel(root)
    plc_window.title("PLC Settings")
    plc_window.geometry("800x600")

    # Calculate the center position
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (800 // 2)
    y = (root.winfo_screenheight() // 2) - (600 // 2)
    plc_window.geometry(f"800x600+{x}+{y}")

    tk.Label(plc_window, text="PLC IP Address:", font=("Helvetica", 12)).pack(pady=10)
    ip_entry = tk.Entry(plc_window, font=("Helvetica", 12))
    ip_entry.pack(pady=5)

    tk.Label(plc_window, text="PLC Port:", font=("Helvetica", 12)).pack(pady=10)
    port_entry = tk.Entry(plc_window, font=("Helvetica", 12))
    port_entry.pack(pady=5)

    # Load settings
    ip_address, port = load_plc_settings()
    ip_entry.insert(0, ip_address)
    port_entry.insert(0, str(port))

    def save_plc_settings():
        ip_address = ip_entry.get()
        port = int(port_entry.get()) if port_entry.get().isdigit() else 502  # Default Modbus TCP port is 502
        plc_communicator = PLCCommunicator(ip_address, port)
        if plc_communicator.test_connection():
            messagebox.showinfo("Connection Status", "Connection to PLC successful.")
            save_plc_settings(ip_address, port)  # Save settings to file
        else:
            messagebox.showerror("Connection Status", "Failed to connect to PLC.")

    save_button = tk.Button(plc_window, text="Save", command=save_plc_settings, font=("Helvetica", 12), bg="green", fg="white")
    save_button.pack(pady=20)

if __name__ == '__main__':
    root = tk.Tk()
    open_plc_settings(root)
    root.mainloop()