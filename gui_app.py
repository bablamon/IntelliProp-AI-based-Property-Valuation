# gui_app.py
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np

# =======================
# Load trained model + encoders
# =======================
model = joblib.load("model/random_forest_model.pkl")
encoders = joblib.load("model/encoder.pkl")
scaler = joblib.load("model/scaler.pkl")

# Load property data for locality dropdown + recommendations
data = pd.read_csv("data/pune_properties.csv")
if "locality" not in data.columns:
    raise KeyError("❌ Your dataset must contain a 'locality' column for dropdown to work!")

# =======================
# Tkinter GUI Setup
# =======================
root = tk.Tk()
root.title("🏠 Pune Property Price Predictor")
root.geometry("600x520")
root.resizable(False, False)

tk.Label(root, text="🏙️ Pune Property Price Predictor", font=("Segoe UI", 18, "bold")).pack(pady=15)

frame = tk.Frame(root)
frame.pack(pady=10)

# =======================
# Input Fields
# =======================
labels = [
    "Bedrooms (BHK)",
    "Carpet Area (sqft)",
    "Bathrooms",
    "Budget (₹ Lakhs)",
    "Workplace Locality"
]
entries = {}

# Add entry boxes for first 4 fields
for i, lbl in enumerate(labels[:-1]):
    tk.Label(frame, text=lbl, font=("Segoe UI", 11)).grid(row=i, column=0, sticky="w", pady=5)
    ent = tk.Entry(frame, width=25)
    ent.grid(row=i, column=1, padx=10)
    entries[lbl] = ent

# Dropdown for locality
tk.Label(frame, text=labels[-1], font=("Segoe UI", 11)).grid(row=4, column=0, sticky="w", pady=5)

localities = sorted(data["locality"].dropna().unique().tolist())
locality_var = tk.StringVar()
locality_dropdown = ttk.Combobox(frame, textvariable=locality_var, values=localities, width=22, state="readonly")
locality_dropdown.grid(row=4, column=1, padx=10)
locality_dropdown.current(0)

entries[labels[-1]] = locality_var

# =======================
# Result label
# =======================
result_label = tk.Label(root, text="", font=("Segoe UI", 12), fg="blue")
result_label.pack(pady=15)

# =======================
# Prediction Logic
# =======================
def predict():
    try:
        bhk = float(entries["Bedrooms (BHK)"].get())
        carpet = float(entries["Carpet Area (sqft)"].get())
        bath = float(entries["Bathrooms"].get())
        budget = float(entries["Budget (₹ Lakhs)"].get()) * 100000  # convert to rupees
        workplace = entries["Workplace Locality"].get().strip()

        if not workplace:
            messagebox.showwarning("Input Required", "Please select a workplace locality.")
            return

        # Prepare input DataFrame
        x = pd.DataFrame([[bhk, carpet, bath]], columns=["bhk", "carpetarea", "bathroom"])

        # Encode missing categorical columns (safe default 0)
        for col, le in encoders.items():
            x[col] = 0

        # Scale features
        x_scaled = scaler.transform(x.reindex(columns=scaler.feature_names_in_, fill_value=0))

        # Predict price
        pred_price = model.predict(x_scaled)[0]
        result_label.config(text=f"🏡 Estimated Property Price: ₹{pred_price:,.0f}")

        # Locality recommendation logic
        data["affordability"] = abs(data["price"] - budget)
        recommended = (
            data.groupby("locality")["affordability"]
            .mean()
            .nsmallest(3)
            .index.tolist()
        )

        messagebox.showinfo("🏘️ Recommended Localities",
                            f"Based on your budget and preferences, consider:\n\n" +
                            "\n".join(f"• {loc}" for loc in recommended))

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numerical values for BHK, Area, Bathroom, and Budget.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# =======================
# Button
# =======================
tk.Button(
    root,
    text="🔍 Predict & Recommend",
    command=predict,
    font=("Segoe UI", 12),
    bg="#0078D7",
    fg="white",
    relief="raised",
    padx=10,
    pady=5
).pack(pady=10)

root.mainloop()
