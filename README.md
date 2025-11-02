#  Pune Property Price Prediction & Recommendation System

## 📋 Project Overview
This project uses **Machine Learning (Random Forest Regression)** to predict **property prices and rent** in Pune based on various features like number of bedrooms, area, facing, and locality. 
It also includes a **recommendation system** that suggests the best areas to buy or rent based on a user’s preferences such as budget, BHK, and carpet area.

---

##  Technologies Used
- **Python 3.11+**
- **Scikit-Learn**
- **Pandas & NumPy**
- **Tkinter** (for GUI)
- **Matplotlib / Seaborn** (for EDA and visualization)

---

##  Machine Learning Model
We used a **Random Forest Regressor** for its robustness and ability to handle both numerical and categorical data.

### Model Training Steps
1. **Data Preprocessing:**
   - Cleaned missing or inconsistent values.
   - Encoded categorical columns (like `facing`, `ownership`, etc.) using `LabelEncoder`.
   - Scaled numeric features for better model performance.

2. **Feature Selection:**
   Used columns:
   ```
   additionalrooms, age, amenitiesavailable, amenitiesnot, area, balconies, 
   bathroom, bhk, carpetarea, facing, floor, locality, neworold, opensides, 
   overlooking, ownership, possesiondate, price, pricepersquare, projectname, 
   roadfaceing, status, totalfloor
   ```

3. **Model Training:**
   ```python
   from sklearn.ensemble import RandomForestRegressor
   model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
   model.fit(X_train, y_train)
   ```

4. **Evaluation Metrics:**
   - **R² Score**
   - **Mean Absolute Error (MAE)**
   - **Root Mean Squared Error (RMSE)**

   Example output:
   ```
   R² Score: 0.91
   MAE: 1.25 Lakhs
   RMSE: 2.85 Lakhs
   ```

5. **Model Saving:**
   ```python
   import joblib
   joblib.dump(model, "property_price_model.pkl")
   ```

---

##  Recommendation System
The system takes user input like:
- **BHK**
- **Carpet Area**
- **Budget**
- **Preferred Locality**

Then, it filters and recommends the **top 5 most suitable areas** in Pune based on:
- Predicted property prices
- Affordability index
- Distance from workplace (optional field for future enhancement)

---

##  GUI Application (Tkinter)
The desktop app allows users to:
- Select **property features** from dropdowns.
- Get **predicted price/rent**.
- See **recommended localities**.
- All wrapped in a clean, responsive Tkinter interface.

Run the app using:
```bash
python app.py
```

---

##  Model Retraining
To retrain the model on updated data:
```bash
python train_model.py
```
This will generate a new model file (`property_price_model.pkl`) which the GUI automatically uses.

---

##  Project Structure
```
│
├── train_model.py        # ML training and model saving script
├── app.py                # Tkinter GUI for predictions & recommendations
├── property_data.csv     # Pune property dataset
├── property_price_model.pkl   # Saved trained Random Forest model
├── README.txt            # Project documentation
└── requirements.txt      # Required Python libraries
```

---

## 🏁 Future Improvements
- Integration with **Google Maps API** to calculate real-time distance to workplace.
- Addition of **rent prediction** as a separate model.
- Deployment as a **Streamlit web app**.

---

##Author
**Atharva Deshmukh**  
B.Tech IT | PCCOE Nigdi  
atharvapdeshmukh2411@gmail.com  
Pune, India
