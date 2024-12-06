# %%



import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tkinter as tk
import re
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Function to clean and normalize city names
def clean_city_names(city_name):
    city_name = re.sub(r",\s?[a-z]{2}$", "", city_name.strip().lower())  # Remove state abbreviations
    city_name = city_name.replace("-", " ")  # Replace dashes with spaces
    return city_name


# Step 1: Load and Prepare Zillow Data
def prepare_zillow_data():
    zillow_file = "Zillow_Home_Value_Index.csv"
    zillow_data = pd.read_csv(zillow_file)
    zillow_data = zillow_data.rename(columns={"RegionName": "City"})
    zillow_data["City"] = zillow_data["City"].apply(clean_city_names)
    zillow_data["ZHVI"] = zillow_data.iloc[:, -1]  # Use the most recent ZHVI column
    return zillow_data[["City", "ZHVI"]]


# Step 2: Load and Prepare Kaggle Data
def prepare_kaggle_data():
    kaggle_file = "Kaggle_Redfin_Housing.csv"
    kaggle_data = pd.read_csv(kaggle_file)
    kaggle_data = kaggle_data.rename(columns={"Area (SQFT)": "Property_Size", "Beds": "Bedrooms", "Baths": "Bathrooms"})
    kaggle_data["City"] = kaggle_data["City"].apply(clean_city_names)
    kaggle_data["Property_Size"] = pd.to_numeric(kaggle_data["Property_Size"], errors="coerce")
    kaggle_data["Bathrooms"] = pd.to_numeric(kaggle_data["Bathrooms"], errors="coerce")
    return kaggle_data[["City", "Property_Size", "Bedrooms", "Bathrooms", "Price (USD)"]]


# Step 3: Merge Datasets and Prepare All Unique Cities
def merge_and_prepare_cities(zillow_data, kaggle_data):
    merged_data = pd.merge(zillow_data, kaggle_data, on="City", how="inner")
    merged_data.rename(columns={"Price (USD)": "Median_Sales_Price"}, inplace=True)
    merged_data = merged_data.dropna()  # Drop rows with missing values

    # Combine unique cities from both datasets
    all_cities = pd.concat([zillow_data["City"], kaggle_data["City"]]).unique()
    all_cities = sorted(all_cities)  # Sort alphabetically

    return merged_data, all_cities


# Step 4: Train the Model
def train_model(data):
    # Include Bathrooms and Property_Size as features
    X = data[["Property_Size", "Bedrooms", "Bathrooms", "Median_Sales_Price"]]
    y = data["ZHVI"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, scaler


# Load and prepare datasets
zillow_data = prepare_zillow_data()
kaggle_data = prepare_kaggle_data()
merged_data, all_cities = merge_and_prepare_cities(zillow_data, kaggle_data)
model, scaler = train_model(merged_data)


# Function to clear all inputs and calculations
def clear_inputs():
    city_var.set(all_cities[0])  # Reset to the first city in the dropdown
    search_var.set("")  # Clear the search bar
    current_location_label.config(text="Current Location: None", fg="green")  # Reset the current location

    # Clear filters
    for key in filters:
        filters[key].delete(0, tk.END)

    # Clear mortgage calculator inputs
    for key in mortgage_inputs:
        mortgage_inputs[key].delete(0, tk.END)

    # Clear result labels
    price_label.config(text="Predicted ZHVI: ", fg="green")
    mortgage_label.config(text="Mortgage Payment: ", fg="green")


# Filter and Predict Function
def filter_and_predict():
    try:
        selected_city = current_location_label.cget("text").replace("Current Location: ", "").strip().lower()
        if selected_city == "none" or selected_city == "location not found.":
            price_label.config(text="Error: Please select a valid location.")
            return

        filtered_data = merged_data[merged_data["City"] == selected_city]

        for key, column in {
            "min_bedrooms": "Bedrooms",
            "max_bedrooms": "Bedrooms",
            "min_price": "Median_Sales_Price",
            "max_price": "Median_Sales_Price",
            "min_bathrooms": "Bathrooms",
            "max_bathrooms": "Bathrooms",
            "min_property_size": "Property_Size",
            "max_property_size": "Property_Size",
        }.items():
            user_input = filters[key].get()
            if user_input:
                try:
                    value = float(user_input)
                    if "min" in key:
                        filtered_data = filtered_data[filtered_data[column] >= value]
                    elif "max" in key:
                        filtered_data = filtered_data[filtered_data[column] <= value]
                except ValueError:
                    price_label.config(text=f"Error: Invalid input for '{key.replace('_', ' ').title()}'.")
                    return

        if filtered_data.empty:
            price_label.config(text="No data found for the selected filters.")
            return

        avg_features = filtered_data[["Property_Size", "Bedrooms", "Bathrooms", "Median_Sales_Price"]].mean().values
        scaled_features = scaler.transform([avg_features])
        predicted_price = model.predict(scaled_features)[0]
        price_label.config(text=f"Predicted ZHVI: ${predicted_price:,.2f}")
    except Exception as e:
        price_label.config(text=f"Error: {e}")


# Mortgage Calculation Function
def calculate_mortgage():
    try:
        price_text = price_label.cget("text")
        if "Predicted ZHVI:" not in price_text or "Error" in price_text or "No data" in price_text:
            mortgage_label.config(text="Error: No valid predicted price available.")
            return

        predicted_price = float(price_text.split("$")[-1].replace(",", "").strip())
        interest_rate = float(mortgage_inputs["interest_rate"].get()) / 100 / 12
        loan_term = int(mortgage_inputs["loan_term"].get()) * 12
        monthly_payment = predicted_price * interest_rate / (1 - (1 + interest_rate) ** -loan_term)
        mortgage_label.config(text=f"Mortgage Payment: ${monthly_payment:,.2f}")
    except ValueError:
        mortgage_label.config(text="Error: Invalid input for mortgage calculation.")
    except Exception as e:
        mortgage_label.config(text=f"Error: {e}")



# Visualization: Bar Chart for Top Cities by ZHVI
def show_bar_chart():
    if "Predicted ZHVI:" not in price_label.cget("text"):
        tk.messagebox.showerror("Error", "Please calculate the predicted price first!")
        return

    for widget in bar_chart_frame.winfo_children():
        widget.destroy()
    tk.Label(bar_chart_frame, text="Top 5 Cities by ZHVI", font=("Helvetica", 16)).pack(pady=20)

    # Get data for the bar chart
    filtered_data = merged_data.groupby("City").mean().sort_values("ZHVI", ascending=False).head(5)
    cities = filtered_data.index
    values = filtered_data["ZHVI"]

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(cities, values, color="skyblue")
    ax.set_title("Top 5 Cities by ZHVI")
    ax.set_ylabel("ZHVI")
    ax.set_xlabel("Cities")
    ax.tick_params(axis="x", rotation=45)

    # Format y-axis to show full numbers
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Add padding for long city names
    plt.tight_layout()

    # Render the chart in the Tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=bar_chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=20)

    tk.Button(bar_chart_frame, text="Back", command=lambda: switch_frame(details_frame), font=("Helvetica", 14)).pack(pady=10)
    switch_frame(bar_chart_frame)



# Visualization: Scatter Plot for Property Size vs. Price
def show_scatter_plot():
    if "Predicted ZHVI:" not in price_label.cget("text"):
        tk.messagebox.showerror("Error", "Please calculate the predicted price first!")
        return

    for widget in scatter_plot_frame.winfo_children():
        widget.destroy()
    tk.Label(scatter_plot_frame, text="Property Size vs. Price", font=("Helvetica", 16)).pack(pady=20)

    selected_city = current_location_label.cget("text").replace("Current Location: ", "").strip().lower()
    if selected_city == "none" or selected_city == "location not found.":
        tk.Label(scatter_plot_frame, text="Error: Select a valid city for visualization", font=("Helvetica", 14), fg="red").pack(pady=10)
        return

    city_data = merged_data[merged_data["City"] == selected_city]

    # Create Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(city_data["Property_Size"], city_data["Median_Sales_Price"], alpha=0.5)
    ax.set_title(f"Property Size vs. Price in {selected_city.title()}")
    ax.set_xlabel("Property Size (sqft)")
    ax.set_ylabel("Median Sales Price")

    # Format y-axis to show full numbers
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    canvas = FigureCanvasTkAgg(fig, master=scatter_plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=20)

    tk.Button(scatter_plot_frame, text="Back", command=lambda: switch_frame(details_frame), font=("Helvetica", 14)).pack(pady=10)
    switch_frame(scatter_plot_frame)


# Show Price Distribution
def show_price_distribution():
    for widget in price_distribution_frame.winfo_children():
        widget.destroy()
    tk.Label(price_distribution_frame, text="Price Distribution", font=("Helvetica", 16)).pack(pady=20)

    # Filter data based on user input
    selected_city = current_location_label.cget("text").replace("Current Location: ", "").strip().lower()
    if selected_city == "none" or selected_city == "location not found.":
        tk.Label(price_distribution_frame, text="Error: Select a valid city for visualization", font=("Helvetica", 14), fg="red").pack(pady=10)
        return

    filtered_data = merged_data[merged_data["City"] == selected_city]

    # Apply filters for bedrooms, bathrooms, etc.
    filters_map = {
        "min_bedrooms": "Bedrooms",
        "max_bedrooms": "Bedrooms",
        "min_price": "Median_Sales_Price",
        "max_price": "Median_Sales_Price",
        "min_bathrooms": "Bathrooms",
        "max_bathrooms": "Bathrooms",
        "min_property_size": "Property_Size",
        "max_property_size": "Property_Size",
    }
    for key, column in filters_map.items():
        user_input = filters[key].get()
        if user_input:
            try:
                value = float(user_input)
                if "min" in key:
                    filtered_data = filtered_data[filtered_data[column] >= value]
                elif "max" in key:
                    filtered_data = filtered_data[filtered_data[column] <= value]
            except ValueError:
                tk.Label(price_distribution_frame, text=f"Error: Invalid input for '{key}'", font=("Helvetica", 14), fg="red").pack(pady=10)
                return

    # Check if filtered data is empty
    if filtered_data.empty:
        tk.Label(price_distribution_frame, text="No data available for the selected filters.", font=("Helvetica", 14), fg="red").pack(pady=10)
        return

    # Create histogram for filtered data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(filtered_data["Median_Sales_Price"], bins=20, color="skyblue", edgecolor="black")
    ax.set_title(f"Price Distribution in {selected_city.title()}")
    ax.set_xlabel("Median Sales Price")
    ax.set_ylabel("Frequency")

    # Format x-axis and y-axis to show real numbers
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x):,}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y):,}"))


    canvas = FigureCanvasTkAgg(fig, master=price_distribution_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=20)

    tk.Button(price_distribution_frame, text="Back", command=lambda: switch_frame(details_frame), font=("Helvetica", 14)).pack(pady=10)
    switch_frame(price_distribution_frame)





# Function to switch frames
def switch_frame(frame):
    frame.tkraise()

# Create GUI
def create_gui():
    global city_var, search_var, current_location_label, filters, mortgage_inputs, price_label, mortgage_label, bar_chart_frame, scatter_plot_frame, details_frame, price_distribution_frame

    app = tk.Tk()
    app.title("House Hunt: Predicting Your Perfect Price")
    app.geometry("1000x900")

    main_menu_frame = tk.Frame(app)
    details_frame = tk.Frame(app)
    bar_chart_frame = tk.Frame(app)
    scatter_plot_frame = tk.Frame(app)
    price_distribution_frame = tk.Frame(app)
    
    for frame in (main_menu_frame, details_frame, bar_chart_frame, scatter_plot_frame, price_distribution_frame):
        frame.grid(row=0, column=0, sticky="nsew")

    # Main Menu
    tk.Label(main_menu_frame, text="Welcome to House Hunt!", font=("Helvetica", 24)).pack(pady=20)
    tk.Button(main_menu_frame, text="Enter House Details", command=lambda: switch_frame(details_frame), font=("Helvetica", 16)).pack(pady=10)

    # Details Frame
    tk.Label(details_frame, text="Enter House Details:", font=("Helvetica", 18)).pack(pady=10)

    city_var = tk.StringVar()
    tk.Label(details_frame, text="Filter by Location (Dropdown):", font=("Helvetica", 16)).pack(pady=10)
    city_menu = tk.OptionMenu(details_frame, city_var, *all_cities)
    city_menu.pack()

    search_var = tk.StringVar()
    tk.Label(details_frame, text="Search by City Name (Type):", font=("Helvetica", 16)).pack(pady=10)
    tk.Entry(details_frame, textvariable=search_var, font=("Helvetica", 14)).pack()

    current_location_label = tk.Label(details_frame, text="Current Location: None", font=("Helvetica", 14), fg="green")
    current_location_label.pack(pady=10)

    def update_location(source):
        selected_city = city_var.get().strip().lower() if source == "dropdown" else search_var.get().strip().lower()
        current_location_label.config(
            text=f"Current Location: {selected_city.title()}" if selected_city in all_cities else "Location not found.",
            fg="green" if selected_city in all_cities else "red",
        )

    city_var.trace_add("write", lambda *args: update_location("dropdown"))
    search_var.trace_add("write", lambda *args: update_location("search"))

    filter_frame = tk.Frame(details_frame)
    filter_frame.pack(pady=20)

    filters = {}
    filter_fields = {
        "Min Bedrooms": "min_bedrooms",
        "Max Bedrooms": "max_bedrooms",
        "Min Price": "min_price",
        "Max Price": "max_price",
        "Min Bathrooms": "min_bathrooms",
        "Max Bathrooms": "max_bathrooms",
        "Min Property Size": "min_property_size",
        "Max Property Size": "max_property_size",
    }

    row, col = 0, 0
    for label, var in filter_fields.items():
        tk.Label(filter_frame, text=label).grid(row=row, column=col, padx=10, pady=5, sticky="e")
        entry = tk.Entry(filter_frame)
        entry.grid(row=row, column=col + 1, padx=10, pady=5)
        filters[var] = entry
        col += 2
        if col >= 4:
            row, col = row + 1, 0

    tk.Label(details_frame, text="Mortgage Calculator:", font=("Helvetica", 16)).pack(pady=10)
    mortgage_frame = tk.Frame(details_frame)
    mortgage_frame.pack(pady=10)

    mortgage_inputs = {}
    mortgage_fields = {"Interest Rate (%)": "interest_rate", "Loan Term (Years)": "loan_term"}
    for row, (label, var) in enumerate(mortgage_fields.items()):
        tk.Label(mortgage_frame, text=label).grid(row=row, column=0, padx=10, pady=5, sticky="e")
        entry = tk.Entry(mortgage_frame)
        entry.grid(row=row, column=1, padx=10, pady=5)
        mortgage_inputs[var] = entry

    price_label = tk.Label(details_frame, text="Predicted ZHVI: ", font=("Helvetica", 14), fg="green")
    price_label.pack(pady=10)

    mortgage_label = tk.Label(details_frame, text="Mortgage Payment: ", font=("Helvetica", 14), fg="green")
    mortgage_label.pack(pady=10)

    button_frame = tk.Frame(details_frame)
    button_frame.pack(pady=20)

    tk.Button(button_frame, text="Filter and Predict", command=filter_and_predict, font=("Helvetica", 14)).grid(row=0, column=0, padx=10)
    tk.Button(button_frame, text="View Bar Chart", command=show_bar_chart, font=("Helvetica", 14)).grid(row=0, column=1, padx=10)
    tk.Button(button_frame, text="Calculate Mortgage", command=calculate_mortgage, font=("Helvetica", 14)).grid(row=1, column=0, padx=10)
    tk.Button(button_frame, text="View Scatter Plot", command=show_scatter_plot, font=("Helvetica", 14)).grid(row=1, column=1, padx=10)
    tk.Button(button_frame, text="Price Distribution", command=show_price_distribution, font=("Helvetica", 14)).grid(row=0, column=4, padx=5)



    clear_button = tk.Button(details_frame, text="Clear", command=clear_inputs, font=("Helvetica", 14))
    clear_button.place(x=10, y=10)
    tk.Button(details_frame, text="Back to Main Menu", command=lambda: switch_frame(main_menu_frame), font=("Helvetica", 14)).pack(pady=10)

    switch_frame(main_menu_frame)
    app.mainloop()


# Start GUI
create_gui()



# %%

