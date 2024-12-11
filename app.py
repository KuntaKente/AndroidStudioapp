## app.py – Backend Code

from flask import Flask, request, jsonify, send_file
import requests
import googlemaps
import io
import cv2
import numpy as np

app = Flask(__name__)

# Replace with your Google API Key
API_KEY = 'AIzaSyB48hAwacBNlyMxykH9GJc-k_x4A0fmXbc'

# Initialize Google Maps Client
gmaps = googlemaps.Client(key=API_KEY)

### API Endpoints ###

# 1. Geocode Address
@app.route('/geocode', methods=['POST'])
def geocode_address():
    data = request.json
    address = data.get("address")
    if not address:
        return jsonify({"error": "Address is required"}), 400
    
    result = gmaps.geocode(address)
    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "Geocoding failed"}), 500

# 2. Get Nearby Parking Locations
@app.route('/nearby_parking', methods=['POST'])
def nearby_parking():
    data = request.json
    location = data.get("location")
    radius = data.get("radius", 1000)
    
    if not location:
        return jsonify({"error": "Location is required"}), 400
    
    places_result = gmaps.places_nearby(
        location=location,
        radius=radius,
        type="parking"
    )
    return jsonify(places_result)

# 3. Retrieve Street View Image
@app.route('/street_view', methods=['POST'])
def street_view():
    data = request.json
    location = data.get("location")
    if not location:
        return jsonify({"error": "Location is required"}), 400
    
    street_view_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x480",
        "location": location,
        "fov": "90",
        "heading": "0",
        "pitch": "0",
        "key": API_KEY
    }
    response = requests.get(street_view_url, params=params)
    
    if response.status_code == 200:
        return send_file(
            io.BytesIO(response.content),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='street_view.jpg'
        )
    else:
        return jsonify({"error": "Failed to retrieve image"}), 500

if __name__ == '__main__':
    app.run(debug=True)
 ## Geocoding API (Convert Address to Coordinates)

    [
    {
        "formatted_address": "Sandton, South Africa",
        "geometry": {
            "location": {
                "lat": -26.1076,
                "lng": 28.0567
            }
        }
    }
    ]
## Updated Backend Code

import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# ... (other imports and routes from previous backend)

# Detect parking spaces using basic image processing
def detect_parking_spaces(image_data):
    # Convert the raw image data to OpenCV format
    image = np.array(Image.open(BytesIO(image_data)))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to isolate bright areas (parking spaces)
    _, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours to detect parking regions
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise)
    parking_contours = [c for c in contours if cv2.contourArea(c) > 500]  # Minimum area threshold

    return len(parking_contours), parking_contours

# New API endpoint: Analyze parking spaces
@app.route('/analyze_parking', methods=['POST'])
def analyze_parking():
    data = request.json
    location = data.get("location")
    if not location:
        return jsonify({"error": "Location is required"}), 400

    # Fetch Street View image
    image_data = get_street_view_image(location) # type: ignore
    if not image_data:
        return jsonify({"error": "Failed to fetch image"}), 500

    # Detect parking spaces
    num_spaces, _ = detect_parking_spaces(image_data)

    return jsonify({"parking_spaces": num_spaces})

    
## Example Response:

{
    "parking_spaces": 7
}
## Code Update Annotate Parking Spaces

def detect_parking_spaces(image_data):
    image = np.array(Image.open(BytesIO(image_data)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parking_contours = [c for c in contours if cv2.contourArea(c) > 500]

    # Annotate the image
    annotated_image = image.copy()
    for contour in parking_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return len(parking_contours), parking_contours, annotated_image
## Serve Annotated Image in API

@app.route('/annotated_parking', methods=['POST'])
def annotated_parking():
    data = request.json
    location = data.get("location")
    if not location:
        return jsonify({"error": "Location is required"}), 400

    image_data = get_street_view_image(location) # type: ignore
    if not image_data:
        return jsonify({"error": "Failed to fetch image"}), 500

    _, _, annotated_image = detect_parking_spaces(image_data)

    # Convert annotated image to bytes
    _, buffer = cv2.imencode('.jpg', annotated_image)
    return send_file(
        BytesIO(buffer),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name='annotated_parking.jpg'
    )
## Loading and Using Pre-Trained YOLOv5 Model

import torch
from PIL import Image
import numpy as np
import cv2

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5 small model

# Function to detect parking spaces
def detect_parking_yolo(image_data):
    image = Image.open(BytesIO(image_data))  # Load image from bytes
    results = model(image)  # Perform detection

    # Extract results
    detections = results.xyxy[0].numpy()  # Bounding box, confidence, class
    annotated_image = np.array(image)
    
    for x1, y1, x2, y2, conf, cls in detections:
        if int(cls) == 2:  # Assume "car" class is ID 2 (adjust based on dataset)
            # Draw bounding boxes
            cv2.rectangle(
                annotated_image, 
                (int(x1), int(y1)), 
                (int(x2), int(y2)), 
                (0, 255, 0), 2
            )
    
    return annotated_image, detections

# Flask route to process parking detection with YOLO
@app.route('/yolo_parking', methods=['POST'])
def yolo_parking():
    data = request.json
    location = data.get("location")
    if not location:
        return jsonify({"error": "Location is required"}), 400

    # Fetch image
    image_data = get_street_view_image(location)
    if not image_data:
        return jsonify({"error": "Failed to fetch image"}), 500

    # Detect parking using YOLO
    annotated_image, detections = detect_parking_yolo(image_data)

    # Encode annotated image for response
    _, buffer = cv2.imencode('.jpg', annotated_image)
    return send_file(
        BytesIO(buffer),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name='yolo_parking.jpg'
    )

## Fetch Real-Time location 
import googlemaps

# Replace with your Google Maps API key
API_KEY = 'YAIzaSyBfuOCG39QMpX-zEo7dzhSHKRH3kHzuP0E'

def get_real_time_location():
    # Initialize Google Maps client
    gmaps = googlemaps.Client(key=YAIzaSyBfuOCG39QMpX-zEo7dzhSHKRH3kHzuP0E)
    
    # Simulate a request (this assumes your app runs on a network-enabled device)
    # Replace with an actual 'considerIp' or other location parameters
    location_data = gmaps.geolocate(considerIp="true")
    
    # Extract latitude and longitude
    location = location_data['location']
    latitude = location['lat']
    longitude = location['lng']
    return latitude, longitude

    ## Updating the parking tracker to use real time location: (Code ## Updated with additional safeguards and debugging. )

def get_location(self):
    try:
        # Fetch real-time location
        self.user_location = get_real_time_location()
        if not self.user_location:
            raise ValueError("Location could not be determined.")

        # Debugging: Print the fetched location
        print(f"Fetched Location: {self.user_location}")

        # Find the nearest parking
        self.nearest_parking = self.find_nearest_parking(self.user_location)
        if not self.nearest_parking:
            self.output_label.config(text="No parking locations found.")
            self.map_button.config(state=tk.DISABLED)
            return

        # Debugging: Print the nearest parking details
        print(f"Nearest Parking: {self.nearest_parking}")

        # Unpack the parking details
        parking_name, parking_coords, distance = self.nearest_parking

        # Update the label and enable map button
        self.output_label.config(
            text=f"Nearest Parking: {parking_name}\n"
                 f"Distance: {distance:.2f} km\n"
                 f"Coordinates: {parking_coords}"
        )
        self.map_button.config(state=tk.NORMAL)

    except ValueError as ve:
        messagebox.showerror("Location Error", str(ve))
    except Exception as e:
        # Debugging: Print the error to console
        print(f"Error: {e}")
        messagebox.showerror("Error", f"Failed to get location: {e}")


## Creating the interface with Ktinker:

import tkinter as tk
from tkinter import messagebox
import folium
import webbrowser

# Initialize Tkinter App
class SmartParkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Parking Tracker")
        self.root.geometry("400x600")
        self.root.configure(bg="#e8f1ff")  # Light blue background

        self.setup_home_screen()

    def setup_home_screen(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="Smart Parking Tracker",
            font=("Arial", 18, "bold"),
            bg="#e8f1ff",
            fg="#002f5f"
        )
        title_label.pack(pady=20)

        # Description
        desc_label = tk.Label(
            self.root,
            text="Find parking near you instantly.",
            font=("Arial", 12),
            bg="#e8f1ff",
            fg="#003f7f"
        )
        desc_label.pack(pady=10)

        # Buttons
        detect_button = tk.Button(
            self.root,
            text="Detect My Location",
            font=("Arial", 12),
            bg="#007acc",
            fg="white",
            relief="flat",
            command=self.detect_location
        )
        detect_button.pack(pady=15, ipadx=10, ipady=5)

        view_parking_button = tk.Button(
            self.root,
            text="View Nearby Parking",
            font=("Arial", 12),
            bg="#007acc",
            fg="white",
            relief="flat",
            command=self.view_parking
        )
        view_parking_button.pack(pady=10, ipadx=10, ipady=5)

        # Map Preview Placeholder
        map_preview_frame = tk.Frame(self.root, bg="#d9ebff", height=300, width=300)
        map_preview_frame.pack(pady=20)
        map_preview_frame.pack_propagate(0)  # Prevent resizing

        map_label = tk.Label(
            map_preview_frame,
            text="Map Preview\n[Mock]",
            bg="#d9ebff",
            font=("Arial", 10, "italic"),
            fg="#0055a5"
        )
        map_label.pack(expand=True)

    def detect_location(self):
        # Simulated Location Detection
        messagebox.showinfo("Detect Location", "Simulated: Current location detected!")
        # In production, integrate GPS or API here.

    def view_parking(self):
        # Generate Map with Parking Spots
        self.create_map()
        webbrowser.open("parking_map.html")

    def create_map(self):
        # Sample Map with Markers
        m = folium.Map(location=[-26.2041, 28.0473], zoom_start=15)
        parking_spots = [
            ("Parking Lot A", -26.2041, 28.0473),
            ("Parking Lot B", -26.1929, 28.0395),
            ("Parking Lot C", -26.2023, 28.0347)
        ]
        for name, lat, lon in parking_spots:
            folium.Marker([lat, lon], popup=name, icon=folium.Icon(color="green")).add_to(m)

        # Save Map
        m.save("parking_map.html")


# Run the Application
if __name__ == "__main__":
    root = tk.Tk()
    app = SmartParkingApp(root)
    root.mainloop()

    ## 1.A Code to get location

    import requests

def get_real_time_location():
    api_key = "YAIzaSyBfuOCG39QMpX-zEo7dzhSHKRH3kHzuP0E"
    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={api_key}"
    try:
        response = requests.post(url)
        if response.status_code == 200:
            data = response.json()
            lat = data["location"]["lat"]
            lng = data["location"]["lng"]
            return lat, lng
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error fetching location: {e}")
        return None

    ## 1.B. Using Geopy for Testing (Local Simulation) for simulation on a local computer:

    from geopy.geocoders import Nominatim

def get_real_time_location():
    geolocator = Nominatim(user_agent="parking_tracker")
    location = geolocator.geocode("Johannesburg, South Africa")
    if location:
        return location.latitude, location.longitude
    return None

    ## Mobile App Code Example with Kivy

    from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.mapview import MapView, MapMarker

class ParkingApp(App):
    def build(self):
        # Main Layout
        layout = BoxLayout(orientation='vertical', padding=10)

        # Title
        title_label = Label(
            text="Smart Parking Tracker",
            font_size=24,
            size_hint=(1, 0.1)
        )
        layout.add_widget(title_label)

        # MapView to Show Locations
        self.map_view = MapView(zoom=15, lat=-26.2041, lon=28.0473)  # Johannesburg
        layout.add_widget(self.map_view)

        # Fetch Location Button
        detect_button = Button(
            text="Detect My Location",
            size_hint=(1, 0.1),
            background_color=(0, 0.5, 1, 1)
        )
        detect_button.bind(on_press=self.detect_location)
        layout.add_widget(detect_button)

        # Parking Spots Button
        parking_button = Button(
            text="Find Parking Nearby",
            size_hint=(1, 0.1),
            background_color=(0, 0.8, 0, 1)
        )
        parking_button.bind(on_press=self.find_parking)
        layout.add_widget(parking_button)

        return layout

    def detect_location(self, instance):
        # Simulate Real-Time Location Detection
        lat, lon = -26.2041, 28.0473
        self.map_view.center_on(lat, lon)
        self.map_view.add_marker(MapMarker(lat=lat, lon=lon))

    def find_parking(self, instance):
        # Mock Nearby Parking Spots
        parking_spots = [
            (-26.2023, 28.0347),  # Spot 1
            (-26.1929, 28.0395),  # Spot 2
            (-26.2048, 28.0500)   # Spot 3
        ]
        for lat, lon in parking_spots:
            self.map_view.add_marker(MapMarker(lat=lat, lon=lon))

# Run the App
if __name__ == "__main__":
    ParkingApp().run()

# an updated version of the app using plyer for GPS functionality:

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.mapview import MapView, MapMarker

class ParkingApp(App):
    def build(self):
        # Main Layout
        layout = BoxLayout(orientation='vertical', padding=10)

        # Title
        title_label = Label(
            text="Smart Parking Tracker",
            font_size=24,
            size_hint=(1, 0.1)
        )
        layout.add_widget(title_label)

        # MapView to Show Locations
        self.map_view = MapView(zoom=15, lat=-26.2041, lon=28.0473)  # Johannesburg
        layout.add_widget(self.map_view)

        # Fetch Location Button
        detect_button = Button(
            text="Detect My Location",
            size_hint=(1, 0.1),
            background_color=(0, 0.5, 1, 1)
        )
        detect_button.bind(on_press=self.detect_location)
        layout.add_widget(detect_button)

        # Parking Spots Button
        parking_button = Button(
            text="Find Parking Nearby",
            size_hint=(1, 0.1),
            background_color=(0, 0.8, 0, 1)
        )
        parking_button.bind(on_press=self.find_parking)
        layout.add_widget(parking_button)

        return layout

    def detect_location(self, instance):
        # Simulate Real-Time Location Detection
        lat, lon = -26.2041, 28.0473
        self.map_view.center_on(lat, lon)
        self.map_view.add_marker(MapMarker(lat=lat, lon=lon))

    def find_parking(self, instance):
        # Mock Nearby Parking Spots
        parking_spots = [
            (-26.2023, 28.0347),  # Spot 1
            (-26.1929, 28.0395),  # Spot 2
            (-26.2048, 28.0500)   # Spot 3
        ]
        for lat, lon in parking_spots:
            self.map_view.add_marker(MapMarker(lat=lat, lon=lon))

# Run the App
if __name__ == "__main__":
    ParkingApp().run()

    # Deploying to Android with Buildozer  -- see buildozer spec. 

    # Add Real-Time GPS Integration:

    from plyer import gps

gps.configure(on_location=self.on_location)
gps.start()

def on_location(self, **kwargs):
    print(f"Location: {kwargs}")