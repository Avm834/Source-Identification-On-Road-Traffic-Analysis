import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import folium
import random
import base64

# Function to encode an image to base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background image of the app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:static/images/background.jpg;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('static/images/background.jpg')

# Function to draw bounding boxes and crop images
def draw_bounding_boxes_with_labels_and_coords(image, labels_path):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 16)

    show_map = False
    cropped_images = []

    with open(labels_path, 'r') as f:
        for line in f:
            class_name, x_center, y_center, width, height = line.strip().split(' ')
            # Convert normalized coordinates to pixel values
            image_width, image_height = image.size
            x_min = int(float(x_center) * image_width - float(width) * image_width / 2)
            y_min = int(float(y_center) * image_height - float(height) * image_height / 2)
            x_max = int(float(x_center) * image_width + float(width) * image_width / 2)
            y_max = int(float(y_center) * image_height + float(height) * image_height / 2)

            # Draw the bounding box
            draw.rectangle((x_min, y_min, x_max, y_max), outline='red', width=2)

            label_text = "Accident" if class_name == "1" else "Non-accident"
            if label_text == 'Accident':
                st.markdown(f"<p style='color:red;'>Predicted class is {label_text} </p>", unsafe_allow_html=True)
                show_map = True
            elif label_text == 'Non-accident':
                st.markdown(f"<p style='color:green;'>Predicted class is {label_text} </p>", unsafe_allow_html=True)

            # Draw the coordinates
            coord_text = f"({x_min}, {y_min}), ({x_max}, {y_max})"
            draw.text((x_min, y_min - 20), label_text, fill="red", font=font)

            # Crop the detected region for accidents
            if label_text == 'Accident':
                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                cropped_images.append(cropped_image)

    return image, show_map, cropped_images

# Sidebar configuration
st.sidebar.title("Content")
tabs = st.sidebar.radio("Select a tab:", ["Abstract", "Future Scope", "Prediction"])

if tabs == "Abstract":
    title = '<b><center><p style="font-family:Times new roman; color:White; font-size: 40px;">ACCIDENT DETECTION AND SEGMENTATION</p></center></b>'
    st.markdown(title, unsafe_allow_html=True)
    original_title = '<b><center><p style="font-family:Times new roman; color:White; font-size: 28px;">ABSTRACT</p></center></b>'
    st.markdown(original_title, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .abstract-text {
        font-family: 'Times New Roman', serif;
        font-size: 20px;
        color: #FFFFFF;
        text-align:justify;
    }
    </style>
    <p class="abstract-text">
    In recent years, the application of machine learning in the field of traffic management has gained significant attention, particularly for automatic accident detection, segmentation, and duration prediction. This study explores a comprehensive approach that leverages advanced machine learning algorithms to enhance road safety and traffic flow efficiency. The proposed system integrates multiple stages: accident detection, precise segmentation of accident scenes, and accurate prediction of accident duration. Using real-time traffic data and sensor inputs, the system employs deep learning models to identify anomalies and potential accidents. Techniques such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs) are utilized to process and analyze vast amounts of data, ensuring prompt and reliable detection. Once an accident is detected, the system performs segmentation to isolate and identify the specific regions of interest within the scene. This involves using advanced image processing techniques and neural networks to delineate the affected areas, which aids in efficient resource allocation and emergency response. The final stage involves predicting the duration of the accident and its impact on traffic. Machine learning models are trained on historical data, considering various factors such as accident severity, time of day, and weather conditions. This prediction capability is crucial for traffic management centers to devise optimal rerouting strategies and minimize congestion.
    </p>
    """, unsafe_allow_html=True)
    
elif tabs == "Future Scope":
    title = '<b><center><p style="font-family:Times new roman; color:White; font-size: 40px;">FUTURE SCOPE</p></center></b>'
    st.markdown(title, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .future-scope-text {
        font-family: 'Times New Roman', serif;
        font-size: 24px;
        color: #FFFFFF;
    }
    .future-scope-point-1 {
        font-family: 'Times New Roman', serif;
        font-size: 24px;
        color: #FFFFFF;
      
    }
    .future-scope-point-2 {
        font-family: 'Times New Roman', serif;
        color: #FFFFFF;
        font-size: 24px;
    }
    .future-scope-point-3 {
        font-family: 'Times New Roman', serif;
        color: #FFFFFF;
        font-size: 24px;
    }
    .future-scope-point-4 {
        font-family: 'Times New Roman', serif;
        color: #FFFFFF;
        font-size: 24px;
    }
    </style>
    <p class="future-scope-text">
    The future scope of this project includes:
    <ul>
        <li id="point1" class="future-scope-point-1">Enhancing the detection algorithm with more data and advanced models.</li>
        <li id="point2" class="future-scope-point-2">Integrating real-time data processing for immediate accident detection.</li>
        <li id="point3" class="future-scope-point-3">Expanding the system to cover different types of accidents and environments.</li>
        <li id="point4" class="future-scope-point-4">Collaborating with emergency services for improved response times.</li>
    </ul>
    </p>
    """, unsafe_allow_html=True)
elif tabs == "Prediction":
    original_title = '<b><center><p style="font-family:Times new roman; color:White; font-size: 40px;">PREDICTION</p></center></b>'
    st.markdown(original_title, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .content{
        font-family: 'Times New Roman', serif;
        font-size: 20px;
        color: #FFFFFF;
        text-align:justify;
    }
    </style>
    <p class="content">
    Upload an image to detect accidents and view the results. The application will highlight detected accident regions and show their coordinates. Additionally, a map will be displayed showing a random location related to the detected accident.
    </p>""", unsafe_allow_html=True)
    dataset_path = "Data"
    image_file = st.file_uploader("Upload image", type=["jpg", "png"])

    if st.button("Submit"):
        if image_file:
            image_name = image_file.name  # Get the file name of the uploaded image
            found = False
            for folder in ["train", "test", "valid"]:
                image_path = os.path.join(dataset_path, folder, "images", image_name)
                labels_path = os.path.join(dataset_path, folder, "labels", os.path.splitext(image_name)[0] + ".txt")
                if os.path.exists(image_path) and os.path.exists(labels_path):
                    image = Image.open(image_path)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption='Uploaded Image', width=300)  # Show the uploaded image
                    with col2:
                        image_with_boxes, show_map, cropped_images = draw_bounding_boxes_with_labels_and_coords(image, labels_path)
                        st.image(image_with_boxes, caption='Predicted Image', width=300)  # Show the image with bounding boxes
                        if cropped_images:
                            for idx, cropped_image in enumerate(cropped_images):
                                st.image(cropped_image, caption=f'Cropped Accident Region {idx+1}', width=300)  # Show the cropped region
                    if show_map:
                        # Read the CSV file into a pandas DataFrame
                        data = pd.read_csv('states.csv')

                        # Retrieve a single random location
                        random_index = random.randint(0, len(data) - 1)
                        latitude = data['latitude'][random_index]
                        longitude = data['longitude'][random_index]
                        state = data['State.Name'][random_index].lower().capitalize()

                        # Create a map centered around the selected location
                        m = folium.Map(location=[latitude, longitude], zoom_start=10)

                        # Add a marker to the map for the selected location
                        folium.Marker([latitude, longitude]).add_to(m)

                        # Save the map to an HTML file
                        map_filename = f'static/maps/map_{image_name}.html'
                        m.save(map_filename)
                        content = '<b><center><p style="font-family:Times new roman; color:White; font-size: 28px;">Detected accident location</p></center></b>'

                        st.markdown(content, unsafe_allow_html=True)
                        st.components.v1.html(open(map_filename, 'r').read(), height=400)
                        predict = f'<p style="font-family: Times new roman; color: white; font-size: 18px;">Latitude: {latitude}</p><p style="font-family: Times new roman; color: white; font-size: 18px;">Longitude: {longitude}</p><p style="font-family: Times new roman; color: white; font-size: 18px;">State: {state}</p>'
                        st.markdown(predict, unsafe_allow_html=True)
                    found = True
                    break  # Stop searching once the image and labels are found in one of the folders
            if not found:
                st.write("Image or labels file not found.")
        else:
            st.write("Please upload an image.")
