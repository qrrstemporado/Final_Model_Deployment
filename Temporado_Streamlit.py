import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    return model

def import_and_predict(image_data, model):
    size = (28, 28)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = image.convert('L')  # Convert image to grayscale
    img = np.asarray(image)
    img = img.reshape(1, 28, 28, 1)  # Reshape to match the model's input shape
    img = img / 255.0  # Normalize pixel values to range [0, 1]
    prediction = model.predict(img)
    return prediction

def about():
    st.sidebar.write("""
    This is a simple Fashion Mnist Classifier app that predicts the category of a clothing item based on an uploaded image.It uses a pre-trained deep learning model to make the predictions, and the model that is being used by the app is Convolutional Neural Network (CNN) from Hands-On Activity 9, which produces an accuracy of 85% in classifying a clothing item.
    You can see the whole simulation and source code through this link: https://colab.research.google.com/drive/1rTMyjcOSjNeI5mLKsTmZq_aohfLjz0sZ?usp=share_link""")

def author():
    st.sidebar.write("""
    This app was created by Roland Ross Temporado, a 3rd year CPE student from the Technological Institute of the Philippines, Quezon City.
    """)

def conclusion():
    st.sidebar.write("""
    Thank you for using this app! If you have any feedback or suggestions, feel free to reach out.
    """)

def main():
    model = load_model()

    st.sidebar.title("Menu")
    menu_selection = st.sidebar.selectbox("Select Option", ["About", "Author", "Conclusion"])

    if menu_selection == "About":
        about()
    elif menu_selection == "Author":
        author()
    elif menu_selection == "Conclusion":
        conclusion()

    st.write("""
    # Fashion Mnist Classifier
    Upload an image of a clothing item to classify its category.
    """)

    file = st.file_uploader("Choose an image file (jpg/png)", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        with st.spinner("Classifying..."):
            prediction = import_and_predict(image, model)

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        class_index = np.argmax(prediction)
        class_name = class_names[class_index]

        st.success(f"Predicted Class: {class_name} (Confidence: {prediction[0][class_index]*100:.2f}%)")
        # Auto-scroll to the bottom of the page
        scroll_js = """
        <script>
            window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
        </script>
        """
        st.write(scroll_js, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
