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
    This is a simple Fashion Mnist Classifier app that predicts the category of a clothing item based on an uploaded image.It uses a pre-trained deep learning model to make the predictions, and the model that is being used by the app is Convolutional Neural Network (CNN) from Hands-On Activity 9, which achieved an accuracy of 85% in classifying a clothing item.
    
    You can see the whole simulation and source code through this link: https://colab.research.google.com/drive/1rTMyjcOSjNeI5mLKsTmZq_aohfLjz0sZ?usp=share_link""")

def author():
    st.sidebar.write("""
    This app was created by Roland Ross Temporado, a 3rd year CPE student from the Technological Institute of the Philippines, Quezon City.
    """)

def conclusion():
    st.sidebar.write("""
    In conclusion, this activity talks about uploading your created deep learning model to the cloud using streamlit, which is an open-source app framework in Python that helps create web apps for data science and machine learning in a short time. It is compatible with major Python libraries such as Scikit-Learn, Keras, PyTorch, SymPy (latex), NumPy, Pandas, Matplotlib, etc. That being said, Streamlit is the best platform to deploy your deep learning model. 
    
    With the help of the guide provided by the instructor, the student successfully learned how to deploy deep learning models using Streamlit, wherein it is required first to have your best model, which will serve as the basis of the classification or prediction that will be used by the app. Moreover, the student used the CNN model on Hands-On Activity 9, wherein on the said activity it uses the Fashion MNIST dataset in order to develop a CNN model that solves the problem of image classification of clothing items. 
    
    Creating a CNN model for Fashion MNIST can be applied to inventory management systems and can be used to cater to customers, particularly in finding their preferred clothes or shoes, wherein the said model accumulates an accuracy of 85% on classifying clothing items. Lastly, with the help of a Github repository, the student successfully compiled all the necessary things needed to establish a streamlit deployment.  
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
    Upload an image of a clothing item to classify its category. [T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot]
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
