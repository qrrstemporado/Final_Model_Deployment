def main():
    model = load_model()

    st.sidebar.title("Menu")
    menu_options = ["Author", "About", "Conclusion"]
    selected_menu = st.sidebar.selectbox("", menu_options)

    if selected_menu == "Author":
        st.title("Author")
        st.write("This application was created by [Your Name].")

    elif selected_menu == "About":
        st.title("About")
        st.write("""
        This is a Fashion Mnist Classifier. 
        Upload an image of a clothing item to classify its category.
        """)

    elif selected_menu == "Conclusion":
        st.title("Conclusion")
        st.write("Thank you for using this application!")

    else:
        st.title("Fashion Mnist Classifier")
        st.write("Upload an image of a clothing item to classify its category.")

        file = st.file_uploader("Choose an image file (jpg/png)", type=["jpg", "png"])

        if file is None:
            st.text("Please upload an image file")
        else:
            image = Image.open(file)
            st.image(image, use_column_width=True)

            if st.button("Classify"):
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
