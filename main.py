import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  #Converts single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])


#Home Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "green-leaves-background.webp"
    st.image(image_path,use_container_width=True)
    st.markdown("""
# Welcome to the **Tomato Disease Recognition System!** 🔍  

Our goal is to help farmers and plant enthusiasts **identify plant diseases quickly and accurately**. Simply upload an image of a plant, and our system will analyze it using cutting-edge technology to detect any potential diseases. Together, let's safeguard crops and promote healthier harvests!  

## **🌱 How It Works**  
1. **Upload an Image** – Navigate to the **Disease Recognition** page and upload a photo of the affected plant.  
2. **Automated Analysis** – Our AI-powered system will process the image using advanced machine learning models.  
3. **Get Results Instantly** – View the disease detection results along with recommendations for possible solutions.  

## **✅ Why Choose Our System?**  
- **High Accuracy** – Built on state-of-the-art deep learning techniques for precise disease detection.  
- **Easy to Use** – A seamless, user-friendly interface for quick and hassle-free navigation.  
- **Fast & Reliable** – Get results within seconds, enabling prompt decision-making.  

## **🚀 Get Started**  
Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of AI-driven plant disease detection!  

## **ℹ️ About Us**  
Learn more about our mission, the technology behind the system, and the team driving this initiative on the **About** page.  

**Developed by Chinmay Kale** . 
""")

#About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### **About the Dataset**  
This dataset has been **enhanced through offline augmentation** based on the original dataset, which is available on a GitHub repository.  

It contains approximately **87,000 RGB images** of both **healthy and diseased crop leaves**, categorized into **38 different classes**. The dataset is split into **80% training** and **20% validation**, maintaining the original directory structure.  

                 """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        with st.spinner("Please Wait"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            #Reading Labels
            class_name = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))