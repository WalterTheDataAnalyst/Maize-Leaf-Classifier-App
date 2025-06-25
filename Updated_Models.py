import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
import cv2
import os
import transformers
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# Set page config
st.set_page_config(
    page_title="Corn Leaf Disease Classifier",
    page_icon="indaba.png",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #000000;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title {
        color: #2e8b57;
        text-align: center;
    }
    .model-card {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        padding: 15px;
        margin-bottom: 20px;
        background-color: white;
    }
    .prediction {
        font-size: 1.2em;
        font-weight: bold;
        color: #2e8b57;
    }
</style>
""", unsafe_allow_html=True)

# Load your logo/image
logo = Image.open("indaba.png")  # Replace with your image path
# Create a centered layout using columns
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust ratios for spacing

# Class names
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Load models (cache them to avoid reloading on every interaction)
@st.cache_resource
def load_custom_model():
    return load_model('model2.h5')

@st.cache_resource
def load_resnet_model():
    base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

@st.cache_resource
def load_vgg_model():
    base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

@st.cache_resource
def load_swin_model():
    # Load the Swin Transformer model
    model_path = "final_swin_model"  # Update this path
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    return processor, model

# Preprocess image function
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Preprocess image for Swin Transformer
def preprocess_swin_image(image, processor):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    inputs = processor(images=image, return_tensors="pt")
    return inputs

# LIME explanation function
def lime_explanation(model, image, top_labels=5, hide_color=0, num_samples=1000):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image[0].astype('double'), 
        model.predict, 
        top_labels=top_labels, 
        hide_color=hide_color, 
        num_samples=num_samples
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    return mark_boundaries(temp / 2 + 0.5, mask)

# SHAP explanation function
def shap_explanation(model, image):
    def f(x):
        tmp = x.copy()
        return model(tmp)
    
    masker = shap.maskers.Image("blur(224,224)", shape=(224, 224, 3))
    explainer = shap.Explainer(f, masker)
    shap_values = explainer(
        image[np.newaxis, :, :, :], 
        max_evals=5000, 
        outputs=shap.Explanation.argsort.flip[:4]
    )
    return shap_values

# Main app
def main():
    st.image(logo, width = 150)
    st.title("🌽 Corn Leaf Disease Classifier")
    st.markdown("Upload an image of a corn leaf to classify its disease status using multiple deep learning models.")
    
    # Sidebar
    st.sidebar.title("Options")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    show_lime = st.sidebar.checkbox("Show LIME Explanation", value=True)
    show_shap = st.sidebar.checkbox("Show SHAP Explanation", value=True)
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Load models
        custom_model = load_custom_model()
        resnet_model = load_resnet_model()
        vgg_model = load_vgg_model()
        swin_processor, swin_model = load_swin_model()
        
        # Make predictions
        custom_pred = custom_model.predict(processed_image)[0]
        resnet_pred = resnet_model.predict(processed_image)[0]
        vgg_pred = vgg_model.predict(processed_image)[0]
        
        # Swin Transformer prediction
        swin_inputs = preprocess_swin_image(image, swin_processor)
        with torch.no_grad():
            swin_outputs = swin_model(**swin_inputs)
        swin_probs = torch.nn.functional.softmax(swin_outputs.logits, dim=-1)[0]
        swin_pred = swin_probs.numpy()
        
        # Create a DataFrame for predictions
        pred_data = {
            'Model': ['Custom CNN', 'ResNet50', 'VGG16', 'Swin Transformer'],
            'Prediction': [
                CLASS_NAMES[np.argmax(custom_pred)],
                CLASS_NAMES[np.argmax(resnet_pred)],
                CLASS_NAMES[np.argmax(vgg_pred)],
                CLASS_NAMES[np.argmax(swin_pred)]
            ],
            'Confidence': [
                f"{np.max(custom_pred)*100:.2f}%",
                f"{np.max(resnet_pred)*100:.2f}%",
                f"{np.max(vgg_pred)*100:.2f}%",
                f"{np.max(swin_pred)*100:.2f}%"
            ]
        }
        
        # Display predictions
        st.subheader("Model Predictions")
        pred_df = pd.DataFrame(pred_data)
        st.table(pred_df)
        
        # Visualize predictions
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))  # Added one more subplot for Swin
        
        # Custom CNN predictions
        axes[0].bar(CLASS_NAMES, custom_pred)
        axes[0].set_title('Custom CNN Predictions')
        axes[0].set_ylabel('Confidence')
        axes[0].tick_params(axis='x', rotation=45)
        
        # ResNet predictions
        axes[1].bar(CLASS_NAMES, resnet_pred)
        axes[1].set_title('ResNet50 Predictions')
        axes[1].tick_params(axis='x', rotation=45)
        
        # VGG predictions
        axes[2].bar(CLASS_NAMES, vgg_pred)
        axes[2].set_title('VGG16 Predictions')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Swin Transformer predictions
        axes[3].bar(CLASS_NAMES, swin_pred)
        axes[3].set_title('Swin Transformer Predictions')
        axes[3].tick_params(axis='x', rotation=45)
        
        st.pyplot(fig)
        
        # Model explanations
        st.subheader("Model Explanations")
        
        if show_lime:
            st.markdown("#### LIME Explanations")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**Custom CNN**")
                lime_img = lime_explanation(custom_model, processed_image)
                st.image(lime_img, use_column_width=True)
            
            with col2:
                st.markdown("**ResNet50**")
                lime_img = lime_explanation(resnet_model, processed_image)
                st.image(lime_img, use_column_width=True)
            
            with col3:
                st.markdown("**VGG16**")
                lime_img = lime_explanation(vgg_model, processed_image)
                st.image(lime_img, use_column_width=True)
            
            # Note: LIME for Swin would require additional implementation
            
        if show_shap:
            st.markdown("#### SHAP Explanations")
            st.warning("SHAP explanations may take a few minutes to generate...")
            
            try:
                # Use Custom CNN for SHAP (others would take too long)
                shap_values = shap_explanation(custom_model, processed_image[0])
                fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
                shap.image_plot(shap_values.values, processed_image)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating SHAP explanation: {str(e)}")
                st.info("Try with a smaller image or disable SHAP explanations.")
    
    else:
        st.info("Please upload an image using the sidebar to get started.")
        
        # Sample images
        st.subheader("Sample Images")
        col1, col2, col3, col4 = st.columns(4)
        
        sample_images = {
            "Blight": "sample_blight.jpg",
            "Common Rust": "sample_rust.jpg",
            "Gray Leaf Spot": "sample_spot.jpg",
            "Healthy": "sample_healthy.jpg"
        }
        
        for name, img_path in sample_images.items():
            try:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    if name == "Blight":
                        with col1:
                            st.image(img, caption=name, use_column_width=True)
                    elif name == "Common Rust":
                        with col2:
                            st.image(img, caption=name, use_column_width=True)
                    elif name == "Gray Leaf Spot":
                        with col3:
                            st.image(img, caption=name, use_column_width=True)
                    else:
                        with col4:
                            st.image(img, caption=name, use_column_width=True)
            except:
                pass

if __name__ == "__main__":
    main()