# Streamlit web interface for the cat vs dog classifier

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server deployment
import matplotlib.pyplot as plt

# Import from src folder
sys.path.append(str(Path(__file__).parent / "src"))

from model_utils import CatDogClassifier, get_prediction_emoji, format_confidence
from visualization import ModelVisualizer, create_confidence_gauge

# Basic page setup
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        color: #333;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .metric-card h4 {
        color: #666;
        font-weight: 600;
    }
    .metric-card h2 {
        color: #333;
        font-weight: bold;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Cache model loading for performance"""
    model_path = Path(__file__).parent / "cat_dog_cnn_model.h5"
    if not model_path.exists():
        st.error(f"❌ Model file not found at {model_path}")
        st.info("Please ensure 'cat_dog_cnn_model.h5' is in the project root directory")
        return None, None
    
    classifier = CatDogClassifier(str(model_path))
    visualizer = ModelVisualizer(classifier.model)
    return classifier, visualizer


def create_confidence_chart(cat_prob, dog_prob):
    """Bar chart showing cat vs dog probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Cat 🐱', 'Dog 🐕'],
            y=[cat_prob, dog_prob],
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f'{cat_prob:.1%}', f'{dog_prob:.1%}'],
            textposition='auto',
            hovertemplate='%{x}: %{y:.2%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        showlegend=False,
        height=400,
        template="plotly_white",
        hoverlabel=dict(bgcolor="white", font_size=16)
    )
    
    return fig


def create_gauge_chart(confidence, predicted_class):
    """Gauge chart for prediction confidence"""
    color = "#28a745" if confidence > 0.8 else "#ffc107" if confidence > 0.6 else "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        title={'text': f"Confidence: {predicted_class}"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': '#ffebee'},
                {'range': [60, 80], 'color': '#fff3e0'},
                {'range': [80, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def process_image(image, classifier):
    """Handle image upload and get prediction"""
    # Handle PNG transparency - convert to RGB
    if image.mode in ('RGBA', 'LA', 'P'):
        # White background for transparent images
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        # Paste with alpha mask
        if image.mode == 'RGBA':
            rgb_image.paste(image, mask=image.split()[3])  # Alpha channel mask
        else:
            rgb_image.paste(image)
        image = rgb_image
    
    # Save temp file for processing
    temp_path = "temp_upload.jpg"
    image.save(temp_path, 'JPEG')
    
    # Get model prediction
    prediction = classifier.predict(temp_path)
    
    # Clean up temp file
    os.remove(temp_path)
    
    return prediction


def main():
    # App header
    st.markdown("""
        <h1 style='text-align: center; color: #333; font-size: 3rem;'>
            🐱 Cat vs Dog Classifier 🐶
        </h1>
        <p style='text-align: center; color: #666; font-size: 1.2rem;'>
            Powered by Deep Learning CNN • 88.7% Accuracy
        </p>
    """, unsafe_allow_html=True)
    
    # Initialize classifier
    classifier, visualizer = load_model()
    
    if classifier is None:
        return
    
    # Navigation sidebar
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        page = st.radio("Choose a page:", 
                       ["🏠 Home", "📊 Model Info", "🔬 Visualizations", "📈 Batch Analysis"])
        
        st.markdown("---")
        st.markdown("## 📋 About")
        st.info("""
        This app uses a custom CNN model trained on 25,000 images from the 
        Dogs vs Cats dataset. The model achieves 88.7% validation accuracy.
        """)
        
        st.markdown("## 🛠️ Features")
        st.markdown("""
        - ✅ Real-time prediction
        - ✅ Confidence visualization
        - ✅ Feature map analysis
        - ✅ Batch processing
        - ✅ Model interpretability
        """)
    
    # Route to selected page
    if page == "🏠 Home":
        show_home_page(classifier)
    elif page == "📊 Model Info":
        show_model_info_page(classifier)
    elif page == "🔬 Visualizations":
        show_visualization_page(classifier, visualizer)
    elif page == "📈 Batch Analysis":
        show_batch_analysis_page(classifier)


def show_home_page(classifier):
    """Main upload and prediction interface"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose a cat or dog image...",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Show image metadata
            st.markdown("#### 📷 Image Details")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("Size", f"{image.size[0]}x{image.size[1]}")
            with info_col2:
                st.metric("Mode", image.mode)
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### 🔮 Prediction Results")
            
            with st.spinner("🧠 Analyzing image..."):
                prediction = process_image(image, classifier)
            
            # Show prediction results
            emoji = get_prediction_emoji(prediction['class'])
            
            st.markdown(f"""
                <div class="prediction-box">
                    <h1 style='font-size: 4rem; margin: 0;'>{emoji}</h1>
                    <h2 style='margin: 0.5rem 0;'>It's a {prediction['class']}!</h2>
                    <p style='font-size: 1.5rem; margin: 0;'>
                        Confidence: {prediction['confidence']:.1%}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Confidence Analysis")
            
            # Probability breakdown
            fig = create_confidence_chart(
                prediction['probability_cat'],
                prediction['probability_dog']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence gauge
            gauge_fig = create_gauge_chart(
                prediction['confidence'],
                prediction['class']
            )
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Interpret confidence level
            confidence_level = (
                "Very High" if prediction['confidence'] > 0.9 else
                "High" if prediction['confidence'] > 0.75 else
                "Medium" if prediction['confidence'] > 0.6 else
                "Low"
            )
            
            st.info(f"""
            **Confidence Level:** {confidence_level}
            
            The model is {prediction['confidence']:.1%} confident that this image 
            contains a {prediction['class'].lower()}.
            """)
        else:
            # Center the placeholder message
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("""
                <div style='text-align: center; padding: 3rem 1rem; background-color: #f0f2f6; border-radius: 15px; border: 2px dashed #4CAF50;'>
                    <h2 style='color: #4CAF50; margin-bottom: 1rem;'>📸 Ready to Classify!</h2>
                    <p style='color: #666; font-size: 1.2rem;'>👈 Upload an image on the left to see if it's a cat or dog</p>
                    <p style='color: #888; font-size: 0.9rem; margin-top: 1rem;'>Supported formats: JPG, JPEG, PNG</p>
                </div>
            """, unsafe_allow_html=True)


def show_model_info_page(classifier):
    """Model architecture and performance stats"""
    st.markdown("## 📊 Model Information")
    
    model_info = classifier.get_model_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card" style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h4 style="color: #333; margin: 0;">Total Parameters</h4>
                <h2 style="color: #4CAF50; margin: 0.5rem 0; font-size: 2rem;">{:,}</h2>
            </div>
        """.format(model_info['total_parameters']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card" style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h4 style="color: #333; margin: 0;">Model Layers</h4>
                <h2 style="color: #2196F3; margin: 0.5rem 0; font-size: 2rem;">{}</h2>
            </div>
        """.format(model_info['num_layers']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card" style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h4 style="color: #333; margin: 0;">Input Shape</h4>
                <h2 style="color: #FF9800; margin: 0.5rem 0; font-size: 2rem;">150×150×3</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card" style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h4 style="color: #333; margin: 0;">Accuracy</h4>
                <h2 style="color: #9C27B0; margin: 0.5rem 0; font-size: 2rem;">88.7%</h2>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🏗️ Architecture Details")
    
    architecture_data = {
        'Layer': ['Conv2D-1', 'MaxPool-1', 'Conv2D-2', 'MaxPool-2', 
                 'Conv2D-3', 'MaxPool-3', 'Conv2D-4', 'MaxPool-4',
                 'Flatten', 'Dropout', 'Dense-1', 'Dense-2'],
        'Output Shape': ['148×148×32', '74×74×32', '72×72×64', '36×36×64',
                        '34×34×128', '17×17×128', '15×15×128', '7×7×128',
                        '6272', '6272', '512', '1'],
        'Parameters': [896, 0, 18496, 0, 73856, 0, 147584, 0, 0, 0, 3211776, 513]
    }
    
    df = pd.DataFrame(architecture_data)
    st.dataframe(df, use_container_width=True)
    
    # Training and validation metrics
    st.markdown("### 📈 Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Performance")
        training_metrics = {
            'Metric': ['Final Training Accuracy', 'Final Training Loss', 'Epochs', 'Batch Size'],
            'Value': ['86.9%', '0.3015', '15', '32']
        }
        st.table(pd.DataFrame(training_metrics))
    
    with col2:
        st.markdown("#### Validation Performance")
        val_metrics = {
            'Metric': ['Final Validation Accuracy', 'Final Validation Loss', 'Best Epoch', 'Data Split'],
            'Value': ['88.7%', '0.2648', '14', '80/20']
        }
        st.table(pd.DataFrame(val_metrics))


def show_visualization_page(classifier, visualizer):
    """Feature map visualization page"""
    st.markdown("## 🔬 Model Visualizations")
    
    st.info("Upload an image to visualize what the model 'sees' at different layers!")
    
    uploaded_file = st.file_uploader(
        "Choose an image for visualization...",
        type=['jpg', 'jpeg', 'png'],
        key="viz_uploader"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Handle transparency
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                rgb_image.paste(image, mask=image.split()[3])
            else:
                rgb_image.paste(image)
            image = rgb_image
        
        temp_path = "temp_viz.jpg"
        image.save(temp_path, 'JPEG')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)
            
            # Show what the model predicted
            prediction = classifier.predict(temp_path)
            st.success(f"Predicted: **{prediction['class']}** ({prediction['confidence']:.1%})")
        
        with col2:
            st.markdown("### 🧠 Feature Maps")
            
            # Find conv layers for visualization
            available_layers = visualizer.get_available_layers()
            conv_layer_options = list(available_layers['name_mapping'].keys())
            conv_layer_options = [name for name in conv_layer_options if name.startswith('conv')]
            
            layer_option = st.selectbox(
                "Select a layer to visualize:",
                conv_layer_options if conv_layer_options else ['conv1', 'conv2', 'conv3', 'conv4']
            )
            
            with st.spinner("Generating visualization..."):
                try:
                    # Get feature maps for the selected layer
                    feature_maps = visualizer.get_feature_maps(temp_path, [layer_option])
                    
                    if layer_option in feature_maps:
                        activation = feature_maps[layer_option]['activation']
                        
                        # Display feature map grid
                        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
                        
                        for i in range(min(16, activation.shape[-1])):
                            ax = axes[i // 4, i % 4]
                            ax.imshow(activation[0, :, :, i], cmap='viridis')
                            ax.axis('off')
                            ax.set_title(f'Filter {i+1}')
                        
                        plt.suptitle(f'Feature Maps for {layer_option}', fontsize=14, fontweight='bold')
                        st.pyplot(fig)
                        
                        # Show layer details
                        layer_info = feature_maps[layer_option]
                        
                        st.info(f"""
                        **Layer:** {layer_option} (actual: {layer_info.get('actual_layer_name', layer_option)})  
                        **Output Shape:** {activation.shape[1:]}  
                        **Total Filters:** {activation.shape[-1]}  
                        **Layer Type:** {layer_info.get('layer_type', 'Unknown')}
                        """)
                    else:
                        st.warning(f"Could not generate feature maps for {layer_option}. Try another layer.")
                except Exception as e:
                    st.error(f"Visualization error: {str(e)}")
                    
                    # Debug layer mapping issues
                    with st.expander("🔧 Debug Information"):
                        available_layers = visualizer.get_available_layers()
                        st.write("**Available layer mapping:**")
                        st.json(available_layers['name_mapping'])
                        st.write("**All model layers:**")
                        for layer_info in available_layers['all_layers'][:10]:  # Show first 10
                            st.write(f"- {layer_info['name']} ({layer_info['type']})")
                    
                    st.info("💡 Try selecting a different layer from the dropdown.")
        
        # Remove temp file
        os.remove(temp_path)
    else:
        st.warning("Please upload an image to see visualizations.")


def show_batch_analysis_page(classifier):
    """Multi-image processing interface"""
    st.markdown("## 📈 Batch Analysis")
    
    st.markdown("""
    Upload multiple images to analyze them all at once! 
    Perfect for testing your model on a dataset.
    """)
    
    uploaded_files = st.file_uploader(
        "Choose multiple images...",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        if st.button("🚀 Analyze All Images", type="primary"):
            results = []
            progress_bar = st.progress(0)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                
                # Handle transparency
                if image.mode in ('RGBA', 'LA', 'P'):
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'RGBA':
                        rgb_image.paste(image, mask=image.split()[3])
                    else:
                        rgb_image.paste(image)
                    image = rgb_image
                
                temp_path = f"temp_batch_{idx}.jpg"
                image.save(temp_path, 'JPEG')
                
                prediction = classifier.predict(temp_path)
                prediction['filename'] = uploaded_file.name
                results.append(prediction)
                
                os.remove(temp_path)
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # Show batch processing results
            st.markdown("### 📊 Results Summary")
            
            # Summary stats
            cat_count = sum(1 for r in results if r['class'] == 'Cat')
            dog_count = len(results) - cat_count
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cats", f"{cat_count} 🐱")
            with col2:
                st.metric("Total Dogs", f"{dog_count} 🐕")
            with col3:
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Results table
            st.markdown("### 📋 Detailed Results")
            
            df_results = pd.DataFrame(results)
            df_results['confidence'] = df_results['confidence'].apply(lambda x: f"{x:.1%}")
            df_results = df_results[['filename', 'class', 'confidence']]
            
            st.dataframe(df_results, use_container_width=True)
            
            # Class distribution pie chart
            st.markdown("### 📊 Class Distribution")
            
            fig = px.pie(
                values=[cat_count, dog_count],
                names=['Cat', 'Dog'],
                color_discrete_map={'Cat': '#FF6B6B', 'Dog': '#4ECDC4'}
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Upload multiple images to perform batch analysis")


if __name__ == "__main__":
    main()