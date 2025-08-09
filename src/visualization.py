"""
Visualization utilities for Cat vs Dog Classifier
Handles feature maps, gradients, and other visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from typing import Optional, Tuple
import io
import base64


class ModelVisualizer:
    """Handles various model visualization techniques."""
    
    def __init__(self, model):
        """
        Initialize visualizer with a trained model.
        
        Args:
            model: Trained Keras model
        """
        self.model = model
        self.img_size = (150, 150)
        self.layer_name_mapping = self._create_layer_mapping()
    
    def _create_layer_mapping(self) -> dict:
        """
        Create mapping between common layer names and actual model layer names.
        
        Returns:
            Dictionary mapping expected names to actual layer names
        """
        mapping = {}
        conv_layers = []
        pool_layers = []
        
        # Find all conv and pool layers
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append(layer.name)
            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                pool_layers.append(layer.name)
        
        # Create standard mappings
        for i, layer_name in enumerate(conv_layers):
            mapping[f'conv{i+1}'] = layer_name
        
        for i, layer_name in enumerate(pool_layers):
            mapping[f'pool{i+1}'] = layer_name
        
        return mapping
    
    def get_layer_by_name(self, layer_name: str):
        """
        Get layer by name, checking both direct name and mapped name.
        
        Args:
            layer_name: Layer name to find
            
        Returns:
            Layer object or None if not found
        """
        # First try direct name
        for layer in self.model.layers:
            if layer.name == layer_name:
                return layer
        
        # Then try mapped name
        if layer_name in self.layer_name_mapping:
            actual_name = self.layer_name_mapping[layer_name]
            for layer in self.model.layers:
                if layer.name == actual_name:
                    return layer
        
        return None
    
    def get_available_layers(self) -> dict:
        """
        Get all available layers for visualization.
        
        Returns:
            Dictionary with layer info
        """
        layers_info = {
            'conv_layers': [],
            'pool_layers': [],
            'all_layers': []
        }
        
        for i, layer in enumerate(self.model.layers):
            layer_info = {
                'index': i,
                'name': layer.name,
                'type': type(layer).__name__,
                'output_shape': layer.output_shape if hasattr(layer, 'output_shape') else None
            }
            
            layers_info['all_layers'].append(layer_info)
            
            if isinstance(layer, tf.keras.layers.Conv2D):
                layers_info['conv_layers'].append(layer_info)
            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                layers_info['pool_layers'].append(layer_info)
        
        # Add mapping info
        layers_info['name_mapping'] = self.layer_name_mapping
        
        return layers_info
    
    def get_feature_maps(self, img_path: str, layer_names: list = None) -> dict:
        """
        Extract feature maps from specified layers.
        
        Args:
            img_path: Path to input image
            layer_names: List of layer names to visualize (supports both direct and mapped names)
            
        Returns:
            Dictionary containing activations for each layer
        """
        if layer_names is None:
            # Default to all Conv2D layers using mapped names
            layer_names = [f'conv{i+1}' for i in range(len([l for l in self.model.layers if isinstance(l, tf.keras.layers.Conv2D)]))]
        
        # Preprocess image
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        results = {}
        
        # Process each requested layer
        for layer_name in layer_names:
            try:
                # Get the actual layer
                layer = self.get_layer_by_name(layer_name)
                if layer is None:
                    continue
                
                # Create intermediate model up to this layer
                # Use self.model.inputs which works for loaded H5 models
                intermediate_model = Model(inputs=self.model.inputs, outputs=layer.output)
                activation = intermediate_model.predict(img_array, verbose=0)
                
                # Store with the requested name (could be mapped name like 'conv1')
                results[layer_name] = {
                    'activation': activation,
                    'shape': activation.shape,
                    'actual_layer_name': layer.name,
                    'layer_type': type(layer).__name__
                }
                
            except Exception as e:
                print(f"Warning: Could not extract feature map for layer '{layer_name}': {e}")
                continue
        
        return results
    
    def plot_feature_maps(self, img_path: str, max_features: int = 64) -> plt.Figure:
        """
        Create a visualization of feature maps from Conv layers.
        
        Args:
            img_path: Path to input image
            max_features: Maximum number of feature maps to display
            
        Returns:
            Matplotlib figure object
        """
        feature_maps = self.get_feature_maps(img_path)
        
        # Get only Conv2D layers
        conv_layers = {k: v for k, v in feature_maps.items() if 'conv' in k.lower()}
        
        if not conv_layers:
            raise ValueError("No convolutional layers found")
        
        num_layers = len(conv_layers)
        fig = plt.figure(figsize=(20, 4 * num_layers))
        
        for layer_idx, (layer_name, data) in enumerate(conv_layers.items()):
            activation = data['activation']
            num_features = min(activation.shape[-1], max_features)
            
            # Calculate grid size
            grid_size = int(np.ceil(np.sqrt(num_features)))
            
            for i in range(num_features):
                ax = plt.subplot(num_layers * grid_size, grid_size, 
                               layer_idx * grid_size * grid_size + i + 1)
                ax.imshow(activation[0, :, :, i], cmap='viridis')
                ax.axis('off')
            
            # Add layer title
            if layer_idx == 0:
                plt.suptitle(f'Feature Maps Visualization\n{layer_name}: {data["shape"][1:]}', 
                           fontsize=14, y=1.02)
        
        plt.tight_layout()
        return fig
    
    def create_activation_heatmap(self, img_path: str, layer_name: str = 'conv4') -> np.ndarray:
        """
        Create an activation heatmap for a specific layer.
        
        Args:
            img_path: Path to input image
            layer_name: Name of the layer to visualize (supports mapped names)
            
        Returns:
            Heatmap array
        """
        # Get the layer (supports both direct and mapped names)
        layer = self.get_layer_by_name(layer_name)
        
        if layer is None:
            available_layers = [f"'{name}'" for name in self.layer_name_mapping.keys()]
            raise ValueError(f"Layer '{layer_name}' not found. Available mapped layers: {', '.join(available_layers)}")
        
        # Create sub-model using model.inputs (works with H5 models)
        activation_model = Model(inputs=self.model.inputs, outputs=layer.output)
        
        # Get activation
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        activation = activation_model.predict(img_array, verbose=0)
        
        # Average across all filters
        heatmap = np.mean(activation, axis=-1)
        heatmap = np.squeeze(heatmap)
        
        # Normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
        
        return heatmap
    
    def plot_layer_statistics(self) -> plt.Figure:
        """
        Plot statistics about model layers.
        
        Returns:
            Matplotlib figure object
        """
        layer_names = []
        param_counts = []
        output_shapes = []
        
        for layer in self.model.layers:
            if hasattr(layer, 'count_params'):
                layer_names.append(layer.name)
                param_counts.append(layer.count_params())
                if hasattr(layer, 'output_shape'):
                    output_shapes.append(np.prod(layer.output_shape[1:]))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Parameter count
        ax1.bar(range(len(layer_names)), param_counts, color='steelblue')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Number of Parameters')
        ax1.set_title('Parameters per Layer', fontweight='bold')
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels(layer_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Output size
        ax2.bar(range(len(layer_names)), output_shapes, color='coral')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Output Size')
        ax2.set_title('Output Size per Layer', fontweight='bold')
        ax2.set_xticks(range(len(layer_names)))
        ax2.set_xticklabels(layer_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def create_confidence_gauge(confidence: float) -> str:
    """
    Create a visual confidence gauge using Unicode characters.
    
    Args:
        confidence: Confidence score between 0 and 1
        
    Returns:
        String representation of confidence gauge
    """
    filled = int(confidence * 10)
    empty = 10 - filled
    gauge = "█" * filled + "░" * empty
    return f"[{gauge}] {confidence:.1%}"


def plot_prediction_comparison(predictions: list) -> plt.Figure:
    """
    Create a bar chart comparing multiple predictions.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [f"Image {i+1}" for i in range(len(predictions))]
    cat_probs = [p.get('probability_cat', 0) for p in predictions]
    dog_probs = [p.get('probability_dog', 0) for p in predictions]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cat_probs, width, label='Cat', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, dog_probs, width, label='Dog', color='#4ECDC4')
    
    ax.set_xlabel('Images')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    plt.tight_layout()
    return fig