import joblib
import numpy as np
import pandas as pd
import argparse
import os
import torch
import torch.nn as nn
from thop import profile

def load_model(model_path='models/gradient_boosting_model.joblib'):
    """Load the trained prediction model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Loaded prediction model from {model_path}")
    return model

def extract_model_features(model, input_shape):
    """Extract features from a PyTorch model"""
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    # Count FLOPs (optional)
    try:
        device = torch.device("cpu")
        dummy_input = torch.randn(1, *input_shape, device=device)
        macs, _ = profile(model, inputs=(dummy_input,))
        flops = macs * 2
    except Exception as e:
        print(f"Warning: Could not count FLOPs: {e}")
        flops = 0
    
    features = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": model_size_mb,
        "flops": flops
    }
    
    return features

def predict_execution_time(prediction_model, model_features, batch_sizes=[1, 2, 4, 8]):
    """Predict execution time for different batch sizes"""
    predictions = []
    
    for batch_size in batch_sizes:
        # Create feature vector for prediction
        features = {
            "total_parameters": model_features["total_parameters"],
            "trainable_parameters": model_features["trainable_parameters"],
            "model_size_mb": model_features["model_size_mb"],
            "batch_size": batch_size
        }
        
        # Convert to DataFrame for prediction
        X = pd.DataFrame([features])
        
        # Predict execution time
        execution_time = prediction_model.predict(X)[0]
        
        predictions.append({
            "batch_size": batch_size,
            "predicted_execution_time_ms": execution_time
        })
    
    return predictions

def predict_memory_usage(prediction_model, model_features, batch_sizes=[1, 2, 4, 8]):
    """Predict memory usage for different batch sizes"""
    # For now, we'll use a simple heuristic based on your data
    # In a more advanced implementation, you would train a separate model for this
    base_memory = model_features["model_size_mb"]
    
    predictions = []
    for batch_size in batch_sizes:
        # Memory usage typically scales with batch size, but not linearly
        # This is a simplified approximation based on your collected data
        estimated_memory = base_memory * (1 + 0.2 * (batch_size - 1))
        
        predictions.append({
            "batch_size": batch_size,
            "predicted_memory_usage_mb": estimated_memory
        })
    
    return predictions

def predict_for_custom_model(prediction_model, custom_model, input_shape, batch_sizes=[1, 2, 4, 8]):
    """Predict execution time and memory usage for a custom PyTorch model"""
    # Extract features from the model
    features = extract_model_features(custom_model, input_shape)
    
    # Predict execution time
    time_predictions = predict_execution_time(prediction_model, features, batch_sizes)
    
    # Predict memory usage
    memory_predictions = predict_memory_usage(prediction_model, features, batch_sizes)
    
    # Combine predictions
    combined_predictions = []
    for i in range(len(batch_sizes)):
        combined_predictions.append({
            "batch_size": batch_sizes[i],
            "predicted_execution_time_ms": time_predictions[i]["predicted_execution_time_ms"],
            "predicted_memory_usage_mb": memory_predictions[i]["predicted_memory_usage_mb"]
        })
    
    # Print results
    print(f"\nModel Features:")
    print(f"  Total Parameters: {features['total_parameters']:,}")
    print(f"  Model Size: {features['model_size_mb']:.2f} MB")
    
    print("\nPredicted Performance:")
    for pred in combined_predictions:
        print(f"  Batch Size {pred['batch_size']}:")
        print(f"    Execution Time: {pred['predicted_execution_time_ms']:.2f} ms")
        print(f"    Memory Usage: {pred['predicted_memory_usage_mb']:.2f} MB")
    
    return combined_predictions

def create_sample_model(num_layers=3, channels=16):
    """Create a sample CNN model for testing"""
    class SampleCNN(nn.Module):
        def __init__(self, num_layers=3, channels=16):
            super(SampleCNN, self).__init__()
            layers = []
            in_channels = 3
            
            for i in range(num_layers):
                out_channels = channels * (2 ** i)
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(2))
                in_channels = out_channels
            
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(out_channels * (224 // (2**num_layers)) * (224 // (2**num_layers)), 10)
            
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    return SampleCNN(num_layers, channels)

def main():
    parser = argparse.ArgumentParser(description='Predict GPU usage for deep learning models')
    parser.add_argument('--model-path', type=str, default='models/gradient_boosting_model.joblib',
                        help='Path to the trained prediction model')
    parser.add_argument('--test-model', action='store_true',
                        help='Test with a sample model')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of layers for the test model')
    parser.add_argument('--channels', type=int, default=16,
                        help='Base number of channels for the test model')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Batch sizes to predict for')
    
    args = parser.parse_args()
    
    # Load the prediction model
    prediction_model = load_model(args.model_path)
    
    if args.test_model:
        # Create a sample model
        print(f"Creating a sample CNN with {args.layers} layers and {args.channels} base channels...")
        sample_model = create_sample_model(args.layers, args.channels)
        
        # Predict execution time
        predictions = predict_for_custom_model(
            prediction_model, 
            sample_model, 
            (3, 224, 224), 
            args.batch_sizes
        )
    else:
        print("Use --test-model to test with a sample model")
        print("You can also import this module and use predict_for_custom_model() with your own models")

if __name__ == "__main__":
    main()
