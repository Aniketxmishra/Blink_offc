import torch
import time
import os
import sys
from pathlib import Path
import hashlib
import json

# Import the custom model handler
from custom_model_handler import CustomModelHandler

# Import prediction API
from prediction_api import load_model, predict_execution_time

class ExternalModelPredictor:
    """Handles importing, analyzing, and predicting performance for external models"""
    
    def __init__(self, max_cache_size=1000):
        """Initialize the predictor with required handlers"""
        self.model_handler = CustomModelHandler()
        # Load the performance prediction model
        self.prediction_model = load_model('models/gradient_boosting_model.joblib')
        
        # Initialize prediction cache
        self.prediction_cache = {}
        self.max_cache_size = max_cache_size
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_cache_stats(self):
        """Return statistics about the prediction cache"""
        # Calculate hit rate with safe division to avoid errors when no requests
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        # Return cache statistics with defaults for empty cache
        return {
            "hit_rate": hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.prediction_cache),
            "max_cache_size": self.max_cache_size
        }
        
    def _generate_cache_key(self, features):
        """Generate a unique key for caching based on feature values"""
        # Create a deterministic string representation of the features
        feature_str = json.dumps(features, sort_keys=True)
        # Hash the feature string to create a compact key
        return hashlib.md5(feature_str.encode()).hexdigest()
    
    def _add_to_cache(self, key, prediction):
        """Add a prediction result to the cache"""
        # Implement simple LRU-like behavior: if cache is full, remove a random entry
        if len(self.prediction_cache) >= self.max_cache_size:
            # Remove the first item (approximating LRU behavior without tracking access times)
            self.prediction_cache.pop(next(iter(self.prediction_cache)))
        
        # Add the new prediction to the cache
        self.prediction_cache[key] = prediction
    def measure_actual_execution_time(self, model, framework, input_shape, batch_sizes=[1, 2, 4], num_iterations=10):
        """Measure actual execution time for a model"""
        # Handle different frameworks
        if framework == "pytorch":
            return self._measure_pytorch_execution_time(model, input_shape, batch_sizes, num_iterations)
        elif framework == "tensorflow":
            return self._measure_tensorflow_execution_time(model, input_shape, batch_sizes, num_iterations)
        elif framework == "onnx":
            return self._measure_onnx_execution_time(model, input_shape, batch_sizes, num_iterations)
        else:
            raise ValueError(f"Execution time measurement not supported for framework: {framework}")
    
    def _measure_pytorch_execution_time(self, model, input_shape, batch_sizes=[1, 2, 4], num_iterations=10):
        """Measure execution time for PyTorch models"""
        device = torch.device("cpu")  # Use CPU for consistency with your data
        model = model.to(device)
        model.eval()
        
        results = []
        
        for batch_size in batch_sizes:
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            
            # Warm-up
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy_input)
            
            # Measure execution time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(dummy_input)
            end_time = time.time()
            
            # Calculate average execution time
            avg_execution_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms
            
            results.append({
                "batch_size": batch_size,
                "actual_execution_time_ms": avg_execution_time
            })
            
            print(f"Batch size {batch_size}: {avg_execution_time:.2f} ms")
        
        return results
        
    def _measure_tensorflow_execution_time(self, model, input_shape, batch_sizes=[1, 2, 4], num_iterations=10):
        """Measure execution time for TensorFlow models"""
        try:
            import tensorflow as tf
            import numpy as np
            
            results = []
            
            for batch_size in batch_sizes:
                # Create dummy input
                dummy_input = np.random.randn(batch_size, *input_shape).astype(np.float32)
                
                # Warm-up
                for _ in range(3):
                    _ = model(dummy_input, training=False)
                
                # Measure execution time
                start_time = time.time()
                for _ in range(num_iterations):
                    _ = model(dummy_input, training=False)
                end_time = time.time()
                
                # Calculate average execution time
                avg_execution_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms
                
                results.append({
                    "batch_size": batch_size,
                    "actual_execution_time_ms": avg_execution_time
                })
                
                print(f"Batch size {batch_size}: {avg_execution_time:.2f} ms")
            
            return results
            
        except ImportError:
            print("TensorFlow is not installed. Cannot measure execution time.")
            return []
            
    def _measure_onnx_execution_time(self, model, input_shape, batch_sizes=[1, 2, 4], num_iterations=10):
        """Measure execution time for ONNX models"""
        try:
            import onnxruntime as ort
            import numpy as np
            
            # For ONNX models, we might have received a tuple of (model, session)
            if isinstance(model, tuple) and len(model) == 2:
                _, session = model
            else:
                # Create a new session
                session = ort.InferenceSession(model.SerializeToString())
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            results = []
            
            for batch_size in batch_sizes:
                # Create dummy input
                dummy_input = np.random.randn(batch_size, *input_shape).astype(np.float32)
                
                # Warm-up
                for _ in range(3):
                    _ = session.run(None, {input_name: dummy_input})
                
                # Measure execution time
                start_time = time.time()
                for _ in range(num_iterations):
                    _ = session.run(None, {input_name: dummy_input})
                end_time = time.time()
                
                # Calculate average execution time
                avg_execution_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms
                
                results.append({
                    "batch_size": batch_size,
                    "actual_execution_time_ms": avg_execution_time
                })
                
                print(f"Batch size {batch_size}: {avg_execution_time:.2f} ms")
            
            return results
            
        except ImportError:
            print("ONNX Runtime is not installed. Cannot measure execution time.")
            return []
    def predict_model_performance(self, model_source, **kwargs):
        """Import a model from any source and predict its performance"""
        # Get parameters
        input_shape = kwargs.get('input_shape', (3, 224, 224))
        batch_sizes = kwargs.get('batch_sizes', [1, 2, 4, 8])
        measure_actual = kwargs.get('measure_actual', True)
        
        # Import the model using the custom handler
        model, framework = self.model_handler.import_model(model_source, **kwargs)
        print(f"Successfully imported {kwargs.get('model_class', model.__class__.__name__)} model using {framework} framework")
        
        # Extract features
        features = self.model_handler.extract_model_features(model, framework, input_shape)
        print(f"Model has {features['total_parameters']:,} parameters and is {features['model_size_mb']:.2f} MB in size")
        
        # Make predictions for each batch size
        predictions = []
        for batch_size in batch_sizes:
            # Create a copy of features with the specific batch size
            batch_features = features.copy()
            batch_features['batch_size'] = batch_size
            
            # Generate a cache key for this prediction
            cache_key = self._generate_cache_key(batch_features)
            
            # Check if prediction is in cache
            if cache_key in self.prediction_cache:
                # Cache hit
                prediction = self.prediction_cache[cache_key]
                self.cache_hits += 1
                print(f"Cache hit for batch size {batch_size}")
            else:
                # Cache miss - make new prediction
                prediction = predict_execution_time(self.prediction_model, batch_features)
                self._add_to_cache(cache_key, prediction)
                self.cache_misses += 1
                print(f"Cache miss for batch size {batch_size}")
            
            # Extract the predicted time value
            predicted_time = 0.0  # Default value
            
            if isinstance(prediction, list):
                # Handle list of prediction dictionaries
                for pred_dict in prediction:
                    if isinstance(pred_dict, dict):
                        # Find dictionary with matching batch size
                        if pred_dict.get('batch_size') == batch_size:
                            # Try to get predicted time
                            if 'predicted_execution_time_ms' in pred_dict:
                                predicted_time = float(pred_dict['predicted_execution_time_ms'])
                                break
                # If no matching batch size was found, try the first item
                if predicted_time == 0.0 and len(prediction) > 0:
                    if isinstance(prediction[0], dict) and 'predicted_execution_time_ms' in prediction[0]:
                        predicted_time = float(prediction[0]['predicted_execution_time_ms'])
            elif isinstance(prediction, dict):
                # For single dictionary predictions, extract the execution time value
                if 'predicted_execution_time_ms' in prediction:
                    predicted_time = float(prediction['predicted_execution_time_ms'])
                elif 'execution_time_ms' in prediction:
                    predicted_time = float(prediction['execution_time_ms'])
            else:
                # Try to use the prediction directly as a number
                try:
                    predicted_time = float(prediction)
                except (TypeError, ValueError):
                    print(f"Warning: Could not extract prediction time from: {prediction}")
            
            # Estimate memory usage
            memory_usage = batch_features['model_size_mb'] + (batch_features['model_size_mb'] * 0.5 * batch_size)
            
            predictions.append({
                "batch_size": batch_size,
                "predicted_time": predicted_time,
                "estimated_memory_mb": memory_usage
            })
            
            # Debug print showing the extracted prediction time
            print(f"Debug - Batch size {batch_size}: Predicted time = {predicted_time:.2f} ms")
        
        # Measure actual execution time if requested
        if measure_actual:
            actual_times = self.measure_actual_execution_time(model, framework, input_shape, batch_sizes)
            
            # Combine predictions with actual measurements
            for i, pred in enumerate(predictions):
                pred["actual_time"] = actual_times[i]["actual_execution_time_ms"] if i < len(actual_times) else None
        
        # Print summary
        # Print summary
        print("\nPerformance Summary:")
        print(f"{'Batch Size':<10} {'Predicted (ms)':<15} {'Actual (ms)':<15} {'Error (%)':<10} {'Memory (MB)':<12}")
        print("-" * 65)
        
        for pred in predictions:
            batch_size = pred["batch_size"]
            predicted = float(pred["predicted_time"])  # Ensure these are floats
            actual = float(pred.get("actual_time", 0)) if pred.get("actual_time") is not None else None
            memory = float(pred["estimated_memory_mb"])
            
            if actual is not None:
                error_percent = abs(predicted - actual) / actual * 100 if actual > 0 else 0
                print(f"{batch_size:<10} {predicted:<15.2f} {actual:<15.2f} {error_percent:<10.2f} {memory:<12.2f}")
            else:
                print(f"{batch_size:<10} {predicted:<15.2f} {'N/A':<15} {'N/A':<10} {memory:<12.2f}")
        return predictions

    def optimize_batch_size(self, model_source, **kwargs):
        """Find the optimal batch size for a model"""
        # Analyze model performance
        performance = self.predict_model_performance(model_source, **kwargs)
        
        # Find the batch size with best throughput
        best_throughput = 0
        optimal_batch = 1
        
        for pred in performance:
            batch_size = pred["batch_size"]
            # Get time in ms, ensuring it's a float
            if "actual_time" in pred and pred["actual_time"] is not None:
                time_ms = float(pred["actual_time"])
            else:
                time_ms = float(pred["predicted_time"])
            
            # Calculate throughput (items/second)
            throughput = (batch_size * 1000) / time_ms if time_ms > 0 else 0
            
            if throughput > best_throughput:
                best_throughput = throughput
                optimal_batch = batch_size
        
        return {
            "optimal_batch_size": optimal_batch,
            "throughput_items_per_second": best_throughput,
            "predictions": performance
        }

def main():
    # Demo: predict performance of a builtin model
    predictor = ExternalModelPredictor()
    
    try:
        # Try to use a model from torchvision
        import torchvision.models as models
        model = models.resnet18(weights=None)
        
        # Save the model for demonstration purposes
        torch.save(model.state_dict(), "resnet18.pt")
        
        # Analyze the model
        results = predictor.predict_model_performance(
            "resnet18.pt",
            model_class="models.resnet18",
            module_path="torchvision",
            input_shape=(3, 224, 224),
            batch_sizes=[1, 2, 4]
        )
    except:
        print("Could not load torchvision. Using GitHub model import instead.")
        
        # Use GitHub import instead
        results = predictor.predict_model_performance(
            "https://github.com/pytorch/vision.git",
            model_path="torchvision/models/resnet.py",
            model_class="ResNet",
            model_args={"num_classes": 1000, "layers": [2, 2, 2, 2]},
            input_shape=(3, 224, 224),
            batch_sizes=[1, 2, 4]
        )

if __name__ == "__main__":
    main()
