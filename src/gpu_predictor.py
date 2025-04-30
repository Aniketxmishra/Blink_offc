import joblib
import pandas as pd
import numpy as np
import os
import torch
from datetime import datetime

class GPUPredictor:
    """Scalable GPU usage prediction system with caching and batch processing"""
    
    def __init__(self, model_path='models/gradient_boosting_model.joblib', cache_size=100):
        self.model = joblib.load(model_path)
        self.prediction_cache = {}  # Cache for fast repeated predictions
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        # Define the feature order used during training
        self.feature_cols = ['total_parameters', 'trainable_parameters', 'model_size_mb', 'batch_size']
    
    def predict(self, features_batch):
        """Make predictions for a batch of models efficiently"""
        if not isinstance(features_batch, list):
            features_batch = [features_batch]
            
        results = []
        features_to_predict = []
        cache_indices = []
        
        # Check cache first
        for i, features in enumerate(features_batch):
            cache_key = self._get_cache_key(features)
            if cache_key in self.prediction_cache:
                results.append(self.prediction_cache[cache_key])
                self.cache_hits += 1
            else:
                results.append(None)
                features_to_predict.append(features)
                cache_indices.append(i)
                self.cache_misses += 1
        
        # Make predictions for cache misses
        if features_to_predict:
            # Extract features in the exact order used during training
            numeric_features = []
            for features in features_to_predict:
                feature_dict = {
                    'total_parameters': features.get('total_parameters', 0),
                    'trainable_parameters': features.get('trainable_parameters', 0),
                    'model_size_mb': features.get('model_size_mb', 0),
                    'batch_size': features.get('batch_size', 1)
                }
                numeric_features.append(feature_dict)
            
            # Convert to DataFrame with specific column order
            features_df = pd.DataFrame(numeric_features)[self.feature_cols]
            
            # Make batch prediction
            predictions = self.model.predict(features_df)
            
            # Update results and cache
            for i, pred_idx in enumerate(cache_indices):
                results[pred_idx] = predictions[i]
                
                # Update cache
                cache_key = self._get_cache_key(features_batch[pred_idx])
                self.prediction_cache[cache_key] = predictions[i]
            
            # Limit cache size
            if len(self.prediction_cache) > self.cache_size:
                # Remove oldest entries (simple approach)
                keys_to_remove = list(self.prediction_cache.keys())[:-self.cache_size]
                for key in keys_to_remove:
                    del self.prediction_cache[key]
        
        return results[0] if len(results) == 1 else results
    
    def _get_cache_key(self, features):
        """Generate a cache key from features"""
        key_parts = []
        for k in self.feature_cols:  # Use the same order as feature columns
            if k in features:
                key_parts.append(f"{k}:{features[k]}")
        return "|".join(key_parts)
    
    def get_cache_stats(self):
        """Return cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "cache_size": len(self.prediction_cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }
    
    def optimize_batch_size(self, model_features, min_batch=1, max_batch=32, memory_limit_mb=8000):
        """Find optimal batch size for throughput within memory constraints"""
        # Base memory is the model size
        base_memory = model_features['model_size_mb']
        
        # Memory scaling factor based on model size
        if model_features['total_parameters'] > 100000000:  # Large models like VGG16
            mem_scale_factor = 0.5
        else:  # Smaller models
            mem_scale_factor = 0.3
            
        best_throughput = 0
        optimal_batch = min_batch
        
        # Test different batch sizes
        batch_results = []
        for batch_size in range(min_batch, max_batch + 1):
            # Estimate memory usage
            memory_usage = base_memory + (base_memory * mem_scale_factor * batch_size)
            
            # Skip if exceeds memory limit
            if memory_usage > memory_limit_mb:
                continue
                
            # Predict execution time
            features = model_features.copy()
            features['batch_size'] = batch_size
            exec_time = self.predict(features)
            
            # Calculate throughput (samples/second)
            throughput = (batch_size * 1000) / exec_time
            
            batch_results.append({
                'batch_size': batch_size,
                'exec_time_ms': exec_time,
                'throughput': throughput,
                'memory_usage_mb': memory_usage
            })
            
            # Update optimal if better
            if throughput > best_throughput:
                best_throughput = throughput
                optimal_batch = batch_size
        
        return {
            'optimal_batch_size': optimal_batch,
            'predicted_execution_time': next((r['exec_time_ms'] for r in batch_results if r['batch_size'] == optimal_batch), None),
            'estimated_memory_usage': next((r['memory_usage_mb'] for r in batch_results if r['batch_size'] == optimal_batch), None),
            'batch_results': batch_results
        }
# Add this to gpu_predictor.py

def predict_with_tiling(self, features, tile_size=128):
    """Predict execution time using tile-based approach for better accuracy
    
    This implements a simplified version of the approach described in NeuSight research,
    which decomposes the prediction problem into smaller tile-level predictions.
    """
    # Extract key features
    total_params = features['total_parameters']
    batch_size = features['batch_size']
    
    # Determine number of tiles based on model size
    num_tiles = max(1, int(total_params / tile_size))
    
    # Make tile-level predictions
    tile_predictions = []
    for i in range(num_tiles):
        # Create tile features (simplified)
        tile_features = {
            'batch_size': batch_size,
            'total_parameters': min(tile_size, total_params - i * tile_size),
            'trainable_parameters': min(tile_size, features['trainable_parameters'] - i * tile_size),
            'model_size_mb': features['model_size_mb'] * (min(tile_size, total_params - i * tile_size) / total_params)
        }
        
        # Predict tile execution time
        tile_prediction = self.predict(tile_features)
        tile_predictions.append(tile_prediction)
    
    # Aggregate tile predictions (with overlap factor)
    # In real implementation, this would use a more sophisticated aggregation
    overlap_factor = 0.8  # Represents execution overlap between tiles
    aggregated_prediction = sum(tile_predictions) * overlap_factor
    
    return aggregated_prediction
