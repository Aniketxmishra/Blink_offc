import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import time
import pynvml as nvml
import json
from datetime import datetime
import os

class ModelProfiler:
    def __init__(self, save_dir='data/raw', num_iterations=10):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            print("WARNING: CUDA is not available. Profiling will use CPU only.")
        
        # Number of iterations for profiling
        self.num_iterations = num_iterations
        
    def get_gpu_utilization(self):
        """Get current GPU utilization"""
        if not self.cuda_available:
            return {"error": "CUDA not available"}
            
        try:
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get utilization rates
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            mem_util = utilization.memory
            
            # Get memory usage
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            used_memory = memory_info.used / (1024**2)  # MB
            total_memory = memory_info.total / (1024**2)  # MB
            
            nvml.nvmlShutdown()
            
            return {
                "gpu_utilization_percent": gpu_util,
                "memory_utilization_percent": mem_util,
                "memory_used_mb": used_memory,
                "memory_total_mb": total_memory
            }
        except Exception as e:
            return {"error": str(e)}
    
    def profile_model(self, model, input_shape, batch_sizes=[1, 4, 16, 32], 
                     num_iterations=10, model_name="unknown"):
        """Profile a PyTorch model with different batch sizes"""
        results = []
        
        device = torch.device("cuda" if self.cuda_available else "cpu")
        model = model.to(device)
        model.eval()
        
        # Calculate model parameters and size
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
        
        for batch_size in batch_sizes:
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            
            # Warm-up
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy_input)
                    if self.cuda_available:
                        torch.cuda.synchronize()
            
            # Measure execution time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(dummy_input)
                    if self.cuda_available:
                        torch.cuda.synchronize()
            end_time = time.time()
            
            # Calculate average execution time
            avg_execution_time = (end_time - start_time) / num_iterations
            
            # Get GPU metrics if available
            gpu_metrics = self.get_gpu_utilization() if self.cuda_available else {"error": "CUDA not available"}
            
            # Record results
            result = {
                "model_name": model_name,
                "batch_size": batch_size,
                "input_shape": str(input_shape),
                "execution_time_ms": avg_execution_time * 1000,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": model_size_mb,
                "device": str(device)
            }
            
            # Add GPU metrics if available
            if isinstance(gpu_metrics, dict) and "error" not in gpu_metrics:
                result.update(gpu_metrics)
            
            results.append(result)
            
            # Print progress
            print(f"Profiled {model_name} with batch size {batch_size}: {avg_execution_time*1000:.2f} ms")
        
        # Create DataFrame from results
        df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/{model_name}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        return df
    
    def profile_common_models(self, input_shape=(3, 224, 224), batch_sizes=[1, 4, 16]):
        """Profile common PyTorch models"""
        common_models = {
            "resnet18": models.resnet18(weights=None),
            "resnet50": models.resnet50(weights=None),
            "mobilenet_v2": models.mobilenet_v2(weights=None),
            "densenet121": models.densenet121(weights=None),
            "vgg16": models.vgg16(weights=None),
        }
        
        all_results = []
        
        for name, model in common_models.items():
            print(f"Profiling {name}...")
            results = self.profile_model(
                model=model,
                input_shape=input_shape,
                batch_sizes=batch_sizes,
                model_name=name
            )
            all_results.append(results)
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/combined_models_{timestamp}.csv"
        combined_df.to_csv(filename, index=False)
        print(f"Combined results saved to {filename}")
        
        return combined_df
