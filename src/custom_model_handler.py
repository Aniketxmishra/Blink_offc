import os
import sys
import torch
import importlib.util
import numpy as np
import json
from pathlib import Path
import subprocess
import inspect
import pkgutil

# Try to import TensorFlow and other frameworks, but don't fail if not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class CustomModelHandler:
    """Handles importing and analyzing custom models from various frameworks."""
    
    def __init__(self, default_cache_dir="models/custom_cache"):
        """Initialize the model handler.
        
        Args:
            default_cache_dir: Directory to cache imported models
        """
        self.default_cache_dir = default_cache_dir
        os.makedirs(default_cache_dir, exist_ok=True)
        self.supported_frameworks = self._detect_supported_frameworks()
        
    def _detect_supported_frameworks(self):
        """Detect which ML frameworks are available in the environment."""
        frameworks = {
            "pytorch": torch is not None,
            "tensorflow": TENSORFLOW_AVAILABLE,
            "onnx": ONNX_AVAILABLE,
            "sklearn": SKLEARN_AVAILABLE
        }
        
        print(f"Supported frameworks: {[k for k, v in frameworks.items() if v]}")
        return frameworks
    
    def import_model(self, model_source, framework=None, **kwargs):
        """Import a model from various sources based on the source type.
        
        Args:
            model_source: Could be a path to a file, a Python class, or a URL
            framework: Explicitly specify the framework (pytorch, tensorflow, etc.)
            **kwargs: Additional arguments specific to the import method
            
        Returns:
            Loaded model and detected framework
        """
        # Auto-detect the framework if not provided
        if framework is None:
            framework = self._auto_detect_framework(model_source)
        
        if isinstance(model_source, str):
            # Check if it's a file path
            if os.path.exists(model_source):
                return self._import_from_file(model_source, framework, **kwargs)
            
            # Check if it's a GitHub URL
            elif model_source.startswith("https://github.com"):
                return self._import_from_github(model_source, **kwargs)
            
            # Check if it's a module path
            elif "." in model_source and not os.path.isfile(model_source):
                return self._import_from_module_path(model_source, **kwargs)
            
            else:
                raise ValueError(f"Unrecognized model source: {model_source}")
        
        # If the model is already a Python object
        elif isinstance(model_source, object) and hasattr(model_source, "__class__"):
            return self._process_model_object(model_source, framework, **kwargs)
        
        else:
            raise ValueError(f"Unsupported model source type: {type(model_source)}")
    
    def _auto_detect_framework(self, model_source):
        """Try to auto-detect the framework from the model source."""
        if isinstance(model_source, str):
            # Check file extension
            if model_source.endswith(".pt") or model_source.endswith(".pth"):
                return "pytorch"
            elif model_source.endswith(".h5") or model_source.endswith(".keras"):
                return "tensorflow"
            elif model_source.endswith(".onnx"):
                return "onnx"
            elif model_source.endswith(".joblib") or model_source.endswith(".pkl"):
                return "sklearn"
        
        # If source is a Python object, check its type
        elif isinstance(model_source, object) and hasattr(model_source, "__class__"):
            module_name = model_source.__class__.__module__
            if module_name.startswith("torch"):
                return "pytorch"
            elif module_name.startswith("tensorflow") or module_name.startswith("keras"):
                return "tensorflow"
            elif module_name.startswith("sklearn"):
                return "sklearn"
            
        # Default to PyTorch if we can't determine
        return "pytorch"
    
    def _import_from_file(self, file_path, framework, **kwargs):
        """Import a model from a file."""
        if framework == "pytorch":
            return self._import_pytorch_from_file(file_path, **kwargs)
        elif framework == "tensorflow" and TENSORFLOW_AVAILABLE:
            return self._import_tensorflow_from_file(file_path, **kwargs)
        elif framework == "onnx" and ONNX_AVAILABLE:
            return self._import_onnx_from_file(file_path, **kwargs)
        elif framework == "sklearn" and SKLEARN_AVAILABLE:
            return self._import_sklearn_from_file(file_path, **kwargs)
        else:
            raise ValueError(f"Framework {framework} is not supported or not installed")
    
    def _import_pytorch_from_file(self, file_path, model_class=None, model_args=None, **kwargs):
        """Import a PyTorch model from a file."""
        if file_path.endswith(".py"):
            # It's a Python file defining a model
            if model_class is None:
                raise ValueError("model_class must be provided for .py files")
                
            # Get the module name from the file path
            module_name = Path(file_path).stem
            
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Get the model class
            model_cls = getattr(module, model_class)
            
            # Instantiate the model
            if model_args is None:
                model = model_cls()
            else:
                model = model_cls(**model_args)
            
        elif file_path.endswith(".pt") or file_path.endswith(".pth"):
            # It's a saved model state
            if model_class is None:
                # Try to load as a complete model
                loaded_data = torch.load(file_path, map_location=torch.device('cpu'))
                
                # Check if loaded_data is already a model or just a state dict
                if isinstance(loaded_data, torch.nn.Module):
                    model = loaded_data
                elif isinstance(loaded_data, dict) and any(k in loaded_data for k in ['state_dict', 'model_state_dict']):
                    # It's a state dict, but we need a model to load it into
                    print("Found state dict but no model class provided. Using torchvision.models.resnet18 as default.")
                    import torchvision.models as models
                    model = models.resnet18(weights=None)
                    
                    # Get the actual state dict
                    state_dict = loaded_data
                    if 'state_dict' in loaded_data:
                        state_dict = loaded_data['state_dict']
                    elif 'model_state_dict' in loaded_data:
                        state_dict = loaded_data['model_state_dict']
                        
                    try:
                        model.load_state_dict(state_dict)
                    except Exception as e:
                        print(f"Warning: Failed to load state dict into default model: {str(e)}")
                else:
                    model = loaded_data  # Hope for the best
            else:
                # Load the model class first, then load state
                if isinstance(model_class, str):
                    # If model_class is a string, look for the class in parent module
                    module_path = kwargs.get("module_path", model_class.rsplit(".", 1)[0] if "." in model_class else "")
                    class_name = model_class.rsplit(".", 1)[1] if "." in model_class else model_class
                    
                    if module_path:
                        try:
                            module = importlib.import_module(module_path)
                            model_cls = getattr(module, class_name)
                        except (ImportError, AttributeError) as e:
                            raise ValueError(f"Could not import {class_name} from {module_path}: {str(e)}")
                    else:
                        # Try to find the class in sys.modules
                        found = False
                        for name, module in sys.modules.items():
                            if hasattr(module, class_name):
                                model_cls = getattr(module, class_name)
                                found = True
                                break
                        
                        if not found:
                            raise ValueError(f"Could not find class {class_name} in loaded modules")
                else:
                    # If model_class is a class object
                    model_cls = model_class
                
                # Instantiate the model
                if model_args is None:
                    model = model_cls()
                else:
                    model = model_cls(**model_args)
                
                # Load the state dictionary
                state_dict = torch.load(file_path, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
        
        else:
            raise ValueError(f"Unsupported PyTorch model file format: {file_path}")
        
        return model, "pytorch"
    
    def _import_tensorflow_from_file(self, file_path, **kwargs):
        """Import a TensorFlow model from a file."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        
        try:
            model = tf.keras.models.load_model(file_path)
            return model, "tensorflow"
        except Exception as e:
            raise ValueError(f"Failed to load TensorFlow model from {file_path}: {e}")
    
    def _import_onnx_from_file(self, file_path, **kwargs):
        """Import an ONNX model from a file."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX and ONNX Runtime are not installed")
        
        try:
            model = onnx.load(file_path)
            # Create an InferenceSession to validate and use the model
            session = ort.InferenceSession(file_path)
            # Return a tuple of both the model and session
            return (model, session), "onnx"
        except Exception as e:
            raise ValueError(f"Failed to load ONNX model from {file_path}: {e}")
    
    def _import_sklearn_from_file(self, file_path, **kwargs):
        """Import a scikit-learn model from a file."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed")
        
        try:
            import joblib
            model = joblib.load(file_path)
            return model, "sklearn"
        except Exception as e:
            raise ValueError(f"Failed to load scikit-learn model from {file_path}: {e}")
    
    def _import_from_github(self, repo_url, model_path=None, model_class=None, model_args=None, **kwargs):
        """Import a model from a GitHub repository."""
        # Extract repository name from URL
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        target_dir = kwargs.get("target_dir", os.path.join(self.default_cache_dir, repo_name))
        
        # Clone the repository if it doesn't exist
        if not os.path.exists(target_dir):
            print(f"Cloning {repo_url} to {target_dir}...")
            subprocess.check_call(['git', 'clone', repo_url, target_dir])
        else:
            print(f"Repository already exists at {target_dir}")
        
        # If model_path is provided, load the model from that path
        if model_path:
            full_model_path = os.path.join(target_dir, model_path)
            framework = kwargs.get("framework", self._auto_detect_framework(full_model_path))
            return self._import_from_file(full_model_path, framework, model_class=model_class, model_args=model_args, **kwargs)
        
        # Otherwise, return the repository directory so the user can explore it
        return target_dir, None
    
    def _import_from_module_path(self, module_path, class_name=None, model_args=None, **kwargs):
        """Import a model from a Python module path."""
        try:
            if "." in module_path and class_name is None:
                # If the module_path includes the class
                module_name, class_name = module_path.rsplit(".", 1)
            else:
                module_name = module_path
            
            # Import the module
            module = importlib.import_module(module_name)
            
            # If class_name is provided, get the class and instantiate it
            if class_name:
                model_cls = getattr(module, class_name)
                
                if model_args is None:
                    model = model_cls()
                else:
                    model = model_cls(**model_args)
                
                # Determine the framework
                framework = self._auto_detect_framework(model)
                
                return model, framework
            else:
                # Return the module if no class specified
                return module, None
                
        except Exception as e:
            raise ValueError(f"Failed to import model from module {module_path}: {e}")
    
    def _process_model_object(self, model_obj, framework=None, **kwargs):
        """Process a model that's already loaded as a Python object."""
        if framework is None:
            framework = self._auto_detect_framework(model_obj)
        
        # If it's already a model object, just return it
        return model_obj, framework
    
    def extract_model_features(self, model, framework=None, input_shape=None):
        """Extract relevant features from a model for prediction.
        
        Args:
            model: The model to analyze
            framework: The framework of the model (pytorch, tensorflow, etc.)
            input_shape: Input shape for the model (excluding batch dimension)
            
        Returns:
            Dictionary of model features
        """
        if framework is None:
            # Try to detect the framework
            if isinstance(model, tuple) and len(model) == 2 and isinstance(model[1], str):
                # If the model was returned with its framework
                model, framework = model
            else:
                framework = self._auto_detect_framework(model)
        
        if framework == "pytorch":
            return self._extract_pytorch_features(model, input_shape)
        elif framework == "tensorflow":
            return self._extract_tensorflow_features(model, input_shape)
        elif framework == "onnx":
            return self._extract_onnx_features(model, input_shape)
        elif framework == "sklearn":
            return self._extract_sklearn_features(model)
        else:
            raise ValueError(f"Feature extraction not supported for framework: {framework}")
    
    def _extract_pytorch_features(self, model, input_shape=None):
        """Extract features from a PyTorch model."""
        # Safety check - make sure we have a proper model
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Expected PyTorch model but got {type(model).__name__}. Make sure to pass a proper model instance.")
            
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size in MB (parameters * 4 bytes for float32)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        features = {
            "model_name": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
        }
        
        # If input shape is provided, run a forward pass to get more details
        if input_shape is not None:
            try:
                # Create a dummy input
                dummy_input = torch.randn(1, *input_shape)
                
                # Record the model architecture details
                from torch.utils.hooks import RemovableHandle
                
                layer_info = []
                hooks = []
                
                def hook_fn(module, input, output):
                    input_shape = input[0].shape if isinstance(input, tuple) and len(input) > 0 else None
                    output_shape = output.shape if hasattr(output, 'shape') else None
                    layer_info.append({
                        'name': module.__class__.__name__,
                        'input_shape': input_shape,
                        'output_shape': output_shape,
                    })
                
                # Register hooks for each module
                for name, module in model.named_modules():
                    if not list(module.children()):  # Only for leaf modules
                        hooks.append(module.register_forward_hook(hook_fn))
                
                # Run forward pass to trigger hooks
                with torch.no_grad():
                    model(dummy_input)
                    
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                # Add flops estimation (simplified)
                features["total_operations"] = self._estimate_flops_pytorch(model, input_shape)
                features["architecture_summary"] = layer_info
                
            except Exception as e:
                print(f"Warning: Failed to extract detailed features from PyTorch model: {e}")
                # Continue with basic features
        
        return features
    
    def _estimate_flops_pytorch(self, model, input_shape):
        """Estimate FLOPs for PyTorch model (simplified)."""
        # This is a simplified estimation - for production use, consider libraries like fvcore
        total_params = sum(p.numel() for p in model.parameters())
        # Roughly estimate - operations are typically 2x the number of parameters in many CNNs
        return total_params * 2
        
    def _extract_tensorflow_features(self, model, input_shape=None):
        """Extract features from a TensorFlow model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        
        features = {
            "model_name": model.__class__.__name__,
        }
        
        # Get trainable parameters
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        total_params = model.count_params()
        
        features["total_parameters"] = total_params
        features["trainable_parameters"] = trainable_params
        features["model_size_mb"] = total_params * 4 / (1024 * 1024)  # Estimate based on float32
        
        # Additional info if model has been built
        if hasattr(model, 'layers'):
            layer_info = []
            for layer in model.layers:
                layer_info.append({
                    'name': layer.name,
                    'class': layer.__class__.__name__,
                    'params': layer.count_params(),
                })
            features["architecture_summary"] = layer_info
        
        return features
    
    def _extract_onnx_features(self, model, input_shape=None):
        """Extract features from an ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is not installed")
        
        # For ONNX models, we might have received a tuple of (model, session)
        if isinstance(model, tuple) and len(model) == 2:
            onnx_model, _ = model
        else:
            onnx_model = model
        
        features = {
            "model_name": onnx_model.graph.name if hasattr(onnx_model.graph, 'name') and onnx_model.graph.name else "ONNXModel",
        }
        
        # Count nodes and estimate parameters (simplified)
        node_count = len(onnx_model.graph.node)
        initializer_count = len(onnx_model.graph.initializer)
        
        # Rough parameter estimation from initializers
        param_count = 0
        for initializer in onnx_model.graph.initializer:
            if hasattr(initializer, 'dims'):
                param_count += np.prod(initializer.dims)
        
        features["total_parameters"] = param_count
        features["trainable_parameters"] = param_count  # Assume all are trainable
        features["model_size_mb"] = param_count * 4 / (1024 * 1024)  # Estimate based on float32
        features["node_count"] = node_count
        features["initializer_count"] = initializer_count
        
        return features
    
    def _extract_sklearn_features(self, model):
        """Extract features from a scikit-learn model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed")
        
        features = {
            "model_name": model.__class__.__name__,
        }
        
        # Estimate parameters (simplified)
        param_size = 0
        param_count = 0
        
        # Get model params
        params = model.get_params()
        
        # Try to get coefficients for linear models
        if hasattr(model, 'coef_'):
            param_count += model.coef_.size
            param_size += model.coef_.nbytes
            
        # Try to get intercept
        if hasattr(model, 'intercept_'):
            if hasattr(model.intercept_, 'size'):
                param_count += model.intercept_.size
                param_size += model.intercept_.nbytes
            else:
                param_count += 1
                param_size += 8  # Assume double
                
        # Try to get feature importances
        if hasattr(model, 'feature_importances_'):
            param_count += model.feature_importances_.size
            param_size += model.feature_importances_.nbytes
            
        # For tree-based models
        if hasattr(model, 'tree_'):
            if hasattr(model.tree_, 'node_count'):
                param_count += model.tree_.node_count * 3  # Each node has 3 parameters (threshold, feature, impurity)
                param_size += model.tree_.node_count * 3 * 8  # Assume 8 bytes per parameter
                
        # For ensemble models
        if hasattr(model, 'estimators_'):
            if isinstance(model.estimators_, list):
                features["n_estimators"] = len(model.estimators_)
            
        features["total_parameters"] = param_count
        features["trainable_parameters"] = param_count  # Sklearn models don't differentiate
        features["model_size_mb"] = param_size / (1024 * 1024) if param_size > 0 else param_count * 8 / (1024 * 1024)
        
        return features
