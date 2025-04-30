import streamlit as st
import re
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

class NLPInterface:
    """Natural language interface for GPU prediction system"""
    
    def __init__(self, predictor, analyzer):
        self.predictor = predictor
        self.analyzer = analyzer
        self.model_keywords = {
            "image": ["image", "picture", "photo", "vision", "cnn", "resnet", "vgg", "efficientnet"],
            "text": ["text", "nlp", "language", "bert", "gpt", "transformer", "word", "sentence"],
            "small": ["small", "tiny", "light", "lightweight", "fast", "quick", "efficient", "mobile"],
            "large": ["large", "big", "heavy", "complex", "powerful", "accurate", "precise"],
        }
        
    def process_query(self, query):
        """Process natural language query and return appropriate model configuration"""
        query = query.lower()
        
        # Detect model domain (image vs text)
        domain_scores = {}
        for domain, keywords in self.model_keywords.items():
            score = sum([1 for keyword in keywords if keyword in query])
            domain_scores[domain] = score
        
        # Determine primary task domain
        is_image = domain_scores["image"] > domain_scores["text"]
        
        # Determine size preference
        is_small = domain_scores["small"] > domain_scores["large"]
        
        # Extract batch size if mentioned
        batch_sizes = [1, 2, 4]  # Default
        batch_match = re.search(r"batch (?:size|sizes)[:\s]+(\d+(?:\s*,\s*\d+)*)", query)
        if batch_match:
            try:
                batch_str = batch_match.group(1)
                batch_sizes = [int(b.strip()) for b in batch_str.split(",")]
            except:
                pass
        
        # Select appropriate model based on query
        if is_image:
            if is_small:
                model_name = "MobileNetV2"
                model = models.mobilenet_v2(weights=None)
                explanation = "I've selected MobileNetV2, a lightweight model designed for efficient image processing on mobile devices."
            else:
                model_name = "ResNet50"
                model = models.resnet50(weights=None)
                explanation = "I've selected ResNet50, a powerful model for image recognition that balances accuracy and performance."
        else:  # Text
            if is_small:
                model_name = "DistilBERT-like"
                # Create a simplified transformer model
                model = self._create_simple_transformer(hidden_size=256, num_layers=2)
                explanation = "I've selected a lightweight transformer model similar to DistilBERT, optimized for efficient text processing."
            else:
                model_name = "BERT-like"
                # Create a larger transformer model
                model = self._create_simple_transformer(hidden_size=512, num_layers=6)
                explanation = "I've selected a BERT-like model with good performance for natural language processing tasks."
        
        return {
            "model": model,
            "model_name": model_name,
            "batch_sizes": batch_sizes,
            "explanation": explanation,
            "is_image": is_image
        }
    
    def _create_simple_transformer(self, hidden_size=512, num_layers=6):
        """Create a simplified transformer model for demonstration"""
        class SimpleTransformer(nn.Module):
            def __init__(self, hidden_size=512, num_layers=6):
                super(SimpleTransformer, self).__init__()
                self.embedding = nn.Embedding(30000, hidden_size)
                encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.classifier = nn.Linear(hidden_size, 2)
                
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                x = x.mean(dim=1)
                x = self.classifier(x)
                return x
        
        return SimpleTransformer(hidden_size, num_layers)
    
    def generate_explanation(self, results, model_name, is_image):
        """Generate plain language explanation of results"""
        fastest_batch = min(results, key=lambda x: x["Execution Time (ms)"] / x["Batch Size"])
        most_efficient_batch = max(results, key=lambda x: x["Batch Size"] / x["Execution Time (ms)"])
        
        task_type = "image processing" if is_image else "text processing"
        
        explanation = f"""
        ## What This Means
        
        The {model_name} model is designed for {task_type} tasks. Based on my analysis:
        
        - **Processing Speed**: At batch size {fastest_batch['Batch Size']}, this model processes each item in about {fastest_batch['Execution Time (ms)'] / fastest_batch['Batch Size']:.2f} milliseconds.
        
        - **Optimal Efficiency**: Batch size {most_efficient_batch['Batch Size']} gives you the best throughput, processing about {1000 * most_efficient_batch['Batch Size'] / most_efficient_batch['Execution Time (ms)']:.1f} items per second.
        
        - **Real-world Impact**: This means the model could process approximately {int(60 * 1000 * most_efficient_batch['Batch Size'] / most_efficient_batch['Execution Time (ms)'])} items per minute.
        
        - **Scaling Behavior**: As batch size increases, the processing time per item {"increases significantly" if fastest_batch['Batch Size'] == 1 else "remains relatively stable"}.
        """
        
        return explanation
