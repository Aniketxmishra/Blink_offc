from external_model_predictor import ExternalModelPredictor
import torch
import torch.nn as nn

# 1. Define different model architectures for testing
class LightweightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
    
    def forward(self, x):
        return self.network(x)

class HeavyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1000)
        )
    
    def forward(self, x):
        return self.network(x)

def analyze_models():
    predictor = ExternalModelPredictor()
    
    models_to_test = [
        ("Lightweight CNN", LightweightCNN()),
        ("Heavy Model", HeavyModel())
    ]
    
    print("Analyzing different model architectures...")
    print("-" * 50)
    
    for model_name, model in models_to_test:
        print(f"\nAnalyzing {model_name}:")
        
        # Save model
        torch.save(model.state_dict(), f"{model_name.lower().replace(' ', '_')}.pt")
        
        # Analyze with different batch sizes
        results = predictor.predict_model_performance(
            f"{model_name.lower().replace(' ', '_')}.pt",
            model_class=model.__class__,
            input_shape=(3, 224, 224),
            batch_sizes=[1, 4, 8, 16],
            measure_actual=True
        )
        
        # Find optimal batch size
        optimization = predictor.optimize_batch_size(
            f"{model_name.lower().replace(' ', '_')}.pt",
            model_class=model.__class__,
            input_shape=(3, 224, 224),
            batch_sizes=[1, 4, 8, 16]
        )
        
        print(f"\nOptimal Configuration for {model_name}:")
        print(f"Best Batch Size: {optimization['optimal_batch_size']}")
        print(f"Throughput: {optimization['throughput_items_per_second']:.2f} items/second")
        
        # Cleanup
        import os
        os.remove(f"{model_name.lower().replace(' ', '_')}.pt")

if __name__ == "__main__":
    analyze_models()

