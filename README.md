<div style="height: 80px; background: linear-gradient(to right, #6a11cb, #2575fc); border-bottom-left-radius: 50% 20%; border-bottom-right-radius: 50% 20%;"></div>







# NeuSight

A machine learning project for GPU-accelerated vision model prediction and optimization with an interactive web dashboard.

## Project Overview

NeuSight provides tools for running, optimizing, and analyzing vision models with GPU acceleration. The project includes:

- Batch processing optimization  
- Model inference and prediction  
- GPU resource monitoring and management  
- Interactive web dashboard for visualization and control  
- Comparative analysis of model performance

## Directory Structure

```
NeuSight/
├── src/                  # Main source code
│   ├── batch_optimization.py   # Batch processing optimization
│   ├── model_handlers.py       # Model loading and management
│   ├── prediction.py           # Model prediction functions
│   ├── gpu_info.py             # GPU resource monitoring
│   └── ...                     # Other source modules
├── tests/                # Test scripts
├── requirements/         # Dependency specifications
│   └── requirements.txt  # Python package requirements
├── web/                  # Web dashboard interface
│   ├── app.py            # Main web application (likely Streamlit)
│   └── templates/        # Web templates
├── static/               # Static assets like images
└── .gitignore            # Git ignore file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aniketxmishra/NeuSight.git
   cd NeuSight
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv neusight_env
   # On Windows:
   neusight_env\Scripts\activate
   # On Linux/Mac:
   source neusight_env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements/requirements.txt
   ```

## Usage

### Running the Web Dashboard

```bash
streamlit run web/app.py
```

### Using the Prediction API

```python
from src.prediction import predict
from src.model_handlers import load_model

# Load a model
model = load_model("path/to/model")

# Run prediction
result = predict(model, input_data)
```

## Dependencies

Key dependencies include:
- streamlit
- torch, torchvision
- numpy, pandas
- scikit-learn
- joblib
- thop (for model analysis)
- plotly, matplotlib, seaborn (for visualization)

## Important Notes

- Large data files, model weights, and results are not tracked in git
- The `github_models/vision` directory contains an external repository that is not tracked
- For GPU acceleration, ensure you have compatible CUDA drivers installed

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License – see the LICENSE file for details.

## Contact

Aniket Mishra - [anik8mishra@gmail.com]

<img src="https://raw.githubusercontent.com/mayhemantt/mayhemantt/Update/svg/Bottom.svg" alt="Wave" style="width: 100%; height: auto;" /> 

