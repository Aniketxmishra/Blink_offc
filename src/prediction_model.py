import pandas as pd
import numpy as np
import os
import json
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_dir='data/processed'):
    """Load feature data from CSV and JSON files"""
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return None

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if not csv_files and not json_files:
        print(f"No CSV or JSON files found in {data_dir}.")
        return None

    all_data = []
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            all_data.extend(df.to_dict('records'))
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    df = pd.DataFrame(all_data)
    print(f"DataFrame columns: {df.columns.tolist()}")
    return df

def prepare_features(df):
    """Prepare features for model training"""
    # Group by model_name and batch_size to get unique model configurations
    # For each configuration, we'll use the model parameters as features
    # and execution time as the target
    
    features = []
    for (model_name, batch_size), group in df.groupby(['model_name', 'batch_size']):
        # Get the first row for this configuration
        row = group.iloc[0]
        
        # Create feature dictionary
        feature_dict = {
            'model_name': model_name,
            'batch_size': batch_size,
            'total_parameters': row['total_parameters'],
            'trainable_parameters': row['trainable_parameters'],
            'model_size_mb': row['model_size_mb'],
            'execution_time_ms': row['execution_time_ms']
        }
        
        features.append(feature_dict)
    
    feature_df = pd.DataFrame(features)
    
    return feature_df

def train_models(df, target='execution_time_ms', test_size=0.2, random_state=42):
    """Train and evaluate multiple regression models"""
    # Define features and target
    feature_cols = ['total_parameters', 'trainable_parameters', 'model_size_mb', 'batch_size']
    X = df[feature_cols]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Define models
    models = {
        'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
        'Ridge Regression': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'Lasso Regression': make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        trained_models[name] = model
    
    # Find best model
    best_model_name = min(results, key=lambda k: results[k]['RMSE'])
    print(f"Best model: {best_model_name}")
    
    return trained_models, results, X_test, y_test, best_model_name

def visualize_results(models, results, X_test, y_test, best_model_name):
    """Create visualizations of model performance and feature importance"""
    # Plot model comparison
    plt.figure(figsize=(12, 6))
    
    metrics = ['RMSE', 'MAE', 'R2']
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        values = [results[model][metric] for model in results]
        sns.barplot(x=list(results.keys()), y=values)
        plt.title(f'Model Comparison - {metric}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # Plot predicted vs actual for best model
    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Execution Time (ms)')
    plt.ylabel('Predicted Execution Time (ms)')
    plt.title(f'Predicted vs Actual - {best_model_name}')
    plt.savefig('predicted_vs_actual.png')
    
    # Plot feature importance for tree-based models
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        # For pipelines, the model is the last step
        if hasattr(best_model, 'steps'):
            model = best_model.steps[-1][1]
        else:
            model = best_model
            
        feature_cols = ['total_parameters', 'trainable_parameters', 'model_size_mb', 'batch_size']
        
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png')

def save_best_model(models, best_model_name, output_dir='models'):
    """Save the best model for later use"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"{best_model_name.lower().replace(' ', '_')}_model.joblib")
    joblib.dump(models[best_model_name], model_path)
    
    print(f"Best model saved to {model_path}")
    
    return model_path

def main():
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Prepare features
    feature_df = prepare_features(df)
    
    # Train models
    models, results, X_test, y_test, best_model_name = train_models(feature_df)
    
    # Visualize results
    visualize_results(models, results, X_test, y_test, best_model_name)
    
    # Save best model
    model_path = save_best_model(models, best_model_name)
    
    print("Model training and evaluation complete!")
    print(f"Results saved to results/ directory")
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    main()
