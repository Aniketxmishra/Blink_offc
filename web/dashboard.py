import streamlit as st

def show_model_upload():
    """Provides a dedicated drag and drop zone for model uploads"""
    st.header("Upload Your Model")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your model file here",
        type=['pt', 'pth', 'onnx', 'h5', 'pkl', 'joblib'],
        help="Supported formats: PyTorch (.pt, .pth), ONNX (.onnx), TensorFlow (.h5), Scikit-learn (.pkl, .joblib)"
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ Model uploaded successfully")
            file_details = {
                "Filename": uploaded_file.name,
                "File Type": uploaded_file.type,
                "Size": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
            
            file_extension = uploaded_file.name.split('.')[-1].lower()
            framework = None
            if file_extension in ['pt', 'pth']:
                st.info("📊 Detected PyTorch model format")
                framework = "pytorch"
            elif file_extension == 'onnx':
                st.info("📊 Detected ONNX model format")
                framework = "onnx"
            elif file_extension == 'h5':
                st.info("📊 Detected TensorFlow model format")
                framework = "tensorflow"
            elif file_extension in ['pkl', 'joblib']:
                st.info("📊 Detected Scikit-learn model format")
                framework = "sklearn"
            else:
                st.warning("⚠️ Unknown model format")
                framework = None

        with col2:
            architecture_options = {
                "pytorch": ["ResNet", "VGG", "MobileNet", "Custom CNN"],
                "tensorflow": ["MobileNet", "EfficientNet", "Custom CNN"],
                "onnx": ["Any architecture"],
                "sklearn": ["Any architecture"]
            }
            
            architecture = None
            input_shape = None
            if framework and framework in architecture_options:
                architecture = st.selectbox(
                    "Select model architecture (if known)",
                    options=architecture_options[framework],
                    index=0
                )
            if framework in ['pytorch', 'tensorflow', 'onnx']:
                input_shape = st.text_input(
                    "Input Shape (comma-separated, without batch dimension)",
                    value="3,224,224",
                    help="For images, typically channels,height,width (e.g., 3,224,224)"
                )
        return {
            "file": uploaded_file,
            "framework": framework,
            "architecture": architecture,
            "input_shape": input_shape
        }
    return None

def main():
    st.set_page_config(page_title="Neusight Model Dashboard", layout="wide")
    st.title("Neusight Model Dashboard")
    
    # Call the model upload function
    model_info = show_model_upload()
    
    # Display additional information if a model was uploaded
    if model_info:
        st.subheader("Model Information")
        st.write(f"Framework: {model_info['framework']}")
        if model_info['architecture']:
            st.write(f"Architecture: {model_info['architecture']}")
        if model_info['input_shape']:
            st.write(f"Input Shape: {model_info['input_shape']}")

if __name__ == "__main__":
    main()
