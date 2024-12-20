# Brain Tumor Detection and Classification

## Overview
This project is designed to detect and classify brain tumors using MRI images. It supports four classes:
- **Glioma Tumor**
- **Meningioma Tumor**
- **No Tumor**
- **Pituitary Tumor**

The project includes functionality for data exploration, model metrics evaluation, explainability insights, and a Streamlit application for testing the model.

## Dataset
The dataset used for this project is available on Kaggle: [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data).

### Data Summary
- The dataset contains MRI scans categorized into four classes as listed above.
- It includes separate folders for training, validation, and testing.
- Images are preprocessed, resized, normalized, and converted to grayscale for model training.

## Features
1. **Data Exploration**: An interactive web-based HTML file to explore the dataset and gain insights.
2. **Model Metrics**: A detailed HTML report showing the evaluation metrics of the trained model.
3. **Explainability**: An HTML file providing insights into model decision-making using techniques like SHAP or Grad-CAM.
4. **Streamlit Application**: A user-friendly interface to test the trained model with custom MRI images.

## Setup and Execution
### Prerequisites
- Python 3.11
- A system capable of running virtual environments

### Instructions
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Run the setup script:
    ```bash
    ./lancez_moi.sh
    ```
    This script will:
    - Create a virtual environment.
    - Activate the virtual environment.

3. After activation, the following operations are available:
    - Open the **data exploration** report in your default web browser.
    - View the **model metrics** report.
    - Access the **explainability** insights.
    - Launch the **Streamlit application**.

### Streamlit Application
To launch the Streamlit app manually:
```bash
streamlit run app.py
```
This app allows you to upload MRI images and view classification results in real time.

## Folder Structure
```
project-root/
|
|-- data/                 # Dataset folder (training, validation, testing)
|-- scripts/              # Scripts for preprocessing, training, and evaluation
|-- reports/              # HTML reports for exploration, metrics, and explainability
|-- app.py                # Streamlit application
|-- lancez_moi.sh         # Setup script
|-- README.md             # Project documentation
```

## Additional Notes
- Ensure that the dataset is downloaded and placed in the `data/` folder before running the scripts.
- For detailed information on each report, refer to the `reports/` folder.
- The project uses libraries like TensorFlow, Scikit-learn, and Streamlit for model training and deployment.

## Contribution
Feel free to open issues or submit pull requests for any enhancements or bug fixes.

## License
This project is licensed under the [MIT License](LICENSE).
