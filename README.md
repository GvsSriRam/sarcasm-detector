# sarcasm-detector

This project is a sarcasm detection system built using a deep learning model. It can be used to detect sarcasm in text and memes.

## Installation

1. Clone the repository: `git clone https://github.com/GvsSriRam/sarcasm-detector.git`
2. Create a virtual environment: `python -m venv .venv`
3. Activate the virtual environment:
    - On Windows: `.venv\Scripts\activate`
    - On macOS and Linux: `source .venv/bin/activate`
4. Install the dependencies: `pip install -r requirements.txt`

### Training

1. **Prepare the data:** 
    - Download, extract the data from `https://www.kaggle.com/datasets/naifislam/goat-bench` and keep it in `Data` folder.
    - Preprocess the data and extract the features using the `feature_extraction.ipynb` in the `Notebooks` directory.
2. **Train the model:** 
    - Use the `Notebooks/multiclass.ipynb` notebook to train the sarcasm detection model. 
    - This notebook includes code for loading data, defining the model architecture, training the model, and saving the trained model.

### Prediction

1. **Load the trained model:** 
    - Load the saved model file (e.g., `Notebooks/meme_sarcasm_detection_model.h5`) in your prediction script.
2. **Preprocess the input:** 
    - Preprocess the input text or meme in the same way as you did during training.
3. **Make predictions:** 
    - Use the loaded model to predict the probability of sarcasm in the input.
    - Prediction script is provided in `scripts/predict.py`.

### App Startup

1. **Run the Flask app:** 
    - Execute `streamlit run app.py` to start the web application.
    - This will start a web server that you can access in your browser.
2. **Access the app:** 
    - Open your web browser and go to `http://127.0.0.1:5000/` (or the appropriate address and port).
    - You should see the web interface of the sarcasm detection app.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvement, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Git Repository

https://github.com/GvsSriRam/sarcasm-detector.git