import streamlit as st

from scripts.predict import Predictor

# Predictor object
predictor = Predictor()

# Predictor function to be exposed
def predict_meme_class(image_file):
    return predictor.predict(image_file)

# --- Streamlit App ---
st.title("Meme Classifier")
st.write("Upload a meme to classify it!")

uploaded_file = st.file_uploader("Choose a meme image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Meme", use_container_width=True)

    # Simulate prediction
    if st.button("Classify Meme"):
        with st.spinner("Classifying..."):
            prediction = predict_meme_class(uploaded_file)
            st.success(f"**{prediction.upper()}**")

# --- CSS Styling (Simple & Beautiful) ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0a0908;
        color: #EAE0D5;
        font-family: sans-serif;
    }
    .stButton button {
        background-color: #22333B;
        color: #C6AC8F;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #C6AC8F; /* Slightly darker white on hover */
        color: #0A0908;
    }
    .stButton button:focus {
        background-color: #C6AC8F; /* Keep white background on focus */
        color: #0A0908;  /* Keep sky blue text color on focus */
        outline: 
    }
    .stSuccess {
        color: #ffffff; /* White success message */
        background-color: #5E503F;
    }
    </style>
    """,
    unsafe_allow_html=True,
)