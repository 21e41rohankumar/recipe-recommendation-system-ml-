import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Namaste Food Online Recipe Recommendation", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("recipes_big.csv")

    def clean_path(path):
        path = str(path).strip().replace("\\", "/")
        if not path.startswith("http") and not path.startswith("project/") and not path.startswith("images/"):
            filename = os.path.basename(path)
            path = f"project/{filename}"
        return path

    data['ImageURL'] = data['ImageURL'].apply(clean_path)
    return data

# Train model
def train_model(data):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data['Ingredients'])
    return tfidf, tfidf_matrix

# Recommend
def recommend_recipe(user_ingredients, tfidf, tfidf_matrix, data):
    user_tfidf = tfidf.transform([user_ingredients])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    index = similarities.argsort()[0][-1]
    return data.iloc[index]

# UI
def main():
    st.markdown("""
        <style>
        .stApp {
            background-color:white;
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .header {
            background-color: #e74c3c;
            padding: 12px;
            text-align: left;
            color: white;
            font-size: 36px;
            font-weight: bold;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .input-style input {
            height: 50px !important;
            font-size: 16px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header">Namaste Food Cart</div>', unsafe_allow_html=True)

    st.title("🍽️ Recipe Recommendation System")
    st.write("Enter the ingredients you have and get a delicious recipe suggestion!")

    data = load_data()
    tfidf, tfidf_matrix = train_model(data)

    with st.container():
        with st.form(key='input_form'):
            user_input = st.text_input("Enter ingredients (comma separated)", "", key="input", help="e.g., tomato, onion, garlic")
            submit_button = st.form_submit_button(label='Recommend Recipe')

    if submit_button and user_input.strip():
        with st.spinner('Finding the best recipe for you... 🍳'):
            recommended = recommend_recipe(user_input, tfidf, tfidf_matrix, data)

        recipe_name = recommended['RecipeName']
        ingredients = recommended['Ingredients']
        recipe_image = recommended['ImageURL']

        st.success('Here is your recommendation! 🎯')

        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.subheader("🍴 Recipe Name:")
            st.markdown(f"**{recipe_name}**")
            st.subheader("🧂 Ingredients:")
            st.markdown(f"{ingredients}")
        with col2:
            if recipe_image and pd.notna(recipe_image):
                if recipe_image.startswith("http"):
                    st.image(recipe_image, caption=f"{recipe_name}", width=350)
                else:
                    if os.path.exists(recipe_image):
                        img = Image.open(recipe_image)
                        st.image(img, caption=f"{recipe_name}", width=350)
                    else:
                        st.warning(f"Image not found: {recipe_image}")
            else:
                st.warning("No image available for this recipe.")

    elif submit_button:
        st.warning("Please enter some ingredients.")

if __name__ == "__main__":
    main()
