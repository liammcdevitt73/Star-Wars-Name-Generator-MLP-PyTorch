import torch
import pickle
import streamlit as st
import random
import math
import base64
#from mlp import MLP

# Adds a background image to the web-app
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Main
if __name__ == '__main__':
    # Open saved MLP model from file
    #mlp = pickle.load(open('./mlp.obj', 'rb'))

    # Initialize generator with random seed
    g = torch.Generator().manual_seed(random.randint(1, 1000000))

    # Setting the web-app title and icon
    st.set_page_config(
        page_title='Star Wars Name Generator',
        page_icon='./icon.png'
    )

    # Writing and centering title
    st.markdown("""<h1 style='text-align: center;'>Star Wars Name Generator</h1>""", unsafe_allow_html=True)

    # Adding background
    add_bg_from_local('./background.jpg')

    # Setting up user form
    form = st.form(key="user_settings")
    with form:
        # Slider to get the number of names to generate
        num_input = st.slider("Number of Names to Generate", value = 10, key = "num_input", min_value=1, max_value=20)
        # Generator button
        generate_button = form.form_submit_button("Generate Names")
        # Setup two columns to print the names
        col1, col2 = st.columns(2)
        # If the button is pressed, generate names
        if generate_button:
            # Sample model for names
            #names = mlp.sample(num_input, g)
            names = ['liam.'] * 20
            # Format readable names
            names = [name[:-1].capitalize() for name in names]
            # Split names into column
            ix1 = math.ceil(len(names) / 2)
            with col1:
                for name in names[:ix1]:
                    st.write(name)
            with col2:
                for name in names[ix1:]:
                        st.write(name)