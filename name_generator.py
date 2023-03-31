import torch
import torch.nn.functional as F
import pickle
import streamlit as st
import random
import math
import base64

class App:

    def __init__(self):
        # Open saved MLP data from file
        mlp_data = pickle.load(open('./mlp_data.pkl', 'rb'))

        # Set object variables from the data
        self.parameters = mlp_data['parameters']
        self.block_size = mlp_data['block_size']
        self.itos = mlp_data['itos']

        # Setup app
        self.web_app_setup()

        # Run form
        self.web_app_form()

    def web_app_setup(self):
        # Setting the web-app title and icon
        st.set_page_config(
            page_title='Star Wars Name Generator',
            page_icon='./icon.png'
        )

        # Writing and centering title
        st.markdown("""<h1 style='text-align: center;'>Star Wars Name Generator</h1>""", unsafe_allow_html=True)

        # Adding background
        self.add_bg_from_local('./background.jpg')

    def web_app_form(self):
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
                names = self.sample(num_input)
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

    # Adds a background image to the web-app
    def add_bg_from_local(self, image_file):
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

    # Sampling from the model
    def sample(self, num_samples):

        # Initialize generator with random seed
        g = torch.Generator().manual_seed(random.randint(1, 1000000))

        # Set specific model parameters
        C = self.parameters[0]
        W1 = self.parameters[1]
        b1 = self.parameters[2]
        W2 = self.parameters[3]
        b2 = self.parameters[4]

        # Holds generated names
        samples = []

        # Forward pass
        for _ in range(num_samples):

            out = []
            context = [0] * self.block_size
            while True:
                emb = C[torch.tensor([context])] # (1, block_size, d)
                h = torch.tanh(emb.view(1, -1) @ W1 + b1)
                logits = h @ W2 + b2
                probs = F.softmax(logits, dim = 1)
                ix = torch.multinomial(probs, num_samples = 1, generator = g).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
            
            samples.append(''.join(self.itos[i] for i in out))
        return samples

# Main
if __name__ == '__main__':
    a = App()