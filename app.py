from typing import Any
import torch
import torch.nn.functional as F
import random
import dill as pickle
import streamlit as st
import math
import base64
import os

# Linear layer of a network
class Linear:
    
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
# Batch normalization 1 dimensional layer of a network
class BatchNorm1d:
    
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # Parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # Buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.zeros(dim)
        
    def __call__(self, x):
        # Calculate the forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True) # Batch mean
            xvar = x.var(dim, keepdim=True) # Batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # Normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # Update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
# Activation function
class Tanh:
    
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
    
# Embedding layer
class Embedding:
    
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]

# Consecutive flatten layer
class FlattenConsecutive:
    
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(self):
        return []
    
# Sequential network 
class Sequential:
    
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        # Get parameters of all layers and stretch them out into one list
        return [p for layer in self.layers for p in layer.parameters()]

# Building a character-level language model
class Network:

    def __init__(self, text_file):

        # Read in words from a text file
        self.words = open(text_file, 'r').read().splitlines()

        # Context length: how many characters do we take to predict the next one?
        self.block_size = 8 

        # Build the vocabulary of character mappings to/from integers
        self.chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s:i+1 for i, s in enumerate(self.chars)} # String -> Integer
        self.stoi['.'] = 0
        self.itos = {i:s for s,i in self.stoi.items()} # Integer -> String
        self.vocab_size = len(self.itos)

        # Setting seeds
        random.seed(0)
        torch.manual_seed(0)

        # Training split (80%), dev/validation split (10%), test split (10%)
        self.splitting()

        # Initialize network
        self.init_network()

        # Train network
        self.training()

        # Evaluate Losses
        self.evaluate_loss()

    # Build the dataset
    def build_dataset(self, words):
        
        X, Y = [], []

        for w in words:
            context = [0] * self.block_size
            for ch in w + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix] # Crop & append

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y
    
    # Split training, dev/validation, and testing dataset
    def splitting(self):
        random.shuffle(self.words)
        n1 = int(0.8 * len(self.words))
        n2 = int(0.9 * len(self.words))
        self.Xtr, self.Ytr = self.build_dataset(self.words[:n1])
        self.Xdev, self.Ydev = self.build_dataset(self.words[n1:n2])
        self.Xte, self.Yte = self.build_dataset(self.words[n2:])

    # Initialize hierarchical network
    def init_network(self):
        
        n_embd = 24 # The dimensionality of the character embedding vectors
        n_hidden = 128 # The number of neurons in the hidden layers of the MLP
        
        self.model = Sequential([
            Embedding(self.vocab_size, n_embd),
            FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
            FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
            FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, self.vocab_size)
        ])
        
        # Make the last layer less confident
        with torch.no_grad():
            self.model.layers[-1].weight *= 0.1
        
        # Set parameters that need gradient calculations for
        self.parameters = self.model.parameters()
        for p in self.parameters:
            p.requires_grad = True


    # Train network
    def training(self):
        
        max_steps = 200000
        batch_size = 32
        
        for i in range(max_steps):

            # Minibatch construct
            ix = torch.randint(0, self.Xtr.shape[0], (batch_size,))
            Xb, Yb = self.Xtr[ix], self.Ytr[ix]

            # Forward pass
            logits = self.model(Xb)
            loss = F.cross_entropy(logits, Yb)

            # Backward pass
            for p in self.parameters:
                p.grad = None
            loss.backward()

            # Update
            lr = 0.1 if i < 150000 else 0.01 # Step learning rate decay
            for p in self.parameters:
                p.data += -lr * p.grad
                
            # Track stats
            if i % 10000 == 0:
                print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

    # Split -> loss
    @torch.no_grad() # Disables gradient tracking
    def split_loss(self, split):
        x, y = {
            'Training': (self.Xtr, self.Ytr),
            'Validation': (self.Xdev, self.Ydev),
            'Testing': (self.Xte, self.Yte)
        }[split]
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        print(f'{split} set loss: {loss}')

    # Evaluate loss
    def evaluate_loss(self):
        # Put layers into eval mode (needed for batchnorm especially)
        for layer in self.model.layers:
            layer.training = False
        # Evaluate training, validation, and testing loss
        self.split_loss('Training')
        self.split_loss('Validation')
        self.split_loss('Testing')

    # Sampling from the model
    def sample(self, num_samples):
        samples = []
        for _ in range(num_samples):

            out = []
            context = [0] * self.block_size
            while True:
                # Forward pass
                logits = self.model(torch.tensor([context])) # (1, block_size, d)
                probs = F.softmax(logits, dim = 1)
                # Sample
                ix = torch.multinomial(probs, num_samples = 1).item()
                # Shift the context windows and track the samples
                context = context[1:] + [ix]
                out.append(ix)
                # If we sample a special '.' token, break
                if ix == 0:
                    break
            
            samples.append(''.join(self.itos[i] for i in out))
        return samples

class App:

    def __init__(self, net):
        # Set network object
        self.net = net

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
                names = self.net.sample(num_input)
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

if __name__ == '__main__':
    # Device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not(os.path.exists('./network_data.pkl')):
        # Create network object
        net = Network('names.txt')
        # Save model data to file
        pickle.dump(net, open('./network_data.pkl', 'wb'))
    # Open saved network object from file
    net = pickle.load(open('./network_data.pkl', 'rb'))
    a = App(Network('names.txt'))