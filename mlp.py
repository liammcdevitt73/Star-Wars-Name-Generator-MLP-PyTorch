import torch
import torch.nn.functional as F
import random
import pickle

# Building a multi-layered perceptron (MLP) character-level language model
class MLP():

    def __init__(self, text_file):

        # Read in words from a text file
        self.words = open(text_file, 'r').read().splitlines()

        # Context length: how many characters do we take to predict the next one?
        self.block_size = 3 

        # Build the vocabulary of character mappings to/from integers
        self.chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s:i+1 for i, s in enumerate(self.chars)} # String -> Integer
        self.stoi['.'] = 0
        self.itos = {i:s for s,i in self.stoi.items()} # Integer -> String

        # Number generator
        g = torch.Generator().manual_seed(0)

        # Training split (80%), dev/validation split (10%), test split (10%)
        self.splitting()

        # Initialize network
        self.init_network(g)

        # Train network
        self.training()

        # Losses
        self.training_loss()
        self.validation_loss()
        self.testing_loss()

        # Sampling
        self.sample(10, g)

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
        print(X.shape, Y.shape)
        return X, Y
    
    # Split training, dev/validation, and testing dataset
    def splitting(self):
        random.seed(1)
        random.shuffle(self.words)
        n1 = int(0.8 * len(self.words))
        n2 = int(0.9 * len(self.words))
        self.Xtr, self.Ytr = self.build_dataset(self.words[:n1])
        self.Xdev, self.Ydev = self.build_dataset(self.words[n1:n2])
        self.Xte, self.Yte = self.build_dataset(self.words[n2:])

    # Initialize network
    def init_network(self, g):
        self.C = torch.randn((27, 10), generator = g)
        self.W1 = torch.randn((30, 300), generator = g)
        self.b1 = torch.randn(300, generator = g)
        self.W2 = torch.randn((300, 27), generator = g)
        self.b2 = torch.randn(27, generator = g)
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]

        # Set parameters that need gradient calculations for
        for p in self.parameters:
            p.requires_grad = True


    # Train network
    def training(self):
        for i in range(200000):

            # Minibatch construct
            ix = torch.randint(0, self.Xtr.shape[0], (64,))

            # Forward pass
            emb = self.C[self.Xtr[ix]]
            h = torch.tanh(emb.view(-1, 30) @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            loss = F.cross_entropy(logits, self.Ytr[ix])

            # Backward pass
            for p in self.parameters:
                p.grad = None
            loss.backward()

            # Update
            lr = 0.1 if i < 100000 else 0.01
            for p in self.parameters:
                p.data += -lr * p.grad

    # Training -> loss
    def training_loss(self):
        emb = self.C[self.Xtr]
        h = torch.tanh(emb.view(-1, 30) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        loss = F.cross_entropy(logits, self.Ytr)
        print(f'Training set loss: {loss}')

    # Validation -> loss
    def validation_loss(self):
        emb = self.C[self.Xdev]
        h = torch.tanh(emb.view(-1, 30) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        loss = F.cross_entropy(logits, self.Ydev)
        print(f'Validation set loss: {loss}')

    # Testing -> loss
    def testing_loss(self):
        emb = self.C[self.Xte]
        h = torch.tanh(emb.view(-1, 30) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        loss = F.cross_entropy(logits, self.Yte)
        print(f'Testing set loss: {loss}')

    # Sampling from the model
    def sample(self, num_samples, g):
        samples = []
        for _ in range(num_samples):

            out = []
            context = [0] * self.block_size
            while True:
                emb = self.C[torch.tensor([context])] # (1, block_size, d)
                h = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = F.softmax(logits, dim = 1)
                ix = torch.multinomial(probs, num_samples = 1, generator = g).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
            
            samples.append(''.join(self.itos[i] for i in out))
        return samples

if __name__ == '__main__':
    # Device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create MLP object
    mlp = MLP('names.txt')
    mlp_data = {
        'parameters' : mlp.parameters,
        'block_size' : mlp.block_size,
        'itos'       : mlp.itos
    }
    # Save model data to file
    pickle.dump(mlp_data, open('./mlp_data.pkl', 'wb'))