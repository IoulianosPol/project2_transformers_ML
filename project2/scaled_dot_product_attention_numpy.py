import numpy as np
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

E = np.load("E.npy")

# Self-Attention matrices
W_Q_self = np.load("W_Q_self.npy")
W_K_self = np.load("W_K_self.npy")
W_V_self = np.load("W_V_self.npy")

#Cross-Attention matrices
W_Q_cross = np.load("W_Q_cross.npy")
W_K_cross = np.load("W_K_cross.npy")
W_V_cross = np.load("W_V_cross.npy")

#Vocabulary
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

print("E:", E.shape)

print("W_Q_self:", W_Q_self.shape)
print("W_K_self:", W_K_self.shape)
print("W_V_self:", W_V_self.shape)

print("W_Q_cross:", W_Q_cross.shape)
print("W_K_cross:", W_K_cross.shape)
print("W_V_cross:", W_V_cross.shape)

def scaled_dot_product_attention(Q, K, V, name=""):
    # Dimension of key/query vectors
    dk = Q.shape[1]

    print(f"\n{name}")
    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)
    # Compute raw attention scores (QK^T) and scale by sqrt(dk)
    scores = (Q @ K.T) / np.sqrt(dk)
    print("scores shape:", scores.shape)

    # Numerical stability trick: subtract max from each row
    scores = scores - np.max(scores, axis=1, keepdims=True)

    # Apply softmax row-wise
    exp_scores = np.exp(scores)
    A = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    print("A shape:", A.shape)
    # Compute weighted sum of values
    Y = A @ V
    print("Y shape:", Y.shape)
    return A, Y

english_tokens = ["Maria", "isn't", "here"]
# Embeddings
X_en = np.array([E[vocab[t]] for t in english_tokens])
# Projections
Q_self = X_en @ W_Q_self
K_self = X_en @ W_K_self
V_self = X_en @ W_V_self
# Attention
A_self, Y_self = scaled_dot_product_attention(
    Q_self, K_self, V_self,
    name="self-attention"
)

greek_tokens = ["Δεν", "είναι", "εδώ", "η", "Μαρία"]

# Embeddings
X_gr = np.array([E[vocab[t]] for t in greek_tokens])
X_en = np.array([E[vocab[t]] for t in english_tokens])

# Cross-attention:
# Q from Greek
Q_cross = X_gr @ W_Q_cross

# K,V from English
K_cross = X_en @ W_K_cross
V_cross = X_en @ W_V_cross

# Attention
A_cross, Y_cross = scaled_dot_product_attention(
    Q_cross, K_cross, V_cross,
    name="CROSS-ATTENTION (GR → EN)"
)



# Convert NumPy arrays to torch tensors (float32)
Q_t = torch.tensor(Q_self, dtype=torch.float32)
K_t = torch.tensor(K_self, dtype=torch.float32)
V_t = torch.tensor(V_self, dtype=torch.float32)

# dk = dimension of queries/keys
d_k = Q_self.shape[1]

# PyTorch implementation
Y_torch = F.scaled_dot_product_attention(
    Q_t,
    K_t,
    V_t,
    scale=1.0 / np.sqrt(d_k)
)

# Convert back to NumPy
Y_torch_np = Y_torch.detach().numpy()

# Compare outputs
print("Self-attention agreement:",
      np.allclose(Y_self, Y_torch_np, atol=1e-5))
Q_t = torch.tensor(Q_cross, dtype=torch.float32)
K_t = torch.tensor(K_cross, dtype=torch.float32)
V_t = torch.tensor(V_cross, dtype=torch.float32)

d_k = Q_cross.shape[1]

Y_torch = F.scaled_dot_product_attention(
    Q_t,
    K_t,
    V_t,
    scale=1.0 / np.sqrt(d_k)
)

Y_torch_np = Y_torch.detach().numpy()

print("Cross-attention agreement:",
      np.allclose(Y_cross, Y_torch_np, atol=1e-5))




plt.figure(figsize=(5, 4))
plt.imshow(A_self, cmap="viridis", aspect="auto")

plt.colorbar(label="Attention weight")

plt.xticks(range(len(english_tokens)), english_tokens)
plt.yticks(range(len(english_tokens)), english_tokens)

plt.xlabel("Keys")
plt.ylabel("Queries")
plt.title("Self-Attention Heatmap (English)")

plt.show()


plt.figure(figsize=(5, 4))
plt.imshow(A_cross, cmap="viridis", aspect="auto")

plt.colorbar(label="Attention weight")

plt.xticks(range(len(english_tokens)), english_tokens)
plt.yticks(range(len(greek_tokens)), greek_tokens)

plt.xlabel("English Keys")
plt.ylabel("Greek Queries")
plt.title("Cross-Attention Heatmap (Greek → English)")

plt.show()