#!/usr/bin/env python
# coding: utf-8
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(y):
    return y * (1.0 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y*y

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs, output_type="sigmoid"):
        self.NI = n_inputs
        self.NH = n_hidden
        self.NO = n_outputs
        self.output_type = output_type
        
        self.W1 = np.random.uniform(-0.5, 0.5, (self.NH, self.NI + 1))
        self.W2 = np.random.uniform(-0.5, 0.5, (self.NO, self.NH + 1))
    
    def forward(self, I):
        self.last_input = I
        I_bias = np.append(I, 1.0)
        self.H = tanh(np.dot(self.W1, I_bias))
        H_bias = np.append(self.H, 1.0)
        
        net = np.dot(self.W2, H_bias)
        if self.output_type == "sigmoid":
            self.O = sigmoid(net)
        elif self.output_type == "linear":
            self.O = net
        elif self.output_type == "softmax":
            e = np.exp(net - np.max(net))
            self.O = e / np.sum(e)
        return self.O
    
    def backward(self, T, lr=0.1):
        T = np.array(T)
        H_bias = np.append(self.H, 1.0)
        I_bias = np.append(self.last_input, 1.0)
    
        if self.output_type == "sigmoid":
            # sigmoid + MSE
            delta_o = (T - self.O) * dsigmoid(self.O)
            delta_h = dtanh(self.H) * np.dot(self.W2[:, :-1].T, delta_o)
            self.W2 += lr * np.outer(delta_o, H_bias)
            self.W1 += lr * np.outer(delta_h, I_bias)
            return np.mean((T - self.O)**2)
    
        elif self.output_type == "linear":
            # linear + MSE
            delta_o = T - self.O
            delta_h = dtanh(self.H) * np.dot(self.W2[:, :-1].T, delta_o)
            self.W2 += lr * np.outer(delta_o, H_bias)
            self.W1 += lr * np.outer(delta_h, I_bias)
            return np.mean((T - self.O)**2)
    
        elif self.output_type == "softmax":
            # softmax + cross entropy
            delta_o = self.O - T
            delta_h = dtanh(self.H) * np.dot(self.W2[:, :-1].T, delta_o)
            self.W2 -= lr * np.outer(delta_o, H_bias)  
            self.W1 -= lr * np.outer(delta_h, I_bias)
            return -np.sum(T * np.log(self.O + 1e-9))
    
        else:
            raise ValueError("Invalid output type")


    def train(self, X, Y, epochs=1000, lr=0.1, error_file="training_error.txt"):
        with open(error_file, "w") as f:
            for epoch in range(epochs):
                total_err = 0
                for i in range(len(X)):
                    self.forward(X[i])
                    total_err += self.backward(Y[i], lr)
                
                avg_err = total_err / len(X)
                
                if epoch % 100 == 0 or epoch == epochs-1:
                    print(f"Epoch {epoch}, Error={avg_err:.6f}")
                
                f.write(f"{epoch},{avg_err:.6f}\n")

    
    def predict(self, X):
        return np.array([self.forward(x) for x in X])


X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

np.random.seed(1)
nn = MLP(2, 4, 1, output_type="sigmoid")

epochs = 5000
lr = 0.5

with open("xor_training_error.txt", "w") as err_file:
    for epoch in range(epochs):
        idx = np.random.permutation(len(X))
        total_err = 0
        for i in idx:
            nn.forward(X[i])
            total_err += nn.backward(Y[i], lr)

        avg_err = total_err / len(X)
        err_file.write(f"{epoch+1},{avg_err:.6f}\n")

        if (epoch+1) % 500 == 0:
            print(f"Epoch {epoch+1}, MSE={avg_err:.6f}")

for x, y in zip(X, Y):
    pred = nn.forward(x)
    print(x, "->", pred)


print("XOR FINAL PREDICTIONS\n")
for x, y in zip(X, Y):
    pred = nn.forward(x)
    correct = int(round(pred[0]) == y[0])
    print(f"Input={x}, Target={y[0]}, Output={pred[0]:.6f}, Correct={correct}")

X = np.random.uniform(-1,1,(500,4))
Y = np.sin(X[:,0]-X[:,1]+X[:,2]-X[:,3]).reshape(-1,1)

nn = MLP(4, 10, 1, output_type="linear")
nn.train(X[:400], Y[:400], epochs=1000, lr=0.1, error_file="sin_training_error.txt")


X_test = X[400:]
Y_test = Y[400:]
pred = nn.predict(X_test)

train_pred = nn.predict(X[:400])
train_mse = np.mean((train_pred - Y[:400])**2)
test_mse = np.mean((pred - Y_test)**2)
mae = np.mean(np.abs(pred - Y_test))

sin_res = np.sum((Y_test - pred)**2)
sin_tot = np.sum((Y_test - np.mean(Y_test))**2)
r2 = 1 - (sin_res / sin_tot)

print("SIN REGRESSION METRICS")
print("Train MSE:", train_mse)
print("Test MSE :", test_mse)
print("Mean Absolute Error:", mae)
print("R^2 score:", r2)

with open("sin_test_output.txt", "w") as f:
    for x, y_true, y_pred in zip(X_test, Y_test, pred):
        f.write(f"Input: {x}, Target: {y_true}, Output: {y_pred}\n")


plt.figure(figsize=(6,6))
plt.scatter(Y_test, pred, alpha=0.6)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs True (SIN Function)")
plt.grid(True)

min_val = min(Y_test.min(), pred.min())
max_val = max(Y_test.max(), pred.max())
plt.plot([min_val, max_val], [min_val, max_val])
plt.show()


residuals = pred - Y_test

plt.figure(figsize=(6,4))
plt.scatter(range(len(residuals)), residuals, alpha=0.6)
plt.axhline(0, color='black')
plt.title("Residuals (Prediction Error)")
plt.xlabel("Sample Index")
plt.ylabel("Prediction - True")
plt.grid(True)
plt.show()


data = []
with open("letter-recognition.data") as f:
    for line in f:
        parts = line.strip().split(",")
        letter = parts[0]
        features = np.array(list(map(float, parts[1:]))) 
        data.append((letter, features))

data = np.array(data, dtype=object)


def onehot(c):
    vec = np.zeros(26)
    vec[ord(c)-ord('A')] = 1
    return vec

X = np.array([d[1] for d in data], dtype=float)
Y = np.array([onehot(d[0]) for d in data])
X = X / 15.0

N = len(X)
indices = np.arange(N)
np.random.shuffle(indices)
X, Y = X[indices], Y[indices]

split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]


nn = MLP(16, 30, 26, output_type="softmax")
nn.train(X_train, Y_train, epochs=1000, lr=0.01, error_file="letter_training_error.txt")


pred = nn.predict(X_test)

correct = 0
for i in range(len(X_test)):
    pred_letter = chr(np.argmax(pred[i]) + ord('A'))
    true_letter = chr(np.argmax(Y_test[i]) + ord('A'))
    if pred_letter == true_letter:
        correct += 1

accuracy = correct / len(X_test)
print("Test accuracy:", accuracy)


# Convert one hot targets and softmax predictions
true_labels = np.argmax(Y_test, axis=1)
pred_labels = np.argmax(pred, axis=1)

conf_matrix = np.zeros((26, 26), dtype=int)
for t, p in zip(true_labels, pred_labels):
    conf_matrix[t, p] += 1

plt.figure(figsize=(8,7))
sns.heatmap(conf_matrix, annot=False, cmap="viridis")
plt.title("Confusion Matrix (A–Z)")
plt.xlabel("Predicted Letter")
plt.ylabel("True Letter")
plt.xticks(np.arange(26)+0.5, [chr(i+65) for i in range(26)], rotation=90)
plt.yticks(np.arange(26)+0.5, [chr(i+65) for i in range(26)], rotation=0)
plt.show()


pred_labels = np.argmax(pred, axis=1)
true_labels = np.argmax(Y_test, axis=1)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, pred_labels, labels=range(26), zero_division=0
)

letters = [chr(ord('A') + i) for i in range(26)]

df_metrics = pd.DataFrame({
    "Letter": letters,
    "Precision": precision,
    "Recall": recall,
    "F1": f1
})


overall_acc = accuracy_score(true_labels, pred_labels)
macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()

print("Overall accuracy:", overall_acc)
print("Macro Precision:", macro_precision)
print("Macro Recall:", macro_recall)
print("Macro F1:", macro_f1)


_, _, _, support = precision_recall_fscore_support(
    true_labels, pred_labels, labels=range(26)
)
df_metrics["Support"] = support
print(df_metrics)


plt.figure(figsize=(8,6))
plt.imshow(nn.W1[:, :-1], aspect='auto', cmap='bwr')  
plt.colorbar()
plt.title("Input to Hidden Weights (Heatmap)")
plt.xlabel("Input Feature Index")
plt.ylabel("Hidden Neuron Index")
plt.show()


data = np.loadtxt("xor_training_error.txt", delimiter=",")

epochs = data[:,0]
errors = data[:,1]

plt.figure()
plt.plot(epochs, errors)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("XOR Training Error Curve")
plt.grid(True)
plt.show()


train_data = np.loadtxt("sin_training_error.txt", delimiter=",")

epochs = train_data[:,0]
errors = train_data[:,1]

plt.figure(figsize=(6,4))
plt.plot(epochs, errors)
plt.title("Training Error Curve (SIN Function)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

train_err = np.loadtxt("letter_training_error.txt", delimiter=",")

epochs = train_err[:,0]
errors = train_err[:,1]

plt.figure(figsize=(7,4))
plt.plot(epochs, errors)
plt.title("Training Error Curve (Letter Recognition)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy")
plt.grid(True)
plt.show()





