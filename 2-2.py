import numpy as np
import matplotlib.pyplot as plt

# Ensure that we always get the same results
np.random.seed(0)

# Helper function to generate a random data sample
def generate_random_data_sample(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy.
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim)+3) * (Y + 1)

    # Specify the data type to match the input variable used later in the tutorial
    # (default type is double)
    X = X.astype(np.float32)

    # convert class 0 into the vector "1 0 0",
    # class 1 into the vector "0 1 0", ...
    class_ind = [Y==class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

a = 0.5
x_dim = 2
sample_dim = 20000
out_dim = 2
minibatch_size = 10

i, o = generate_random_data_sample(sample_dim, x_dim, out_dim)

# let 0 represent malignant/red and 1 represent benign/blue
colors = ['r' if label == 0 else 'b' for label in o[:,0]]

# plt.scatter(i[:,0], i[:,1], c=colors)
# plt.xlabel("Age (scaled)")
# plt.ylabel("Tumor size (in cm)")
# plt.show()

# w = np.ones((x_dim, out_dim))
# b = np.ones(out_dim)
w = np.random.rand(x_dim, out_dim)
b = np.random.rand(out_dim)
print(w)
print(b)
z = []
for l in range(0, int(sample_dim / minibatch_size)):
    # print(np.dot(i[l,], w))
    begin = int(minibatch_size * l)
    end = int(minibatch_size * (l + 1))
    w_grad = np.zeros(2)
    b_grad = np.zeros(2)
    loss = 0
    for q in range(begin, end):
        y = softmax(np.dot(i[q,], w) + b)
        loss += cross_entropy_error(y, o[q,])

        for m in range(0, 2):
            w_grad += i[q,m] * (y - o[q,])
        
        b_grad += y - o[q,]
    
    for m in range(0, 2):
        w[m,] -= a * w_grad / minibatch_size
    
    b -= a * b_grad / minibatch_size
    loss /= minibatch_size

    print(loss)
    z.append(loss)

plt.plot(z)
plt.show()

plt.scatter(i[:,0], i[:,1], c=colors)
plt.plot([0, b[0]/w[0][1]], [b[1]/w[0][0], 0], c = 'g', lw = 3)
plt.xlabel("Age (scaled)")
plt.ylabel("Tumor size (in cm)")
plt.show()

