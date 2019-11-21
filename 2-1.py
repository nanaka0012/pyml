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
sample_dim = 10000

i, o = generate_random_data_sample(sample_dim, x_dim, 2)

# let 0 represent malignant/red and 1 represent benign/blue
colors = ['r' if label == 0 else 'b' for label in o[:,0]]

# plt.scatter(i[:,0], i[:,1], c=colors)
# plt.xlabel("Age (scaled)")
# plt.ylabel("Tumor size (in cm)")
# plt.show()

# w = np.zeros((x_dim, 2))
# b = np.ones(2)
w = np.random.rand(x_dim, 2)
b = np.random.rand(2)
print(w)
print(b)

z = []
for l in range(0, sample_dim):
    # print(np.dot(i[l,], w))
    y = softmax(np.dot(i[l,], w) + b)
    loss = cross_entropy_error(y, o[l,])

    for m in range(0, 2):
        grad = i[l,m] * (y - o[l,])
        w[m,] -= a * grad
    
    grad = y - o[l,]
    b -= a * grad

    print(loss)


plt.scatter(i[:,0], i[:,1], c=colors)
plt.plot([0, b[0]/w[0][1]], [b[1]/w[0][0], 0], c = 'g', lw = 3)
plt.xlabel("Age (scaled)")
plt.ylabel("Tumor size (in cm)")
plt.show()
