import numpy as np
from scipy import spatial
from mnist_loader import MNIST
from tqdm import tqdm

def load_mnist():
    mnist_data = MNIST("./mnist")
    X_train, Y_train = map(np.array, mnist_data.load_training())
    X_test, Y_test = map(np.array, mnist_data.load_testing())
    return X_train, Y_train, X_test, Y_test

x_train, y_train, x_test, y_test = load_mnist()
print(x_train.shape, y_train.shape)

hdv_dim = 10000

# create hdv of specified size
def hdv(size):
    hdv = np.random.randint(2, size=size)
    hdv = np.where(hdv==1, hdv, -1)
    return(hdv)

def binding(hdv1, hdv2):
    return hdv1 * hdv2

greyscale_hdv = hdv((26, hdv_dim)) # quantized to 26 grayscale levels, this seems to slightly improve the accuracy.
position_hdv = hdv((784, hdv_dim))

def embedd_image(image):
    img_hdv = np.zeros(hdv_dim, dtype=int)
    for pixel_idx in range(0, image.shape[0]):
        img_hdv += position_hdv[pixel_idx] * greyscale_hdv[int(image[pixel_idx]//10)]
    img_hdv = np.sign(img_hdv)
    return img_hdv

# embedd all mnist examples into hdv
pbar = tqdm(total=x_train.shape[0])
mnist_train_hdv = []
for idx in range(x_train.shape[0]):
    hdv_emb = embedd_image(x_train[idx])
    mnist_train_hdv.append(hdv_emb)
    pbar.update()
pbar.close()

# create bundle of all examples for specific digit. This will be used for the similarity comparison.
mnist_train_hdv_bundle = np.zeros((10, hdv_dim), dtype=int)
for idx, img_hdv in enumerate(mnist_train_hdv):
    mnist_train_hdv_bundle[y_train[idx]] += img_hdv
mnist_train_hdv_bundle = np.sign(mnist_train_hdv_bundle)

def cos_sim(hdv1, hdv2):
    cos_sim = -1. * (float(spatial.distance.cosine(hdv1, hdv2)) -1)
    return cos_sim

# iterate through all test images and compare similarity to bundles of specific digit.
# Highest similarity is used a precidction
pbar = tqdm(total=x_test.shape[0])
correct, total = 0.0, 0.0
for idx in range(x_test.shape[0]):
    hdv_emb = embedd_image(x_test[idx])
    sims = []
    for digit in range(0, 10):
        sims.append(cos_sim(hdv_emb, mnist_train_hdv_bundle[digit]))

    correct += sims.index(max(sims)) == y_test[idx]
    total += 1.0
    pbar.set_description("%.2f" % ((float(correct)/float(total))*100))
    pbar.update()
pbar.close()

# accouracy arround 80%
print('Accuracy: ', float(correct)/float(total))
