%matplotlib inline

import itertools
import gc
import operator
import os
# import sys
# sys.path.append('./../source/')
import time
import zipfile
import cv2
import mahotas as mt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
# from multiprocessing import Pool
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from scipy.stats import itemfreq
from skimage import feature
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image, ImageStat
from tqdm import tqdm
from joblib import Parallel, delayed
from saliency import Saliency


seed = 777
np.random.seed(seed)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 300)

target_size = (224, 224)

def read_image(path):
    try:
        with zip_file.open(path) as p:
            img = load_img(p, grayscale=False, target_size=target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
    except:
        img = np.zeros((1, *target_size, 3))
    return img
    
def read_images(paths):    
    return np.vstack(Parallel(n_jobs=-1)([delayed(read_image)(path) for path in paths])).astype(np.float32)
    
# from https://github.com/mbeyeler/opencv-python-blueprints/blob/master/chapter5/saliency.py

class Saliency:
    """Generate saliency map from RGB images with the spectral residual method
        This class implements an algorithm that is based on the spectral
        residual approach (Hou & Zhang, 2007).
    """
    def __init__(self, img, use_numpy_fft=True, gauss_kernel=(5, 5)):
        """Constructor
            This method initializes the saliency algorithm.
            :param img: an RGB input image
            :param use_numpy_fft: flag whether to use NumPy's FFT (True) or
                                  OpenCV's FFT (False)
            :param gauss_kernel: Kernel size for Gaussian blur
        """
        self.use_numpy_fft = use_numpy_fft
        self.gauss_kernel = gauss_kernel
        self.frame_orig = img

        # downsample image for processing
        self.small_shape = (64, 64)
        self.frame_small = cv2.resize(img, self.small_shape[1::-1])

        # whether we need to do the math (True) or it has already
        # been done (False)
        self.need_saliency_map = True

    def get_saliency_map(self):
        """Returns a saliency map
            This method generates a saliency map for the image that was
            passed to the class constructor.
            :returns: grayscale saliency map
        """
        if self.need_saliency_map:
            # haven't calculated saliency map for this image yet
            num_channels = 1
            if len(self.frame_orig.shape) == 2:
                # single channel
                sal = self._get_channel_sal_magn(self.frame_small)
            else:
                # multiple channels: consider each channel independently
                sal = np.zeros_like(self.frame_small).astype(np.float32)
                for c in range(self.frame_small.shape[2]):
                    small = self.frame_small[:, :, c]
                    sal[:, :, c] = self._get_channel_sal_magn(small)

                # overall saliency: channel mean
                sal = np.mean(sal, 2)

            # postprocess: blur, square, and normalize
            if self.gauss_kernel is not None:
                sal = cv2.GaussianBlur(sal, self.gauss_kernel, sigmaX=8,
                                       sigmaY=0)
            sal = sal**2
            sal = np.float32(sal)/np.max(sal)

            # scale up
            sal = cv2.resize(sal, self.frame_orig.shape[1::-1])

            # store a copy so we do the work only once per frame
            self.saliencyMap = sal
            self.need_saliency_map = False

        return self.saliencyMap

    def _get_channel_sal_magn(self, channel):
        """Returns the log-magnitude of the Fourier spectrum
            This method calculates the log-magnitude of the Fourier spectrum
            of a single-channel image. This image could be a regular grayscale
            image, or a single color channel of an RGB image.
            :param channel: single-channel input image
            :returns: log-magnitude of Fourier spectrum
        """
        # do FFT and get log-spectrum
        if self.use_numpy_fft:
            img_dft = np.fft.fft2(channel)
            magnitude, angle = cv2.cartToPolar(np.real(img_dft),
                                               np.imag(img_dft))
        else:
            img_dft = cv2.dft(np.float32(channel),
                              flags=cv2.DFT_COMPLEX_OUTPUT)
            magnitude, angle = cv2.cartToPolar(img_dft[:, :, 0],
                                               img_dft[:, :, 1])

        # get log amplitude
        log_ampl = np.log10(magnitude.clip(min=1e-9))

        # blur log amplitude with avg filter
        log_ampl_blur = cv2.blur(log_ampl, (3, 3))

        # residual
        residual = np.exp(log_ampl - log_ampl_blur)

        # back to cartesian frequency domain
        if self.use_numpy_fft:
            real_part, imag_part = cv2.polarToCart(residual, angle)
            img_combined = np.fft.ifft2(real_part + 1j*imag_part)
            magnitude, _ = cv2.cartToPolar(np.real(img_combined),
                                           np.imag(img_combined))
        else:
            img_dft[:, :, 0], img_dft[:, :, 1] = cv2.polarToCart(residual,
                                                                 angle)
            img_combined = cv2.idft(img_dft)
            magnitude, _ = cv2.cartToPolar(img_combined[:, :, 0],
                                           img_combined[:, :, 1])

        return magnitude

    def calc_magnitude_spectrum(self):
        """Plots the magnitude spectrum
            This method calculates the magnitude spectrum of the image passed
            to the class constructor.
            :returns: magnitude spectrum
        """
        # convert the frame to grayscale if necessary
        if len(self.frame_orig.shape) > 2:
            frame = cv2.cvtColor(self.frame_orig, cv2.COLOR_BGR2GRAY)
        else:
            frame = self.frame_orig

        # expand the image to an optimal size for FFT
        rows, cols = self.frame_orig.shape[:2]
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        frame = cv2.copyMakeBorder(frame, 0, ncols-cols, 0, nrows-rows,
                                   cv2.BORDER_CONSTANT, value=0)

        # do FFT and get log-spectrum
        img_dft = np.fft.fft2(frame)
        spectrum = np.log10(np.abs(np.fft.fftshift(img_dft)))

        # return for plotting
        return 255*spectrum/np.max(spectrum)

    def plot_power_spectrum(self):
        """Plots the power spectrum
            This method plots the power spectrum of the image passed to
            the class constructor.
            :returns: power spectrum
        """
        # convert the frame to grayscale if necessary
        if len(self.frame_orig.shape) > 2:
            frame = cv2.cvtColor(self.frame_orig, cv2.COLOR_BGR2GRAY)
        else:
            frame = self.frame_orig

        # expand the image to an optimal size for FFT
        rows, cols = self.frame_orig.shape[:2]
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        frame = cv2.copyMakeBorder(frame, 0, ncols - cols, 0, nrows - rows,
                                   cv2.BORDER_CONSTANT, value=0)

        # do FFT and get log-spectrum
        if self.use_numpy_fft:
            img_dft = np.fft.fft2(frame)
            spectrum = np.log10(np.real(np.abs(img_dft))**2)
        else:
            img_dft = cv2.dft(np.float32(frame), flags=cv2.DFT_COMPLEX_OUTPUT)
            spectrum = np.log10(img_dft[:, :, 0]**2+img_dft[:, :, 1]**2)

        # radial average
        L = max(frame.shape)
        freqs = np.fft.fftfreq(L)[:L/2]
        dists = np.sqrt(np.fft.fftfreq(frame.shape[0])[:, np.newaxis]**2 +
                        np.fft.fftfreq(frame.shape[1])**2)
        dcount = np.histogram(dists.ravel(), bins=freqs)[0]
        histo, bins = np.histogram(dists.ravel(), bins=freqs,
                                   weights=spectrum.ravel())

        centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(centers, histo/dcount)
        plt.xlabel('frequency')
        plt.ylabel('log-spectrum')
        plt.show()

    def get_proto_objects_map(self, use_otsu=True):
        """Returns the proto-objects map of an RGB image
            This method generates a proto-objects map of an RGB image.
            Proto-objects are saliency hot spots, generated by thresholding
            the saliency map.
            :param use_otsu: flag whether to use Otsu thresholding (True) or
                             a hardcoded threshold value (False)
            :returns: proto-objects map
        """
        saliency = self.get_saliency_map()

        if use_otsu:
            _, img_objects = cv2.threshold(np.uint8(saliency*255), 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh = np.mean(saliency)*255*3
            _, img_objects = cv2.threshold(np.uint8(saliency*255), thresh, 255,
                                           cv2.THRESH_BINARY)
        return img_objects
        
 # batch_size = 3000

# def predict(dataframe, preprocess_model, model, batch_size):
#     index = np.arange(len(dataframe))
#     n_iter = len(dataframe)//batch_size + 1
#     predicts = []
#     for i in tqdm(range(n_iter)):
#         if i == n_iter - 1:
#             idx = index[i*batch_size:]
#         else:
#             idx = index[i*batch_size:(i+1)*batch_size]
#         img = read_images(dataframe['image'].iloc[idx])
#         img = preprocess_model.preprocess_input(img)
#         predicts.append(model.predict(img))
#         del img; gc.collect()
#     return np.vstack(predicts)

# resnet_model = resnet50.ResNet50(weights='imagenet')
# pred_resnet = predict(train, resnet50, resnet_model, batch_size)
# np.save('./../input/pred_resnet.npy', pred_resnet)

# xception_model = xception.Xception(weights='imagenet')
# pred_xception = predict(train, xception, xception_model, batch_size)
# np.save('./../input/pred_xception.npy', pred_xception)

# def read_image(path):
#     try:
#         img = image.load_img(path, grayscale=False)
#         img = image.img_to_array(img)
#         return img.shape
#     except:
#         return (0, 0, 0)
    
# def read_images(image_path):    
#     return np.stack(Parallel(n_jobs=-1)([delayed(read_image)(path) for path in image_path])).astype(np.float32)

# batch_size = 3000

# def predict(dataframe, batch_size):
#     index = np.arange(len(dataframe))
#     n_iter = len(dataframe)//batch_size + 1
#     predicts = []
#     for i in tqdm(range(n_iter)):
#         if i == n_iter - 1:
#             idx = index[i*batch_size:]
#         else:
#             idx = index[i*batch_size:(i+1)*batch_size]
#         img = read_images(dataframe['image'].iloc[idx])
#         predicts.append(img)
#         del img; gc.collect()
#     return np.vstack(predicts)

# img_shape = predict(train, batch_size)
# np.save('./../input/image_shape.npy', img_shape)

train_or_test = 'test'

if train_or_test == 'train':
    files = pd.read_feather("../input/train.ftr")['image']
    zip_path = './../input/train_jpg.zip'
    save_path = './../input/train_image_preprocess_{}.npy'
elif train_or_test == 'test':
    files = pd.read_feather("../input/test.ftr")['image']
    zip_path = './../input/test_jpg.zip'
    save_path = './../input/test_image_preprocess_{}.npy'

zip_file = zipfile.ZipFile(zip_path)
filenames = zip_file.namelist()[1:] # exclude the initial directory listing
print(len(filenames))

# from https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality

def get_dullness(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse=True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = light_shade/shade_count
    dark_percent = dark_shade/shade_count
    return np.array([light_percent, dark_percent])
    
def get_average_pixel_width(img):  
    img_array = np.array(img.convert(mode='L'))
    edges_sigma1 = feature.canny(img_array, sigma=3)
    apw = np.sum(edges_sigma1)/img.size[0]/img.size[1]
    return [apw]
    
def get_dominant_color(img):
#     img = np.array(img)
    pixels = img.reshape((-1, 3)).astype('float32')

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    dominant_color = palette[np.argmax(np.unique(labels))]
    
    return (dominant_color/255).squeeze()
    
def get_average_color(img):
#     img = np.asarray(img)
    
    return img.reshape(-1, 3).mean(axis=0)/255
    
def get_blurrness_score(gray):
#     img = np.ndarray(img)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.array([cv2.Laplacian(gray, cv2.CV_64F).var()])
    
def get_imstats(img, cv_img):
    img_stats = ImageStat.Stat(img)
#     img = np.asarray(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    return np.array([*img_stats.sum, 
                             *img_stats.mean, 
                             *img_stats.rms, 
                             *img_stats.var, 
                             *img_stats.stddev, 
                             *cv2.calcHist(cv_img, [0], None, [256], [0, 256]).flatten()
                            ]).flatten()
                            
 def get_stats(img):
    img = img.reshape(-1, 3)
    return np.array([
                        img.mean(axis=0),
                        img.std(axis=0),
                        img.min(axis=0),
                        img.max(axis=0),
                    ])
                    
def get_brightness_and_saturation_and_contrast(img):
#     img = np.array(img)
    return np.array([get_stats(cv2.cvtColor(img, cv2.COLOR_BGR2YUV)), get_stats(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))]).flatten()
    
def get_colorfullness(img):
#     img = np.array(img)
    (B, G, R) = cv2.split(img)
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rb_mean, rb_std) = (np.mean(rg), np.std(rg))
    (yb_mean, yb_std) = (np.mean(yb), np.std(yb))

    std_root = np.sqrt((rb_std ** 2) + (yb_std ** 2))
    mean_root = np.sqrt((rb_mean ** 2) + (yb_mean ** 2))

    return np.array([std_root + (0.3 * mean_root)])
    
# def get_texture(img):
#     return mt.features.haralick(img).mean(axis=0)

# def get_rgb_simplicity(img):
# #     img = np.array(img)
#     b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
#     hist_r, bins_r = np.histogram(r.ravel(), 256, [0, 256])
#     hist_g, bins_g = np.histogram(g.ravel(), 256, [0, 256])
#     hist_b, bins_b = np.histogram(b.ravel(), 256, [0, 256])
#     return hist_r, hist_g, hist_b

fast = cv2.FastFeatureDetector_create()

def get_interest_points(img):
    kp = fast.detect(img, None)
    return [len(kp)]
    
def get_saliency_features(img):
#     img = np.array(img)
    saliency = Saliency(img).get_saliency_map()
    binary_saliency = np.where(saliency>3*saliency.mean(), 1, 0).astype('uint8')
    prop_background = 1 - binary_saliency.mean()
    
    n_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_saliency)
    sizes = stats[:, -1]
    countours = stats[:, :-1]
    max_component_size = max(sizes)/img.shape[0]/img.shape[1]
    bbox = countours[np.argmax(sizes)]
    max_component_avg_saliency = saliency[bbox[1]:bbox[3], bbox[0]:bbox[2]].mean()
    s = centroids/[img.shape[0], img.shape[1]]
    dist = euclidean_distances(s)
    mean_dist = dist[~np.eye(dist.shape[0], dtype=bool)].mean()
    max_component_centorid = s[np.argmax(sizes)]
    min_dist_from_third_points = min(
        np.linalg.norm(max_component_centorid - [1/3, 1/3]),
        np.linalg.norm(max_component_centorid - [1/3, 2/3]),
        np.linalg.norm(max_component_centorid - [2/3, 1/3]),
        np.linalg.norm(max_component_centorid - [2/3, 2/3]),
    )
    dist_from_center = np.linalg.norm(s - [0.5, 0.5], axis=1)
    mean_dist_from_center = dist_from_center.mean()
    sum_dist_from_center = dist_from_center.sum()
    
    return np.array([prop_background, n_components, max_component_size, max_component_avg_saliency, mean_dist, min_dist_from_third_points, mean_dist_from_center, sum_dist_from_center])
    
cascades_dir = os.path.normpath(os.path.join(cv2.__file__, '..', 'data'))
cascade = cv2.CascadeClassifier(os.path.join(cascades_dir, 'haarcascade_frontalface_alt2.xml'))

def get_face_features(gray):
#     img = np.array(img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facerect = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=1, minSize=(20, 20))
    facearea = 0
    if len(facerect) > 0:
        for rect in facerect:
            x, y, w, h = rect
            facearea += w*h

    return np.array([len(facerect), facearea])
    
def get_image_size(img):
#     filename = images_path + filename
#     img_size = IMG.open(filename).size
    return img.size
    
def get_all_image_features(img):
    cv_img = np.array(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return np.hstack([
        get_image_size(img),
        get_dullness(img),
        get_average_pixel_width(img),
        get_average_color(cv_img),
        get_blurrness_score(cv_img),
        get_imstats(img, cv_img),
        get_brightness_and_saturation_and_contrast(cv_img),
        get_colorfullness(cv_img),
        get_interest_points(cv_img),
        get_saliency_features(cv_img),
        get_face_features(gray),
    ])
    
def read_img(name):
    if train_or_test == 'train':
        img_path = 'data/competition_files/train_jpg/{}.jpg'.format(name)
    elif train_or_test == 'test':
        img_path = 'data/competition_files/test_jpg/{}.jpg'.format(name)
    try:
        with zip_file.open(img_path) as img:
            res = Image.open(img)
        return res
    except:
        return 0
        
img = read_img(files[0])
cv_img = np.array(img)
gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

print(np.array(get_image_size(img)).shape)
print(np.array(get_dullness(img)).shape)
print(np.array(get_average_pixel_width(img)).shape)
print(np.array(get_average_color(cv_img)).shape)
print(np.array(get_blurrness_score(cv_img)).shape)
print(np.array(get_imstats(img, cv_img)).shape)
print(np.array(get_brightness_and_saturation_and_contrast(cv_img)).shape)
print(np.array(get_colorfullness(cv_img)).shape)
print(np.array(get_interest_points(cv_img)).shape)
print(np.array(get_saliency_features(cv_img)).shape)
print(np.array(get_face_features(gray)).shape)

len_process = len(get_all_image_features(read_img(files[0])))

def _process(name):
    try:
        return get_all_image_features(read_img(name))
    except:
        return np.zeros(len_process)    
    
def parallel_process(names):    
    return np.vstack(Parallel(n_jobs=1)([delayed(_process)(name) for name in names])).astype('float32')

def process(names):    
    return np.vstack([_process(name) for name in names]).astype('float32')
    
# split files for parallel process

# train_or_test = sys.argv[1]
# n_worker = sys.argv[2]
n_worker = 0

len_files = len(files)
file_size = len_files//5 + 1

files = files[n_worker*file_size:min((n_worker+1)*file_size, len_files)]

batch_size = 1000

res = np.zeros((len(files), len_process))
n_iter = len(files)//batch_size + 1
for i in tqdm(range(n_iter)):
    if i != n_iter-1:
        res[i*batch_size:(i+1)*batch_size, :] = process(files.iloc[i*batch_size:(i+1)*batch_size])
    else:
        res[i*batch_size:, :] = process(files.iloc[i*batch_size:])
print(res.shape)

np.save(save_path.format(n_worker), res)

# nima
train_or_test = 'train'

if train_or_test == 'train':
    files = pd.read_feather("../input/train.ftr")['image']
    files = files.apply(lambda x: 'data/competition_files/train_jpg/{}.jpg'.format(x))
    zip_path = './../input/train_jpg.zip'
    save_path = './../input/train_image_preprocess_nima.npy'
elif train_or_test == 'test':
    files = pd.read_feather("../input/test.ftr")['image']
    files = files.apply(lambda x: 'data/competition_files/test_jpg/{}.jpg'.format(x))
    zip_path = './../input/test_jpg.zip'
    save_path = './../input/test_image_preprocess_nima.npy'

zip_file = zipfile.ZipFile(zip_path)
filenames = zip_file.namelist()[1:] # exclude the initial directory listing
print(len(filenames))

# calculate mean score for AVA dataset
def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si, axis=1)
    return mean

# calculate standard deviation of scores for AVA dataset
def std_score(scores):
    si = np.arange(1, 11, 1).reshape(1, -1)
    mean = mean_score(scores).reshape(-1, 1)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores, axis=1))
    return std
    
target_size = (224, 224) 

base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
x = Dropout(0.75)(base_model.output)
x = Dense(10, activation='softmax')(x)

model = Model(base_model.input, x)
model.load_weights('./../data/weights/mobilenet_weights.h5')

batch_size = 500

res = np.zeros((len(files), 2))
n_iter = len(files)//batch_size + 1
for i in tqdm(range(n_iter)):
    if i != n_iter-1:
        score = model.predict(read_images(files.iloc[i*batch_size:(i+1)*batch_size]), batch_size=batch_size)
        mean = mean_score(score)
        std = std_score(score)
        res[i*batch_size:(i+1)*batch_size, :] = np.vstack((mean, std)).T
    else:
        score = model.predict(read_images(files.iloc[i*batch_size:]), batch_size=batch_size)
        mean = mean_score(score)
        std = std_score(score)
        res[i*batch_size:, :] = np.vstack((mean, std)).T

print(res.shape)

np.save(save_path, res)

train_or_test = 'test'

if train_or_test == 'train':
    files = pd.read_feather("../input/train.ftr")['image']
    files = files.apply(lambda x: 'data/competition_files/train_jpg/{}.jpg'.format(x))
    zip_path = './../input/train_jpg.zip'
    save_path = './../input/train_image_preprocess_nima.npy'
elif train_or_test == 'test':
    files = pd.read_feather("../input/test.ftr")['image']
    files = files.apply(lambda x: 'data/competition_files/test_jpg/{}.jpg'.format(x))
    zip_path = './../input/test_jpg.zip'
    save_path = './../input/test_image_preprocess_nima.npy'

zip_file = zipfile.ZipFile(zip_path)
filenames = zip_file.namelist()[1:] # exclude the initial directory listing
print(len(filenames))

batch_size = 500

res = np.zeros((len(files), 2))
n_iter = len(files)//batch_size + 1
for i in tqdm(range(n_iter)):
    if i != n_iter-1:
        score = model.predict(read_images(files.iloc[i*batch_size:(i+1)*batch_size]), batch_size=batch_size)
        mean = mean_score(score)
        std = std_score(score)
        res[i*batch_size:(i+1)*batch_size, :] = np.vstack((mean, std)).T
    else:
        score = model.predict(read_images(files.iloc[i*batch_size:]), batch_size=batch_size)
        mean = mean_score(score)
        std = std_score(score)
        res[i*batch_size:, :] = np.vstack((mean, std)).T

print(res.shape)

np.save(save_path, res)
