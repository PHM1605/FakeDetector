from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import os

def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename) # new Image with less quality
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    os.remove(resaved_filename)

    return ela_im

def parse_images_and_labels(files, label):
    X, Y = [], []
    for file in files:
        img = convert_to_ela_image(file, 90).resize((128, 128))
        X.append(np.array(img) / 255.0)
        Y.append([0, 1]) if label == 1 else Y.append([1, 0])
    return np.array(X), np.array(Y)

def parse_images(files):
    X = []
    for file in files:
        img = convert_to_ela_image(file, 90).resize((128, 128))
        X.append(np.array(img) / 255.0)
    return np.array(X)