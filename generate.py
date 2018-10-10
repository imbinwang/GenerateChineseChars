#coding=utf-8
import cv2
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import argparse
import os
import math
import io


def parseArgs():
    parser = argparse.ArgumentParser(description="Pleas input parameters for generating Chinese images.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--text_path', dest='text_path', default=None, required=True, help='path of file which contains Chinese characters')
    parser.add_argument('--font_dir', dest='font_dir', default=None, required=True, help='font directory of our target Chinese fonts')
    parser.add_argument('--font_size', dest='font_size', default=None, required=True, help='font size', type=int)
    parser.add_argument('--width', dest='width', default=None, required=True, help='image width', type=int)
    parser.add_argument('--height', dest='height', default=None, required=True, help='image height', type=int)
    parser.add_argument('--out_dir', dest='out_dir', default=None, required=True, help='dir to store the output images')
    parser.add_argument('--rotate_angle', dest='rotate_angle', default=0, required=False, help='max rotation degree 0-45', type=int)
    parser.add_argument('--rotate_step', dest='rotate_step', default=0, required=False, help='rotation step for the rotate angle', type=int)
    parser.add_argument('--need_aug', dest='need_aug', default=False, required=False, help='need data augmentation', action='store_true')
    args = parser.parse_args()
    return args


def getTexts(path):
    fr = io.open(path, 'r', encoding='utf8')
    texts_set = fr.readline()
    return texts_set

def getFonts(font_dir, font_size):
    fonts = []
    file_list = os.listdir(font_dir)
    for fname in file_list:
        file_path = os.path.join(font_dir, fname)
        if os.path.isfile(file_path) and os.path.splitext(fname)[1] in ['.ttf','.ttc','.otf']:
            fonts.append(ImageFont.truetype(file_path, font_size, 0))
    return fonts

def drawText(txt, font, font_size, width, height):
    image = np.ones(shape=(height, width),dtype=np.uint8)*255
    x = Image.fromarray(image)
    draw = ImageDraw.Draw(x)
    draw.text((width/2-font_size/2,height/2-font_size/2), txt, (0), font=font)
    p = np.array(x)
    return p

def rotate(image, angle):
    rows,cols = image.shape[:2]
    diag_length = int(math.floor(math.sqrt(rows*rows + cols*cols)))
    if diag_length%2==1:
        diag_length += 1
    big_img = np.ones(shape=(diag_length,diag_length),dtype=np.uint8)*255
    big_img[diag_length/2-rows/2:diag_length/2+rows/2, diag_length/2-cols/2:diag_length/2+cols/2] = image
    T = cv2.getRotationMatrix2D((diag_length/2,diag_length/2), angle, 1)
    rotated_big_img = cv2.warpAffine(big_img, T, (diag_length,diag_length))
    rotated_img = rotated_big_img[diag_length/2-rows/2:diag_length/2+rows/2, diag_length/2-cols/2:diag_length/2+cols/2]
    return rotated_img

def addNoise(image):
    rows,cols = image.shape[:2]
    noise_image = (image.astype(float)/255)+(np.random.random((rows,cols))*(np.random.random()*0.3))
    norm = (noise_image - noise_image.min())/(noise_image.max() - noise_image.min())
    norm  =(norm * 255).astype(np.uint8)
    return norm

def addErode(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    img = cv2.erode(image,kernel)
    return img

def addDilate(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    img = cv2.dilate(image,kernel)
    return img

def addGaussianBlur(image):
    img = cv2.GaussianBlur(image,(5,5),np.random.randint(1,10))
    return img



################################# main #################################
args = parseArgs()
texts = getTexts(args.text_path)
fonts = getFonts(args.font_dir, args.font_size)
rotate_angles = []
rotate_angle = 0
if args.rotate_angle < 0:
    rotate_angle = -args.rotate_angle
else:
    rotate_angle = args.rotate_angle
if rotate_angle > 45:
    rotate_angle = 45
if rotate_angle > 0 and rotate_angle <= 45:
    for i in range(0, rotate_angle+1, args.rotate_step):
        rotate_angles.append(i)
    for i in range(-rotate_angle, 0, args.rotate_step):
        rotate_angles.append(i)
for txt_id in range(len(texts)):
    txt = texts[txt_id]
    images = []
    for font in fonts:
        image = drawText(txt, font, args.font_size, args.width, args.height)
        if args.rotate_angle!=0:
            for angle in rotate_angles:
                rotated_img = rotate(image, angle)
                images.append(rotated_img)
        else:
            images.append(image)
    augmented_images = []
    if args.need_aug:
        for img in images:
            noised_img = addNoise(img)
            augmented_images.append(noised_img)
            noised_img2 = addNoise(img)
            augmented_images.append(noised_img2)
            blurred_img = addGaussianBlur(img)
            augmented_images.append(blurred_img)
            blurred_img2 = addGaussianBlur(img)
            augmented_images.append(blurred_img2)
            #eroded_img = addErode(img)
            #augmented_images.append(eroded_img)
            #dilated_img = addDilate(img)
            #augmented_images.append(dilated_img)
    images_dir = '{0}/{1:0>4}'.format(args.out_dir, txt_id)
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    img_id = 0
    for img in images:
        img_path = '{0}/{1:0>5}.png'.format(images_dir, img_id)
        cv2.imwrite(img_path,img)
        img_id += 1
    for img in augmented_images:
        img_path = '{0}/{1:0>5}.png'.format(images_dir, img_id)
        cv2.imwrite(img_path,img)
        img_id += 1