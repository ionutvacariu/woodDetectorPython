from PIL import Image, ImageStat
from PIL import ImageEnhance


def adjust_brightness(input_image, factor):
    image = Image.open(input_image)
    enhancer_object = ImageEnhance.Brightness(image)
    out = enhancer_object.enhance(factor)
    return out


def getBrightnessLevel(im_file):
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]