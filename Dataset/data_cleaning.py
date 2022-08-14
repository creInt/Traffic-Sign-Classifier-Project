from PIL import Image
import glob
import os


if __name__ == "__main__":
    data_path = "data\\PetImages"
    dirs = os.listdir(data_path)
    for dir in dirs:
        imgs = glob.glob(os.path.join(data_path, dir, "*.jpg"))
        for img in imgs:
            try:
                im = Image.open(img)
            except (AttributeError, Image.UnidentifiedImageError) as e:
                print(img, e)
                os.remove(img)
