import matplotlib.pyplot as plt
import math
import torchvision.transforms as transforms
import matplotlib.image as mpimg
import torch


def make_grid(samples, class_names, num_img=10, keys=["image", "label"], columns=5):
    invTrans = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )
    if samples[keys[0]].shape[0] < num_img:
        num_img = samples[keys[0]].shape[0]
    rows = int(math.ceil(num_img / columns))
    f, axarr = plt.subplots(rows, columns, figsize=(10, 10))
    k = 0
    for i in range(rows):
        for j in range(columns):
            if k >= num_img:
                break

            axarr[i, j].imshow(samples[keys[0]][k].permute(1, 2, 0).to(torch.int))
            img = samples[keys[0]][k].permute(1, 2, 0).to(torch.int)  # TODO
            img = img.detach().numpy()

            axarr[i, j].title.set_text(class_names[samples[keys[1]][k].int().item()])
            axarr[i, j].axis("off")
            k += 1
    plt.show()
    return f, axarr
