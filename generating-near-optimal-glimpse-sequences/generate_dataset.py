import argparse
from os.path import join
import time
import threading
from PIL import Image
import numpy as np
import torch
from progress.bar import Bar
from math import ceil

from mea.image_sampling.stylegan import get_generator
from mea.config import GET_FAKE_CELEBHQ_IMAGE_PATH, IMG_DIM, IMAGE_GEN_BATCH_SIZE
from mea.utils import set_random_seed

parser = argparse.ArgumentParser(description='Generate fake CelebA-HQ dataset.')
parser.add_argument('--N0', type=int, default=0,
                    help='First index.')
parser.add_argument('--N', type=int,
                    help='Number of images to generate.')
args = parser.parse_args()

generator = get_generator(cuda=True)

bar = Bar('Generating', max=ceil(args.N//IMAGE_GEN_BATCH_SIZE))
assert args.N0 % IMAGE_GEN_BATCH_SIZE == 0, \
    f"N0 should be divisible by {IMAGE_GEN_BATCH_SIZE} to \
      be reproducible if we rerun this."
start = time.time()
for index in range(args.N0, args.N0+args.N, IMAGE_GEN_BATCH_SIZE):
    print(index)
    set_random_seed(index)  # images will be deterministic if batch size is constant and N0 always divisible by batch size

    with torch.no_grad():
        latents = torch.randn(IMAGE_GEN_BATCH_SIZE, 512).cuda()
        images = generator(latents)
        saving = time.time()

        latents = latents.cpu()
        images = images.cpu()

    def save_image(index, b):
        image = images[b]
        # move values into [0, 1]
        image = (image.clamp(-1, 1) + 1) / 2
        # convert to PIL image
        pil = Image.fromarray(np.uint8(255*image.numpy()).transpose(1, 2, 0))
        pil = pil.resize((IMG_DIM, IMG_DIM))
        pil.save(GET_FAKE_CELEBHQ_IMAGE_PATH(index+b))

    threads = []
    for b in range(IMAGE_GEN_BATCH_SIZE):
        t = threading.Thread(target=save_image, args=[index, b])
        t.start()
        threads.append(t)
    map(lambda t: t.join(), threads)

    bar.suffix = f"({index-args.N0}/{args.N}) | Time: {time.time()-start:.1f}s | "
    bar.next()
