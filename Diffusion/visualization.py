import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load_folder', default='outputs')
parser.add_argument('--save_folder', default='images')
parser.add_argument('--npz_filename', default='condTrue_16x64x64x3.npz')
args = parser.parse_args()

def main():
    # load from your saved npz
    path = f'{args.load_folder}/{args.npz_filename}'
    data = np.load(path)
    print(data.files)
    print(data['arr_0'].shape)

    # arr_0 is for the images, and arr_0 is for the class index
    for i, img in enumerate(data['arr_0']):
        plt.imshow(img)
        plt.savefig(f"{args.save_folder}/image_sample0_{i}.png")
    for i, data in enumerate(data['arr_1']):
        print(f'figure {i}; class index:{data}')

if __name__=='__main__':
    main()