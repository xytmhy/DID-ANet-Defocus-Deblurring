import os.path
import glob
from datasets.listdataset import ListDataset
from .util import split2list

'''
Using our dataset Light Field Defocus-Deblur
Thanks to the marvelous work of FlowNet.
'''

def make_dataset(dataset_dir, split):
    gt_dir = 'sharp'
    assert(os.path.isdir(os.path.join(dataset_dir, gt_dir)))
    img_dir = 'image'
    assert(os.path.isdir(os.path.join(dataset_dir, img_dir)))
    de_dir = 'filtered'
    assert (os.path.isdir(os.path.join(dataset_dir, de_dir)))

    images = []

    for gt_map in sorted(glob.glob(os.path.join(dataset_dir, gt_dir, '*.jpg'))):
        gt_map = os.path.relpath(gt_map, os.path.join(dataset_dir, gt_dir))

        scene_dir, filename = os.path.split(gt_map)
        no_ext_filename = os.path.splitext(filename)[0]
        prefix, frame_nb = no_ext_filename.split('_')
        img = os.path.join(img_dir, scene_dir, 'image_{}.jpg'.format(frame_nb))
        de = os.path.join(de_dir, scene_dir, 'defocus_filterd_{}.png'.format(frame_nb))
        gt_map = os.path.join(gt_dir, gt_map)
        if not (os.path.isfile(os.path.join(dataset_dir, img)) or os.path.isfile(os.path.join(dataset_dir, de))):
            continue
        images.append([[img, gt_map], de])
        # put the input and the deblur result together, since they are RGB images and the deblur estimation is single channel

    return split2list(images, split, default_split=0.87)

def defocus_de(root, transform=None, target_transform=None, co_transform=None, split=None):
    train_list, test_list = make_dataset(root, split)
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform, None)

    return train_dataset, test_dataset
