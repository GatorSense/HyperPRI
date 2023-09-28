# -*- coding: utf-8 -*-
"""
Takes after previous code by jpeeples67 in https://github.com/GatorSense/Histological_Segmentation

@author: changspencer
"""
import os
import json
import errno
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging

# HSI Data package
import spectral.io.envi as envi

# PyTorch Packages
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class HyperpriDataset(Dataset):
    """
    Created on Tue Nov 8 09:12:40 2022

    @author: chang.spencer

    Adapted from a previous RootDataset class.
    """
    def __init__(self, root, mode='RGB', img_transform=None, label_transform=None,
                 subset:list=None, label_subset:list=[3], unsqueeze_img=False,
                 hsi_lo=0, hsi_hi=0, json_file:str=None, json_verb=False):

        self.class_list = [
            'Peanut',
            'SweetCorn',
        ]
        if subset is not None:
            self.class_list = subset

        if not label_subset:
            self.label_subset = [0, 3]
        else:
            self.label_subset = list(set(label_subset))   # Actual labels to consider in segmentation
            self.label_subset.sort()
        if 0 not in self.label_subset:  # always have the background be class 0
            self.label_subset.insert(0, 0)
        if self.label_subset[0] != 0:
            self.label_subset = [ele - min(self.label_subset) for ele in self.label_subset]

        assert hsi_lo >= 0  # Input validation
        if hsi_hi <= 0:
            hsi_hi = 299 + hsi_hi
        assert hsi_lo < hsi_hi

        self.root = root
        self.mode = mode
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.unsqueeze_hsi = unsqueeze_img
        self.hsi_lo = hsi_lo  # First Band to pull from HSI data
        self.hsi_hi = hsi_hi  # Last Band to pull from HSI data
        self.files = []

        if not json_file:
            self._parse_train_dir()
        else:
            # JSON file must be the correct path to find proper splits
            self.json_file = json_file
            self._parse_json_file(json_file, verbose=json_verb)

        # Increase the number of times an underrepresented class is sampled
        file_idx = 0
        self.sample_weights = np.zeros(self.class_count.sum())
        for count in self.class_count:
            class_weight = 0 if count == 0 else self.class_count.max() / count

            self.sample_weights[file_idx:file_idx + count] = class_weight
            file_idx += count

    def _parse_train_dir(self):
        '''
        Usual way of going through file directories and pulling out the
            data in a particular folder

        # TODO - Update for HyperPRI release
            - HyperPRI --> rgb_files/hsi_files/mask_files instead of pulling based on class
            - Pull all filenames within a given directory
            - Remove Class counter
        '''
        self.class_count = np.zeros(len(self.class_list), dtype=np.int)
        imgdir = os.path.join(self.root, 'images')
    
        for os_root, dirs, files in os.walk(imgdir):
            curr_class_idx = None
            for class_idx in range(len(self.class_list)):
                if self.class_list[class_idx] in os_root:
                    curr_class_idx = class_idx
                    break

            # The subset of classes does not include 'os_root'
            if curr_class_idx is None:
                continue
            if self.mode.lower() == 'hsi':
                if len(files) > 0:
                    basename = files[0].rsplit('.', 1)[0]
                    namepath = os.path.join(os_root, basename)
                    hdr_path = os.path.join(os_root, "hinalea_hsi.hdr")

                    rootp = pathlib.Path(os_root)
                    index = rootp.parts.index('images')
                    basename_idx = rootp.parts.index(basename)
                    # Only take the next folder after masks_pixel_gt for GT masks
                    labelpath = os.path.join(self.root, 'mask_files', *rootp.parts[index+1:basename_idx])
                    labelpath = os.path.join(labelpath, f"{basename}_mask.png")

                    # Data existence validation
                    if not os.path.exists(f"{hdr_path}"):
                        FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f"{hdr_path}")
                    if not os.path.exists(f"{namepath}.dat"):
                        FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f"{namepath}.dat")
                    if not os.path.exists(labelpath):
                        FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), labelpath)

                    # Label name validation
                    self.files.append({
                        "img": f"{namepath}.png",
                        "hdr": f"{hdr_path}",
                        "dat": f"{namepath}.dat",
                        "label": labelpath
                    })

                    self.class_count[curr_class_idx] += 1
            else:
                for name in files:
                    imgpath = os.path.join(os_root, name)

                    rootp = pathlib.Path(os_root)
                    index = rootp.parts.index('images')
                    labelpath = os.path.join(self.root, 'mask_files', *rootp.parts[index+1:])
                    labelpath = os.path.join(labelpath, f"{name.rsplit('.', 1)[0]}_mask.png")

                    # Label name validation
                    if labelpath.endswith('.png'):
                        self.files.append({
                            "img": imgpath,
                            "label": labelpath
                        })
                    else:
                        ending = '.' + labelpath.split('.')[-1]
                        self.files.append({
                            "img": imgpath,
                            "label": labelpath.replace(ending, '.png')
                        })
                    self.class_count[curr_class_idx] += 1

    def _parse_json_file(self, json_path, verbose=False):
        '''
            This function parses a given JSON path to determine where
                certain rhizoboxes may be stored on your personal computer. Any
                files found in the JSON file and in the directories noted below
                will be added to this dataset object's list of loadable files.

            See the provided JSON files for the proper format per
                plant rhizobox.
        Args:
            json_path: str full path to a desired JSON file.
            verbose: bool flag that indicates when certain files were not found
                though listed in provided JSON file detailed by `json_path`.
        Returns:
            None
        '''
        with open(json_path, 'r') as f:
            data_dict = json.load(f)

        box_names = data_dict.keys()  # Data keys
        self.class_count = np.zeros(len(self.class_list), dtype=int)
        for idx, box in enumerate(box_names):
            if not box.startswith("box") or not data_dict[box]['dates']:
                continue   # Skip extraneous info

            box_class = data_dict[box]['plant_folder']
            box_sz = data_dict[box]['resolution']
            img_dir = f"{self.root}/{box_class}_{box_sz}/{data_dict['img_dir']}/"
            label_dir = f"{self.root}/{box_class}_{box_sz}/{data_dict['mask_dir']}/"

            for date_idx, date in enumerate(data_dict[box]['dates']):

                curr_class_idx = self.class_list.index(box_class)

                basename = f"{date}_{box}_ref"
                img_name = f"{basename}.png"
                mask_name = f"{basename}_mask.png"

                if self.mode.lower() == 'hsi':
                    hsi_dir = f"{self.root}/{box_class}_{box_sz}/{data_dict['hsi_dir']}/"

                    hdr_name = f"{basename}.hdr"
                    dat_name = f"{basename}.dat"
                    imgpath = os.path.join(f"{img_dir}", img_name)
                    hdrpath = os.path.join(f"{hsi_dir}/{basename}", hdr_name)
                    datpath = os.path.join(f"{hsi_dir}/{basename}", dat_name)
                    labelpath = os.path.join(f"{label_dir}", mask_name)
                    logging.info(hdrpath)

                    files_nonexist = os.path.exists(labelpath) and os.path.exists(hdrpath) and os.path.exists(datpath)
                    if not files_nonexist:
                        if verbose:
                            logging.info(f"{basename}: One of the necessary HSI or mask files does not exist. Skipping...")
                        continue

                    # Label name validation
                    self.files.append({
                        "img": imgpath,
                        "hdr": hdrpath,
                        "dat": datpath,
                        "label": labelpath
                    })
                    self.class_count[curr_class_idx] += 1

                else:
                    imgpath = os.path.join(img_dir, img_name)
                    labelpath = os.path.join(label_dir, mask_name)

                    if not os.path.exists(imgpath) or not os.path.exists(labelpath):
                        if verbose:
                            print(f"Either {imgpath} or {labelpath} does not exist. Skipping...")
                        continue

                    # Label name validation
                    if labelpath.endswith('.png'):
                        self.files.append({
                            "img": imgpath,
                            "label": labelpath
                        })
                    else:
                        ending = '.' + labelpath.split('.')[-1]
                        self.files.append({
                            "img": imgpath,
                            "label": labelpath.replace(ending, '.png')
                        })
                    self.class_count[curr_class_idx] += 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        # Get specific filename (not filepath)
        img_file = datafiles["img"]
        img_name = pathlib.PurePath(img_file).name.rsplit('.', 1)[0]

        if self.mode.lower() == 'rgb':
            img = Image.open(img_file).convert('RGB')
        if self.mode.lower() == 'gray':
            img = Image.open(img_file).convert('L')
            img = img.convert('RGB')
        if self.mode.lower() == 'hsi':
            hdr_file = datafiles['hdr']
            dat_file = datafiles['dat']
            logging.info(hdr_file)
            data_ref = envi.open(hdr_file, dat_file)
            # ENVI opens with channels at the last dimension
            img = np.moveaxis(np.array(data_ref.load()), -1, 0)
            img = img[self.hsi_lo:self.hsi_hi, :, :]
            if self.unsqueeze_hsi:
                img = np.expand_dims(img, 0)  #? 3D_UNET: num_channels = 1
            img = torch.tensor(img)

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("L")

        # PIL Images and Tensors are transposes of each other, apparently
        if self.mode.lower() != 'hsi' and img.size[0] < img.size[1]:
            img = img.transpose(method=Image.ROTATE_90)
            label = label.transpose(method=Image.ROTATE_90)

        #! Remember that the HSI data will come out as Tensors and cannot use PIL transforms
        #!    ie. no "transforms.ToTensor()" allowed (This may be deprecated advice)
        state = torch.get_rng_state()
        if self.img_transform is not None:
            img = self.img_transform(img)

        torch.set_rng_state(state)
        if self.label_transform is not None:
            label = self.label_transform(label)
        label = np.array(label) * 255  # Transforms do not preserve Uint label
        label = np.where(label > 0, np.ones_like(label), np.zeros_like(label))   # Make nodules/pegs into roots

        # print(img.shape, label.shape)
        return {'image':img, 'mask': label, 'index': img_name, 'label': label_file}
