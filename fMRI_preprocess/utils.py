'''
!pip install -Uqq ipdp
!pip install wandb
!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git
'''



import os
#import ipdb
import h5py
import yaml
import json
import pickle
import numpy as np
from tqdm import tqdm
import nibabel as nib
import scipy.io as spio
import matplotlib.pyplot as plt
import sys

from scipy.stats import zscore

import torch
from torch import nn, optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

import clip


def get_img_trans(extra_aug=0.9, toPIL=True, img_size=256, color_jitter_p=0.4,
                  gray_scale_p=0.2, gaussian_blur_p=0.5, masking_p=1.0,
                  masking_ratio=0.3):
    '''
    - extra_aug: a value between 0-1. If 0, only apply resizing and to tensor.
                 If > 0, this p controls the probability that an augmentation
                 is actually implemented.
    - toPIL: bool. CLIP need PIL to process, if using other models, set it to F.
    - color_jitter_p: ADA 0.4, VICReg 0.8
    - gray_scale_p: VICReg 0.2
    - gaussian_blur_p: VICReg0.5, similar to ADA's filter.
                       (ADA has multiple, and has p = 1.0)
    - masking_ratio: ADA 0.5
    '''

    img_trans = []
    img_trans.append(transforms.ToTensor())
    img_trans.append(transforms.Resize((img_size, img_size)))

    run_extra = np.random.rand()
    if bool(extra_aug) and (run_extra < extra_aug):
        img_trans.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))
        img_trans.append(transforms.RandomHorizontalFlip(p=0.5))

        cj = np.random.rand()
        # print(f'color jitter {cj}, {cj < color_jitter_p}')
        if cj < color_jitter_p:
            img_trans.append(transforms.ColorJitter(0.4, 0.4, 0.2, 0.1))
        gs = np.random.rand()
        # print(f'grayscale {gs}, {gs < gray_scale_p}')
        if gs < gray_scale_p:
            img_trans.append(transforms.Grayscale(num_output_channels=3))
        gb = np.random.rand()
        # print(f'gaussian blur {gb}, {gb < gaussian_blur_p}')
        if gb < gaussian_blur_p:
            img_trans.append(transforms.GaussianBlur(kernel_size=23))
        # img_trans.append(transforms.RandomSolarize(128, p=0.1))

        img_trans.append(RandomMask(masking_ratio))

    if toPIL:
        img_trans.append(transforms.ToPILImage())
    img_trans = transforms.Compose(img_trans)
    return img_trans

class RandomMask(object):
    ''' Apply random cutouts (masking) on an image'''
    def __init__(self, mask_ratio=0.5):
        ''' - mask_ratio: the ratio of (cutout area width or height) / (image width or height)'''
        self.cx = np.random.rand()
        self.cy = np.random.rand()
        self.m = mask_ratio / 2.0

    def __call__(self, sample):
        cx, cy, m = self.cx, self.cy, self.m
        _, x, y = sample.shape

        start_x = round((cx - m) * x)
        start_y = round((cy - m) * y)
        end_x = round((cx + m) * x)
        end_y = round((cy + m) * y)

        mask = torch.ones_like(sample)
        mask[:, max(0, start_x): min(x-1, end_x), max(0, start_y): min(y-1, end_y)] = 0

        return sample * mask

def thresholding(vec, thr):
    ''' thresholding vec values to [-thr, thr]. '''
    if thr > 0:
        _thr = torch.ones_like(vec) * thr
        vec = torch.minimum(vec, _thr)
        vec = torch.maximum(vec, -_thr)
    return vec


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)



def _load_fmri_init(roi=None, fmri_pad=None, fmri_model=None):
    fmri_dir = roi if roi else FMRI_DIR
    fmri_files = [f for f in os.listdir(fmri_dir) if
                  os.path.isfile(os.path.join(fmri_dir, f)) and
                  f[-5:] == '.hdf5']

    if fmri_pad:
        with h5py.File(os.path.join(fmri_dir, fmri_files[0]),
                       'r') as f:
            x = f['betas'][0]
            num_voxel = x.shape[-1]
        left_pad = (fmri_pad - num_voxel) // 2
        right_pad = fmri_pad - num_voxel - left_pad
    else:
        left_pad = right_pad = 0

    if fmri_model is not None:
        fmri_model.to(device)
        fmri_model.eval()

    return fmri_dir, fmri_files, left_pad, right_pad, fmri_model




def _load_fmri_forward(idx, fmri_dir, fmri_files, fmri_pad, left_pad, right_pad,
                       fmri_model=None, extra_fmri_fn=None, fmri_model_args={},
                       verbose=False,subj=None,ro=None):
    ''' helper func for loading fMRI (used in dataset forward) '''
    sess = idx // TRIAL_PER_SESS + 1
    fmri_file = os.path.join(fmri_dir, f'{fmri_files[0][:13]}{sess:02}_s{subj:02}_{ro}_zscored.hdf5')
    #fmri_file = os.path.join(fmri_dir,f'{fmri_files[0][:-7]}{sess:02}.hdf5')
    #print(fmri_file)
    with h5py.File(fmri_file, 'r') as f:
        fmri_sample = f['betas'][idx % TRIAL_PER_SESS]
    if verbose:
        print('fmri loaded from', fmri_file)
        print('fmri shape:', fmri_sample.shape)
        print('beta min, mean, max:', fmri_sample.min(),
            fmri_sample.mean(), fmri_sample.max())

    fmri_sample = torch.FloatTensor(fmri_sample).to(device)
    if fmri_pad:
        fmri_sample = F.pad(fmri_sample, (left_pad, right_pad),
                            'constant', 0)
    if fmri_model is not None:
        with torch.no_grad():
            fmri_sample = fmri_model(fmri_sample.unsqueeze(0), **fmri_model_args)
    if extra_fmri_fn is not None:
        fmri_sample = extra_fmri_fn(fmri_sample)
    return fmri_sample




class NSDwithCLIP(Dataset):
    def __init__(self, load_fmri=True, fmri_pad=None, roi=None,
                 fmri_model=None, fmri_model_args={}, extra_fmri_fn=None,
                 load_img=True, img_trans=0,
                 load_clip=False, CLIP=None, threshold=0.0, clip_norm=False, clip_std=False, clip_01=False,
                 load_caption=False, caption_selection='avg',
                 load_cat=False, cat_type='things_stuff',
                 load_clip_mapped=False, mapper=None,
                 ):

        """ MAIN DATASET FUNCTION USED """
        '''
        - load_fmri, load_img, load_clip, load_caption, load_cat, load_clip_mapped:
          all bool, choose the modalities you need. By default load fMRI and CLIP.
          (load_clip means whether loading CLIP image vectors,
           load_caption means whether loading CLIP caption vectors,
           load_clip_mapped means whether loading fMRI-mapped CLIP vectors.)
        - fmri_pad: int, pad fMRI vector to this length. Useful when model
                    requires inputs to have a certain shape.
        - roi: string, the directory contains extracted ROI voxels.
               if None, return the 3d fmri activity: (83, 104, 81) for 1.8mm.
        - fmri_model: a PyTorch model that takes in fMRI signal as input. fMRI
                      will be passed through this model as the dataset point.
        - fmri_model_args: args for forward loop of fmri_model.
        - extra_fmri_fn: extra processing steps (after passing through fmri_model).
        - img_trans: a value between 0-1. If 0, only apply resizing and to tensor.
                     If > 0, this p controls the probability that an augmentation
                     images will be augmented with probability p before passed to CLIP.
        - CLIP: CLIP model, if None, load the pre computed vectors.
        - threshold: to thresholding CLIP vector values, set to <= 0 if no
                     thresholding is wanted (set to 1.5 to remove spikes)
        - clip_norm: bool. If True, the CLIP vector will be normalized to a
                     unit sphere.
        - clip_std: bool. If True, the CLIP vector will be standardized.
        - clip_01: bool. If True, the CLIP vector will be normalized to 0-1.
        - caption_selection: either 'first' or 'avg' or 'random'. Each image has
                            multiple captions. 'first' will only keep the first,
                            and 'avg' will average all captions, 'random' will
                            randomly choose one.
        - cat_type: the type of COCO categories, select from "things", "stuff",
                    "things_stuff" (default).
        - mapper: the mapper model to map fMRI to CLIP vectors.
        '''
        assert load_fmri or load_img or load_clip or load_caption or load_cat, (
            'You must choose to load at least one modeality!'
            )

        self.load_fmri = load_fmri
        self.load_img = load_img
        self.load_clip = load_clip
        self.load_caption = load_caption
        self.load_cat = load_cat
        self.load_clip_mapped = load_clip_mapped

        stim_order = loadmat(STIM_ORDER_FILE)
        self.subjectim = stim_order['subjectim']
        self.masterordering = stim_order['masterordering']
        self.stim_file = STIM_FILE
        self.img_trans = img_trans
        self.CLIP = CLIP

        if load_fmri or load_clip_mapped:
            self.fmri_dir, self.fmri_files, self.left_pad, self.right_pad, self.fmri_model = (
                _load_fmri_init(roi=roi, fmri_pad=fmri_pad, fmri_model=fmri_model))
            self.fmri_model_args = fmri_model_args
            self.fmri_pad = fmri_pad
            self.extra_fmri_fn = extra_fmri_fn
            if load_clip_mapped:
                assert mapper is not None
                self.mapper = mapper.to(device)
                self.mapper.eval()

        if load_clip:
            if CLIP: # calculate clip vectors on the go
                if load_caption:
                    vetted_cap_path = os.path.join(DATA_DIR, 'nsddata_stimuli/stimuli/nsd/annotations/', f'nsd_captions_vetted_{SUBJ}.pkl')
                    with open(vetted_cap_path, 'rb') as f:
                        self.cap_vetted = pickle.load(f)
            else: # load pre-computed clip vectors
                clip_image_f_path = os.path.join(DATA_DIR, 'clip_features', 'nsd_images.pkl')
                with open(clip_image_f_path, 'rb') as f:
                    self.clip_image_f = pickle.load(f)
                if load_caption:
                    clip_cap_f_path = os.path.join(DATA_DIR, 'clip_features', 'nsd_captions.pkl')
                    with open(clip_cap_f_path, 'rb') as f:
                        self.clip_cap_f = pickle.load(f)

            self.threshold = threshold
            assert (clip_norm and clip_std and clip_01) is False, (
                'normalization or standardization cannot be applied together.')
            self.clip_norm = clip_norm
            self.clip_std = clip_std
            self.clip_01 = clip_01

        if load_caption:
            assert caption_selection in ['first', 'random', 'avg'], (
                "you must choose from 'first', 'random', 'avg' as your caption selection method")
            self.caption_selection = caption_selection

        if load_cat:
            self.cat_list, self.stim_info, self.nsd_cat, self.num_class = _load_cat_init(cat_type)

    def __len__(self):
        try:
            return TRIAL_PER_SESS * len(self.fmri_files)
        except:
            return TRIAL_PER_SESS * 37 # TODO: hard-coded for now, using # sessions for one suject

    def _get_caption(self, cap, method):
        if cap.dim() > 1:
            if method == 'first':
                cap = cap[0]
            elif method == 'random':
                rand_id = np.random.choice(len(cap))
                cap = cap[rand_id]
            else:
                cap = cap.mean(0)
        return cap

    def _scale_(self, vec):
        '''
        Without thresholding:
        min, max of CLIP vecs are: -9.9688; 5.1289
        mean, std of CLIP vecs are: -0.0045; 0.4602

        With +- 1.5 thresholding:
        mean, std of CLIP vecs are:
        '''
        if self.clip_norm:
            vec = vec / torch.norm(vec, p=2, dim=-1)
        if self.clip_std:
            if self.threshold:
                if self.threshold == 1.5:
                    clip_mean = 0.0047
                    clip_std = 0.3623
                else: # TODO: not the correct way to do, but...
                    clip_mean = vec.mean(-1)
                    clip_std = vec.std(-1)
            else:
                clip_mean = -0.0045
                clip_std = 0.4602
            vec = (vec - clip_mean) / clip_std
        if self.clip_01:
            clip_min = -self.threshold if self.threshold else -9.9688
            clip_max = self.threshold if self.threshold else 5.1289
            vec -= clip_min
            vec /= (clip_max - clip_min)
        return vec

    def get_fmri_shape(self):
        with h5py.File(os.path.join(self.fmri_dir,
                                    self.fmri_files[0]), 'r') as f:
            x = f['betas'][0]
            s = x.shape
            print(f'fmri signal shape: {s}')
            x = torch.FloatTensor(x).to(device)
            if self.fmri_pad:
                x = F.pad(x, (self.left_pad, self.right_pad), 'constant', 0)
                print(f'padded to: {self.fmri_pad}')
            if self.fmri_model is not None:
                with torch.no_grad():
                    x = self.fmri_model(x.unsqueeze(0), **self.fmri_model_args)
            if self.extra_fmri_fn is not None:
                x = self.extra_fmri_fn(x)
            return x.flatten().shape[0]

    def __getitem__(self, idx, verbose=False, load_ori_img=False):
        sample = {}
        nsdId = self.subjectim[int(SUBJ)-1, self.masterordering[idx] - 1] - 1
        sample['nsdId'] = nsdId

        if verbose:
            print(f'stim nsd id: {nsdId}, aka #{nsdId+1} in the tsv files.')

        ##### fMRI #####
        if self.load_fmri or self.load_clip_mapped:
            sample['fmri'] = _load_fmri_forward(idx, self.fmri_dir, self.fmri_files,
                self.fmri_pad, self.left_pad, self.right_pad, fmri_model=self.fmri_model,
                fmri_model_args=self.fmri_model_args, extra_fmri_fn=self.extra_fmri_fn,
                verbose=verbose, subj=SUBJ,ro=RO)

            if self.load_clip_mapped:
                with torch.no_grad():
                    sample['clip_mapped'] = self.mapper(sample['fmri']).squeeze()

        ##### image #####
        if self.load_img:
            with h5py.File(self.stim_file, 'r') as f:
                _image = f['imgBrick'][nsdId]
                sample['img'] = get_img_trans(extra_aug=self.img_trans,
                                              toPIL=False)(_image).to(device)

        ##### clip_image #####
        if self.load_clip:
            if self.CLIP:
                with h5py.File(self.stim_file, 'r') as f:
                    _image = f['imgBrick'][nsdId]
                    # print(f'_image range {_image.min()}, {_image.max()}')
                _image = get_img_trans(extra_aug=self.img_trans)(_image)
                if load_ori_img: sample['_img'] = _image
                _image = self.CLIP[1](_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    clip_image = self.CLIP[0].encode_image(_image).squeeze()
            else:
                clip_image = self.clip_image_f[nsdId]
            clip_image = thresholding(clip_image, self.threshold)
            clip_image = clip_image.float().to(device)

            clip_image = self._scale_(clip_image)
            sample['clip_img'] = clip_image

        ##### clip_caption #####
        if self.load_caption:
            if self.CLIP:
                _cap = clip.tokenize(self.cap_vetted[nsdId]).to(device)
                with torch.no_grad():
                    clip_cap = self._get_caption(self.CLIP[0].encode_text(_cap),
                                                 self.caption_selection
                                                 ).float().to(device)
            else:
                clip_cap = self._get_caption(self.clip_cap_f[nsdId],
                                             self.caption_selection
                                             ).float().to(device)
            clip_cap = thresholding(clip_cap, self.threshold)

            clip_cap = self._scale_(clip_cap)
            sample['clip_cap'] = clip_cap

        ##### categories #####
        if self.load_cat:
            cur_cat = self.nsd_cat[str(self.stim_info['cocoId'][nsdId])]
            cat = torch.zeros(self.num_class)
            for c in cur_cat:
                cat[self.cat_list.index(c)] = 1
            if verbose:
                print('Categories:', cat)
            sample['cat'] = cat.to(device)

        return sample
    
    
def get_dataloader(dataset, batch_size=32):
    ''' Give a whole dataset, seperate train set and val set so that they have different images'''
    stim_order = loadmat(STIM_ORDER_FILE)
    ''' Go through all samples to build a dict with keys being their stimulus (image) IDs. '''
    sig = {}
    # for idx in range(MAX_IDX):
    for idx in range(len(dataset)):
        ''' nsdId as in design csv files'''
        nsdId = stim_order['subjectim'][int(SUBJ)-1, stim_order['masterordering'][idx] - 1] # - 1
        if nsdId not in sig:
            sig[nsdId] = []
        sig[nsdId].append(idx)
    print(len(sig.keys()))

    ''' prepare dataloader '''
    train_idx_len = int(len(sig.keys()) * 0.85)
    train_idx = list(sig.keys())[: train_idx_len]
    val_idx = list(sig.keys())[train_idx_len:]

    train_idx = sorted(np.concatenate([sig[idx] for idx in train_idx]))
    val_idx = sorted(np.concatenate([sig[idx] for idx in val_idx]))
    print(f'num training samples {len(train_idx)}, val samples {len(val_idx)}')

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size)

    # Sanity check dataloader
    try:
        next(iter(train_loader))
    except:
        print('Cant load, double check if lengths for different samples are same, None type, etc.')
    print(f'train loader iter: {len(train_loader)}, val loader iter: {len(val_loader)}')
    return train_set, val_set, train_loader, val_loader





def merge_arrays(arr1, arr2):
    # Consider elements that are not equal to zero in either array
    merged_array = np.where((arr1 != 0) | (arr2 != 0), 1, 0)
    return merged_array.astype(float)




def extract_voxels(roi_dir, fmri_dir,out_dir, subj=None, ro=None):
    """
    

    Parameters
    ----------
    roi_dir : path
        Path of directory where ROI masks are stored.
    fmri_dir : path
        Path of directory where raw fMRI signals from specific subject are stored.
    out_dir : path
        Path of directory where masked fMRI signals will be stored.
    subj : str
        Number of subject i.e.'01' or '02' etc.
    ro : str
        ROI i.e. 'V1', 'V2', 'nsdgeneral'

    Returns
    -------
    Stores masked fMRI signals in out_dir

    """
  rois=os.listdir(roi_dir)
  lh_rois,rh_rois=[],[]

  for roi_file in rois:
      if not roi_file.startswith('.') and ro in roi_file:
          roi_name=os.path.basename(roi_file)[:-4] # Extract the ROI name
          hem=roi_name[:2] # Identify the hemisphere (lh or rh)
          roi_file=os.path.join(roi_dir,roi_file) # Path to ROI file

          # Load and process ROI mask based on hemisphere
          if hem=='lh':
              mask_lh=nib.load(roi_file).get_fdata() # Load left hemisphere ROI

              lh_rois.append(mask_lh)
              available_region = [int(r) for r in set(mask_lh.flatten())] # Identify available regions
              print(f'Extracting ROI based on {roi_name},',
                    f'available_regions: {available_region}')

          else:
              mask_rh=nib.load(roi_file).get_fdata() # Load right hemisphere ROI

              rh_rois.append(mask_rh)
              available_region = [int(r) for r in set(mask_rh.flatten())] # Identify available regions
              print(f'Extracting ROI based on {roi_name},',
                    f'available_regions: {available_region}')




  # Iterate through the rest of the arrays and merge with the accumulated result
  if len(lh_rois)>1:
      mask_lh=merge_arrays(lh_rois[0],lh_rois[1])
      for arr in lh_rois[2:]:

          mask_lh = merge_arrays(mask_lh, arr)

  if len(rh_rois)>1:
      mask_rh=merge_arrays(rh_rois[0],rh_rois[1])
      for arr in rh_rois[2:]:
          mask_rh = merge_arrays(mask_rh, arr)






  # Process fMRI data
  mask_rh_=(mask_rh != 0) # Generate mask for right hemisphere
  mask_lh_=(mask_lh != 0) # Generate mask for left hemisphere
  print(f'\nTotal ROI voxel count for lh: {np.count_nonzero(mask_lh)}\n', flush=True)
  print(f'\nTotal ROI voxel count for rh: {np.count_nonzero(mask_rh)}\n', flush=True)




  fmri_list=os.listdir(fmri_dir)


  sess_list=[]

  # Prepare sessions based on file naming convention
  for sess in range(1,int(len(fmri_list)/2)+1):

      sess_list.append([file for file in fmri_list if os.path.basename(file)[-6:-4]==f'{sess:02d}'])


  # Process fMRI data based on sessions
  for i in tqdm(sess_list):
      if len(i)!=2:
          print(f'Run number {sess_list.index(i)+1} does not include both hemispheres!')
          continue
      for fmri_file in i:


          hem=os.path.basename(fmri_file)[:2]
          fmri_file=os.path.join(fmri_dir,fmri_file)
          fmri = nib.load(fmri_file).get_fdata()
          fmri=np.transpose(fmri, (3, 0, 1, 2))


          # Extract voxel data for each hemisphere

          if hem=="lh":
              fmri = [fmri[trial][mask_lh_] for trial in range(len(fmri))]
              fmri_lh = np.stack(fmri)
              type(fmri_lh)

          else:

              fmri = [fmri[trial][mask_rh_] for trial in range(len(fmri))]
              fmri_rh = np.stack(fmri)
              type(fmri_rh)

      # Combine left and right hemisphere data
      fmri=np.concatenate((fmri_lh, fmri_rh), axis=1)
      fmri_rh=0
      fmri_lh=0

      out_f = os.path.join(out_dir, os.path.basename(fmri_file)[3:-4]+f's{subj:02}{ro}_fsaverage.hdf5')

      # Save processed data in HDF5 format
      with h5py.File(out_f, 'w') as f:
          dset = f.create_dataset('betas', data=fmri)


