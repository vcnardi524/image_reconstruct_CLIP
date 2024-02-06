

RO='nsdgeneral' # Choose ROI from file between V1, V2, V3, V4 or nsdgeneral
SUBJ = '01'



FMRI_DIR='/content/drive/MyDrive/fMRI-reconstruction-NSD/raw_fsaverage_Subj01'
ROI_DIR='/content/drive/MyDrive/fMRI-reconstruction-NSD/ROI'
OUT_DIR='/content/drive/MyDrive/fMRI-reconstruction-NSD/subj01_nsdgeneral_before_zscore' 
zscored_dir='/content/drive/MyDrive/fMRI-reconstruction-NSD/subj01_nsdgeneral_zscore'
ZSCORED_DIR = os.path.join(zscored_dir, f'{RO}') # Directory containing the processed fMRIS (after z-score) will be stored
SESS_NUM=40
STIM_FILE = "/content/drive/MyDrive/ICLR/nsd_stimuli.hdf5"
STIM_ORDER_FILE = "/content/drive/MyDrive/ICLR/nsd_expdesign.mat"
STIM_INFO = "/content/drive/MyDrive/ICLR/nsd_stim_info_merged.pkl"
TRIAL_PER_SESS = 750

MAX_IDX = TRIAL_PER_SESS * SESS_NUM
device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CREATE_DATASET=False
CREATE_LOADERS=False


if __name__=='__main__':

    extract_voxels(ROI_DIR,FMRI_DIR, OUT_DIR, subj=SUBJ,ro=RO)

    

    n = 3 # repetition

    #Z-SCORE PROCESS
    for sess in tqdm(range(SESS_NUM)):
        in_file = os.path.join(OUT_DIR, f'betas_session{sess+1:02}s{SUBJ:02}{RO}_fsaverage.hdf5')
        out_file = os.path.join(ZSCORED_DIR, f'betas_session{sess+1:02}_s{SUBJ:02}_{RO}_zscored.hdf5')
        with h5py.File(in_file, 'r') as f:
            fmri = f['betas'][()]
        fmri = zscore(fmri, 0)
        with h5py.File(out_file, 'w') as f:
            dset = f.create_dataset('betas', data=fmri)


    if CREATE_DATASET:
      clip_model = clip.load("ViT-B/32", device=device)
      dataset = NSDwithCLIP(load_fmri=True, roi=ZSCORED_DIR,
                       CLIP=clip_model, img_trans=0.0,
                       clip_norm=True, clip_std=False, clip_01=False, threshold=1.5,#)
                       caption_selection='random',)



    if CREATE_DATASET and CREATE_LOADERS:
      train_set, val_set, train_loader, val_loader = get_dataloader(dataset, batch_size=32)

    