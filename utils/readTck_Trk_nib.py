import nibabel as nib

def get_tck_trk_streamlines(pathname):
    return nib.streamlines.load(pathname).streamlines