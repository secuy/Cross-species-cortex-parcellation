import numpy as np
import os

from dipy.io.streamline import load_trk, load_tck
from dipy.data.fetcher import fetch_file_formats, get_file_formats
import nibabel as nib


def get_tck_trk_streamlines(pathname):
    fetch_file_formats()
    bundles_filename, ref_anat_filename = get_file_formats()
    references = nib.load(ref_anat_filename)
    file_name, file_extension = os.path.splitext(pathname)
    streamlines = []
    if file_extension == '.tck':
        streamlines = load_tck(pathname, reference=references, bbox_valid_check=False)
        space = streamlines.space
    elif file_extension == '.trk':
        streamlines = load_trk(pathname, reference="same", bbox_valid_check=False)
        space = streamlines.space
    return streamlines.streamlines, references, space
