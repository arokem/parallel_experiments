import os.path as op
import dipy.reconst.csdeconv as csd
from dipy.data.fetcher import fetch_hcp
from dipy.core.gradients import gradient_table
import nibabel  as nib
import time
from dipy.reconst.csdeconv import (auto_response_ssst,
                                   mask_for_response_ssst,
                                   response_from_mask_ssst)
from dipy.align import resample



def main():
    subject = 100307
    dataset_path = fetch_hcp(subject)[1]
    subject_dir = op.join(
        dataset_path,
        "derivatives",
        "hcp_pipeline",
        f"sub-{subject}",
        "ses-01",
        )
    subject_files = [op.join(subject_dir, "dwi",
         f"sub-{subject}_dwi.{ext}") for ext in ["nii.gz", "bval", "bvec"]]
    dwi_img = nib.load(subject_files[0])
    data = dwi_img.get_fdata()
    seg_img = nib.load(op.join(
        subject_dir, "anat", f'sub-{subject}_aparc+aseg_seg.nii.gz'))
    seg_data = seg_img.get_fdata()
    brain_mask = seg_data > 0
    dwi_volume = nib.Nifti1Image(data[..., 0], dwi_img.affine)
    brain_mask_xform = resample(brain_mask, dwi_volume,
                                moving_affine=seg_img.affine)
    brain_mask_data = brain_mask_xform.get_fdata()
    gtab = gradient_table(subject_files[1], subject_files[2])
    response_mask = mask_for_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    response, _ = response_from_mask_ssst(gtab, data, response_mask)
    csdm = csd.ConstrainedSphericalDeconvModel(gtab, response=response)
    for engine in ["serial", "dask", "joblib", "ray"]:
        print("Engine: ", engine)
        t1 = time.time()
        csdf = csdm.fit(data, mask=brain_mask_data, engine=engine)
        print("Duration:", time.time() - t1, "seconds")

if __name__ == "__main__":
    main()
