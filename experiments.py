import os.path as op
import dipy.reconst.csdeconv as csd
from dipy.data.fetcher import fetch_hcp
from dipy.core.gradients import gradient_table
import nibabel  as nib
import time


def main():
    subject = 100307
    dataset_path = fetch_hcp(subject)[1]
    subject_dir = op.join(
        dataset_path,
        "derivatives",
        "hcp_pipeline",
        "sub-{subject}",
        "ses-01",
        )
    subject_files = [op.join(subject_dir, "dwi",
         "sub-{subject}_dwi.{ext}") for ext in ["nii.gz", "bval", "bvec"]]
    data_img = nib.load(subject_files[0])
    data = data_img.get_fdata()
    seg_img = nib.load(op.join(
        subject_dir, "anat", f'sub-{subject}_aparc+aseg_seg.nii.gz'))
    seg_data = seg_img.get_fdata()
    brain_mask = seg_data > 0
    gtab = gradient_table(subject_files[1], subject_files[2])

    csdm = csd.ConstrainedSphericalDeconvModel(gtab)
    for engine in ["serial", "dask", "joblib", "ray"]:
        print("Engine: ", engine)
        t1 = time.time()
        csdf = csdm.fit(data, mask=brain_mask, engine=engine)
        print("Duration:", time.time() - t1, "seconds")

if __name__ == "__main__":
    main()
