import os
import ants
import numpy as np
import SimpleITK as sitk
import shutil
from tqdm import tqdm
import argparse

def n4_bias_correct(path_in):
    img = sitk.ReadImage(path_in, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(img, mask)
    return corrected

def skull_strip_from_T1(t1_sitk):
    arr = sitk.GetArrayFromImage(t1_sitk)
    mask = (arr > arr.mean()).astype("uint8")
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(t1_sitk)
    return mask_img

def preprocess_patient(patient_id, input_root="./data", output_root="./data_traite", skip_steps=None):
    if skip_steps is None:
        skip_steps = []

    steps = [
        "correction_n4",
        "extraction_cerveau",
        "lecture_ants",
        "registration",
        "warp_masque",
        "application_masque",
        "sauvegarde"
    ]

    progress = tqdm(total=len(steps), desc=f"🧠 {patient_id}", unit="étape")

    input_dir = os.path.join(input_root, patient_id)
    output_dir = os.path.join(output_root, patient_id)
    os.makedirs(output_dir, exist_ok=True)

    flair_path = os.path.join(input_dir, "3DFLAIR.nii")
    t1_path = os.path.join(input_dir, "3DT1.nii")
    consensus_path = os.path.join(input_dir, "Consensus.nii")

    if not (os.path.exists(flair_path) and os.path.exists(t1_path) and os.path.exists(consensus_path)):
        progress.close()
        print(f"{patient_id}: ❌ Fichiers requis manquants")
        return

    # Step 1: Correction N4
    if "correction_n4" in skip_steps:
        t1_corrected = sitk.ReadImage(t1_path, sitk.sitkFloat32)
    else:
        t1_corrected = n4_bias_correct(t1_path)
    progress.update(1)

    # Step 2: Skull stripping
    if "extraction_cerveau" in skip_steps:
        brain_mask = sitk.Image(t1_corrected.GetSize(), sitk.sitkUInt8)
        brain_mask.CopyInformation(t1_corrected)
    else:
        brain_mask = skull_strip_from_T1(t1_corrected)
    progress.update(1)

    # Step 3: Lecture ANTs
    if "lecture_ants" not in skip_steps:
        arr = sitk.GetArrayFromImage(t1_corrected)
        spacing = list(t1_corrected.GetSpacing())
        origin = list(t1_corrected.GetOrigin())
        direction_flat = t1_corrected.GetDirection()
        dimension = t1_corrected.GetDimension()
        direction_matrix = np.array(direction_flat).reshape((dimension, dimension))

        t1_ants = ants.from_numpy(arr)
        t1_ants.set_spacing(spacing)
        t1_ants.set_origin(origin)
        t1_ants.set_direction(direction_matrix.tolist())
        flair_ants = ants.image_read(flair_path)
    progress.update(1)

    # Step 4: Enregistrement
    if "registration" not in skip_steps:
        reg = ants.registration(fixed=flair_ants, moving=t1_ants, type_of_transform="Affine")
    progress.update(1)

    # Step 5: Warp du masque
    if "warp_masque" not in skip_steps:
        mask_arr = sitk.GetArrayFromImage(brain_mask)
        mask_ants = ants.from_numpy(mask_arr)
        mask_ants.set_spacing(spacing)
        mask_ants.set_origin(origin)
        mask_ants.set_direction(direction_matrix.tolist())

        warped_mask = ants.apply_transforms(
            fixed=flair_ants,
            moving=mask_ants,
            transformlist=reg['fwdtransforms'],
            interpolator='nearestNeighbor'
        )

        warped_mask = warped_mask.threshold_image(0.5, 1.1, 1, 0)

        # ✅ Nettoyage morphologique
        warped_mask = warped_mask.iMath("GetLargestComponent")
        warped_mask = warped_mask.iMath("FillHoles")

        # ✅ Contrôle des valeurs du masque
        print(f"{patient_id}: masque → min={warped_mask.min()}, max={warped_mask.max()}")
    progress.update(1)

    # Step 6: Application du masque
    if "application_masque" not in skip_steps:
        flair_stripped = flair_ants * warped_mask
    else:
        flair_stripped = flair_ants
    progress.update(1)

    # Step 7: Sauvegarde
    if "sauvegarde" not in skip_steps:
        ants.image_write(flair_stripped, os.path.join(output_dir, "3DFLAIR_traite.nii"))
        shutil.copy(consensus_path, os.path.join(output_dir, "Consensus_traite.nii"))
    progress.update(1)

    progress.close()
    print(f"{patient_id}: ✅ Terminé")

def preprocess_all_patients(input_root="./data", output_root="./data_traite", skip_steps=None):
    os.makedirs(output_root, exist_ok=True)
    patients = [p for p in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, p))]

    for patient_id in patients:
        preprocess_patient(patient_id, input_root, output_root, skip_steps)

    print("\n🎉 Tous les patients ont été traités.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", type=str, help="Étapes à ignorer (séparées par des virgules)", default="")
    args = parser.parse_args()
    skip_steps = [s.strip() for s in args.skip.split(",") if s.strip()]

    print(f"Étapes ignorées : {skip_steps if skip_steps else 'Aucune'}")
    preprocess_all_patients(input_root="./data", output_root="./data_traite", skip_steps=skip_steps)
