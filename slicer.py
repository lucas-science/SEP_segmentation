import os
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.transform import resize

def flexible_slicer(data_dir, output_dir, orientation, structure, target_size):
    os.makedirs(output_dir, exist_ok=True)
    slice_counter = 0

    for patient in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient)
        if not os.path.isdir(patient_path):
            continue

        flair_path = os.path.join(patient_path, '3DFlair.nii')
        mask_path = os.path.join(patient_path, 'Consensus.nii')

        if not os.path.exists(flair_path) or not os.path.exists(mask_path):
            print(f"[!] Fichiers manquants pour {patient}, ignoré.")
            continue

        flair_img = sitk.ReadImage(flair_path)
        mask_img = sitk.ReadImage(mask_path)

        flair_np = sitk.GetArrayFromImage(flair_img)
        mask_np = sitk.GetArrayFromImage(mask_img)

        if orientation == 'axial':
            flair_slices = flair_np
            mask_slices = mask_np
        elif orientation == 'sagittal':
            flair_slices = np.transpose(flair_np, (2, 1, 0))
            mask_slices = np.transpose(mask_np, (2, 1, 0))
        elif orientation == 'coronal':
            flair_slices = np.transpose(flair_np, (1, 2, 0))
            mask_slices = np.transpose(mask_np, (1, 2, 0))
        else:
            raise ValueError("Orientation invalide : axial, sagittal ou coronal")

        for i in range(flair_slices.shape[0]):
            flair_slice = flair_slices[i, :, :]
            mask_slice = mask_slices[i, :, :]

            flair_slice = (flair_slice - np.min(flair_slice)) / (np.ptp(flair_slice) + 1e-8)
            flair_slice = resize(flair_slice, target_size, preserve_range=True).astype(np.uint8)
            mask_slice = (mask_slice > 0).astype(np.uint8) * 255
            mask_slice = resize(mask_slice, target_size, preserve_range=True).astype(np.uint8)

            if structure == 'flat':
                image_out = os.path.join(output_dir, 'images')
                mask_out = os.path.join(output_dir, 'masks')
                os.makedirs(image_out, exist_ok=True)
                os.makedirs(mask_out, exist_ok=True)
                id = f"{slice_counter:04d}"
                plt.imsave(os.path.join(image_out, f"{id}.png"), flair_slice, cmap='gray')
                plt.imsave(os.path.join(mask_out, f"{id}.png"), mask_slice, cmap='gray')

            elif structure == 'per_patient':
                image_out = os.path.join(output_dir, patient, 'IRM')
                mask_out = os.path.join(output_dir, patient, 'masque')
                os.makedirs(image_out, exist_ok=True)
                os.makedirs(mask_out, exist_ok=True)
                id = f"{i:03d}"
                plt.imsave(os.path.join(image_out, f"slice_{id}.png"), flair_slice, cmap='gray')
                plt.imsave(os.path.join(mask_out, f"slice_{id}.png"), mask_slice, cmap='gray')

            elif structure == 'unext':
                image_out = os.path.join(output_dir, 'images')
                mask_out = os.path.join(output_dir, 'masks', '1')
                os.makedirs(image_out, exist_ok=True)
                os.makedirs(mask_out, exist_ok=True)
                id = f"{slice_counter:04d}"
                plt.imsave(os.path.join(image_out, f"{id}.png"), flair_slice, cmap='gray')
                plt.imsave(os.path.join(mask_out, f"{id}.png"), mask_slice, cmap='gray')

            slice_counter += 1

        print(f"[✓] {patient} traité ({flair_slices.shape[0]} slices).")

def parse_args():
    parser = argparse.ArgumentParser(description="Slice .nii into PNGs with flexible structure.")
    parser.add_argument('--data_dir', type=str, required=True, help='Répertoire contenant les dossiers patients.')
    parser.add_argument('--output_dir', type=str, required=True, help='Dossier de sortie.')
    parser.add_argument('--orientation', type=str, choices=['axial', 'sagittal', 'coronal'], default='axial', help='Orientation des slices.')
    parser.add_argument('--structure', type=str, choices=['flat', 'per_patient', 'unext'], default='flat', help='Structure de sortie.')
    parser.add_argument('--size', type=int, nargs=2, default=[512, 512], help='Taille des images (largeur hauteur).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    flexible_slicer(args.data_dir, args.output_dir, args.orientation, args.structure, tuple(args.size))
