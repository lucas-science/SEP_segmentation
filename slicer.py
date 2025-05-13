import os
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.transform import resize

def flexible_slicer(data_dir, output_dir, orientation, structure, target_size, max_empty_ratio=None, slice_range=None):
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

        flair_np = sitk.GetArrayFromImage(flair_img).astype(np.float32)
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

        slice_info = []
        for i in range(flair_slices.shape[0]):
            if slice_range is not None:
                if i < slice_range[0] or i > slice_range[1]:
                    continue

            flair_slice = flair_slices[i, :, :]
            mask_slice = mask_slices[i, :, :]
            is_empty = np.sum(mask_slice) == 0
            slice_info.append((i, flair_slice, mask_slice, is_empty))

        if max_empty_ratio is not None:
            non_empty_slices = [s for s in slice_info if not s[3]]
            empty_slices = [s for s in slice_info if s[3]]

            max_empty = int(max_empty_ratio * len(slice_info))
            np.random.shuffle(empty_slices)
            kept_empty_slices = empty_slices[:max_empty]

            final_slices = non_empty_slices + kept_empty_slices
            final_slices.sort(key=lambda x: x[0])
        else:
            final_slices = slice_info

        for i, flair_slice, mask_slice, _ in final_slices:
            flair_slice = (flair_slice - np.min(flair_slice)) / (np.ptp(flair_slice) + 1e-8)
            flair_slice = resize(flair_slice, target_size, preserve_range=True)
            flair_slice = (flair_slice * 255).clip(0, 255).astype(np.uint8)

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

        print(f"[✓] {patient} traité ({len(final_slices)} slices).")

def parse_args():
    parser = argparse.ArgumentParser(description="Slice .nii into PNGs with flexible structure.")
    parser.add_argument('--data_dir', type=str, required=True, help='Répertoire contenant les dossiers patients.')
    parser.add_argument('--output_dir', type=str, required=True, help='Dossier de sortie.')
    parser.add_argument('--orientation', type=str, choices=['axial', 'sagittal', 'coronal'], default='axial', help='Orientation des slices.')
    parser.add_argument('--structure', type=str, choices=['flat', 'per_patient', 'unext'], default='flat', help='Structure de sortie.')
    parser.add_argument('--size', type=int, nargs=2, default=[512, 512], help='Taille des images (largeur hauteur).')
    parser.add_argument('--max_empty_ratio', type=float, default=None, help='Proportion maximale de slices avec masque vide à conserver (entre 0 et 1).')
    parser.add_argument('--slice_range', type=int, nargs=2, default=None, help='Intervalle des slices à conserver (ex: --slice_range 40 90).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    flexible_slicer(
        args.data_dir,
        args.output_dir,
        args.orientation,
        args.structure,
        tuple(args.size),
        args.max_empty_ratio,
        args.slice_range
    )
