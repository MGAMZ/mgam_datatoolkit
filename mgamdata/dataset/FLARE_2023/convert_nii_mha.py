import os
from tqdm import tqdm
import numpy as np
from mgamdata.dataset.base_convert import format_from_nnUNet
from mgamdata.io.sitk_toolkit import (
    sitk,
    sitk_resample_to_spacing,
    sitk_resample_to_size,
    sitk_resample_to_image,
)


class FLARE2023_formetter(format_from_nnUNet):
    def convert_one_sample(self, args):
        image_path, label_path, dest_folder, series_id, spacing, size = args
        convertion_log = {
            "img": os.path.relpath(image_path, self.args.data_root) if image_path is not None else None,
            "ann": os.path.relpath(label_path, self.args.data_root) if label_path is not None else None,
            "id": series_id,
        }

        # source path and output folder
        output_image_folder = os.path.join(dest_folder, "image")
        output_label_folder = os.path.join(dest_folder, "label")
        output_image_mha_path = os.path.join(output_image_folder, f"{series_id}.mha")
        output_label_mha_path = os.path.join(output_label_folder, f"{series_id}.mha")
        os.makedirs(output_image_folder, exist_ok=True)
        os.makedirs(output_label_folder, exist_ok=True)
        if os.path.exists(output_image_mha_path):
            if label_path is None \
            or os.path.exists(output_label_mha_path) \
            or not os.path.exists(label_path):
                return convertion_log

        try:

            if isinstance(image_path, str) and ".dcm" in image_path:
                input_image_mha, input_label_mha = self.convert_one_sample_dcm(image_path, label_path)
            elif ".nii.gz" in image_path:
                input_image_mha, input_label_mha = self.convert_one_sample_nii(image_path, label_path)
            if input_image_mha is None:
                convertion_log["id"] = "error"
                convertion_log["error"] = "No image found."
                return convertion_log

            # resample
            if spacing is not None:
                assert size is None, "Cannot set both spacing and size."
                input_image_mha = sitk_resample_to_spacing(input_image_mha, spacing, "image")
                if not isinstance(input_image_mha, sitk.Image):
                    convertion_log["id"] = "error"
                    convertion_log["error"] = "Resample to spacing failed."
                    convertion_log["resample_error_detail"] = input_image_mha
                    return convertion_log
            elif size is not None:
                assert spacing is None, "Cannot set both spacing and size."
                input_image_mha = sitk_resample_to_size(input_image_mha, size, "image")

            # Align label to image, if label exists.
            if input_label_mha is not None and os.path.exists(label_path):
                input_label_mha = sitk_resample_to_image(input_label_mha, input_image_mha, "label")

            input_image_mha = sitk.DICOMOrient(input_image_mha, 'LPI')
            sitk.WriteImage(input_image_mha, output_image_mha_path, useCompression=True)
            if input_label_mha is not None and os.path.exists(label_path):
                assert (input_image_mha.GetSize() == input_label_mha.GetSize()), \
                    f"Image {input_image_mha.GetSize()} and label {input_label_mha.GetSize()} size mismatch."
                
                # NOTE FLARE2023 contains partially annotated labels,
                # if the number of unique label <= 4, the label will be deprecated.
                lbl_arr = sitk.GetArrayFromImage(input_label_mha)
                unique_labels = np.unique(lbl_arr)
                if len(unique_labels) <= 3:
                    tqdm.write(f"{series_id} | Label {unique_labels} is deprecated.")
                    convertion_log["label_deprecated"] = str(unique_labels)
                    input_label_mha = None
                else:
                    input_label_mha = sitk.DICOMOrient(input_label_mha, 'LPI')
                    sitk.WriteImage(input_label_mha, output_label_mha_path, useCompression=True)
        
        except Exception as e:
            convertion_log["id"] = "error"
            convertion_log["error"] = str(e)
            error_info = f"SeriesUID{series_id} | " + str(e)
            convertion_log["Unknown_error_detail"] = error_info
            print(error_info)
        
        return convertion_log


if __name__ == "__main__":
    formatter = FLARE2023_formetter()
    formatter.execute()
