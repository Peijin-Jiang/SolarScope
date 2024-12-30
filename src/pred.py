import os
import timm
import torch
import numpy as np
import pandas as pd

from typing import List
from PIL import Image
from tqdm import tqdm
from torch import nn
from torchvision import transforms
from data import get_bounding_box # type: ignore
from models import load_model # type: ignore
from monai import transforms as monai_transforms # type: ignore
from torch.utils.data import Dataset, DataLoader
from classify import make_models_transforms # type: ignore


class ImageDataset(Dataset):
    def __init__(self, 
                images,
                image_size: int=1024,
                mask_size: int=256,
                ignore_index: int=255,
                normalize: bool=True,
                transform=None,
                classifier_transform=None,
                folder_names=None,
                model_name="sam"):
        """This class is used to load the segmentation dataset"""
        # load the dataset csv
        if isinstance(images, str):
          self.img_paths = os.listdir(images)
          self.img_paths = [os.path.join(images, f) for f in self.img_paths]
        else:
          self.img_paths = images
        # set the ignore index
        self.ignore_index = ignore_index
        # set the normalization
        self.normalize = normalize
        # set the transform
        self.transform, self.classifier_transform = transform, classifier_transform
        # set the image and mask size
        self.image_size, self.mask_size = image_size, mask_size
        # set the model name
        self.model_name = model_name

    def __len__(self):
        return len(self.img_paths)

    def __get_one_item__(self, idx):
        img_path = self.img_paths[idx]
        image_pil = Image.open(img_path).convert('RGB')
        original_image = torch.tensor(np.array(image_pil))
        original_image = original_image.permute(2,0,1)
        classifier_image = self.classifier_transform(image_pil)

        resize_func = monai_transforms.Compose([
            monai_transforms.Resized(keys=['image'], spatial_size=(self.image_size, self.image_size), mode=['bilinear']),
        ])
        transformed = resize_func({"image": original_image})
        raw_image = transformed['image'].squeeze(0)
        
        image = raw_image.permute(1, 2, 0).numpy().astype(np.uint8)

        # assert target.max() <= 1
        prompt = get_bounding_box(1024)
        
        if "sam" in self.model_name:
          input_dict = self.transform(image, input_boxes=[[prompt]], return_tensors="pt")
          input_dict = {k: v.squeeze(0) for k, v in input_dict.items()}
        elif "deeplab" in self.model_name or "unet" in self.model_name:
          image = self.transform(image, return_tensors="pt")
          # if image.max() > 1:
          #   image = image / 255.0
          input_dict = {"pixel_values": torch.from_numpy(image).permute(2, 0, 1).float()}
        elif "segformer" in self.model_name:
          image = self.transform(image, return_tensors="pt").pixel_values.squeeze()
          input_dict = {"pixel_values": image}
        else:
          raise NotImplementedError
        input_dict["classifier_images"] = classifier_image
        input_dict["raw_images"] = raw_image.permute(1, 2, 0)
        input_dict["image_names"] = os.path.basename(img_path)
        return input_dict
    
    def __getitem__(self, idx):
        return self.__get_one_item__(idx)


def overlay_masks_on_images(images, masks):
    overlayed_images = []
    for img, mask in zip(images, masks):
        img_pil = Image.fromarray(img.astype(np.uint8))
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        red_mask = Image.new("RGBA", mask_pil.size, (255, 0, 0, 0))
        red_mask.paste((255, 0, 0, 128), mask_pil)  # 128 = 0.5 alpha in [0, 255]

        img_pil = img_pil.convert("RGBA").resize(red_mask.size)
        overlayed_img = Image.alpha_composite(img_pil, red_mask)
        overlayed_images.append(overlayed_img)
    return overlayed_images


# convert raw images to classifier
def classifier_transform(images: np.ndarray, image_size: int=224):
    VAL_TRANSFORM = transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])
    images = [VAL_TRANSFORM(Image.fromarray(img.astype(np.uint8))) for img in images]
    images = torch.stack(images)
    return images
    

import os
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


def run_prediction(
    model_path: str,
    classifier_path: str,
    classifier_name: str,
    images,
    save_dir: str,
    model_name="sam",
    batch_size=8,
    device="cuda",
    save_mask=False,
    classifier_threshold=0.5,
    seg_threshold=0.05,
    enable_geo=True
):
    os.makedirs(save_dir, exist_ok=True)

    # Create classifier
    if "dinov2" in classifier_name:
        classifier_image_size = 518
    else:
        classifier_image_size = 518
    classifier, _, classifier_transform = make_models_transforms(
        classifier_name, classifier_image_size, True
    )
    classifier_state_dict = torch.load(classifier_path)
    classifier.load_state_dict(classifier_state_dict)
    classifier = classifier.cuda()
    print("Loaded classifier")

    # Load model
    model, processor, image_size, mask_size = load_model(model_name, device)
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)
    print("Loaded model")
    classifier.eval()
    model.eval()

    # Create data
    pred_data = ImageDataset(
        images=images,
        transform=processor,
        classifier_transform=classifier_transform,
        model_name=model_name,
        image_size=image_size,
    )
    pred_dl = DataLoader(pred_data, batch_size=batch_size, num_workers=10, shuffle=False, drop_last=False)

    # Constants for calculating the step size
    EARTH_CIRCUMFERENCE = 40075017  # Earth's circumference in meters
    ZOOM_LEVEL = 20  # Zoom level for the map
    COORD_SCALE = 2  # Adjust for compressed image size

    # Function to calculate meters per pixel at a given latitude
    def meters_per_pixel(latitude, zoom):
        return (EARTH_CIRCUMFERENCE * math.cos(math.radians(latitude))) / (2 ** (zoom + 8))

    # Calculate the approximate size (in degrees) covered by each pixel at a given latitude and zoom
    def pixel_to_geo_step(latitude, zoom):
        meters_per_pixel_value = meters_per_pixel(latitude, zoom)
        lat_step = meters_per_pixel_value / 111320  # Convert meters to degrees latitude
        lon_step = meters_per_pixel_value / (111320 * math.cos(math.radians(latitude)))  # Adjust for longitude
        return lat_step, lon_step

    # Initialize stats
    positive_stats = {
        "image_names": [],
        "latitude,longitude": [],
        "mask_area": [],
    }
    prediction = {
        "image_names": [],
        "preds": [],
        "logits": [],
        "seg_ratio": [],
    }

    # Main processing loop
    with torch.no_grad():
        for batch_idx, val_batch in tqdm(enumerate(pred_dl)):
            classifier_input = val_batch["classifier_images"].cuda()
            classifier_logits = classifier(classifier_input)
            classifier_output = (torch.sigmoid(classifier_logits) > classifier_threshold).int().cpu().numpy()

            # Run the segmentation model
            if "sam" in model_name:
                outputs = model(
                    pixel_values=val_batch["pixel_values"].to(device),
                    input_boxes=val_batch["input_boxes"].to(device),
                    multimask_output=False,
                )
                predicted_masks = outputs.pred_masks.squeeze()
            elif "segformer" in model_name:
                outputs = model(val_batch["pixel_values"].to(device))
                predicted_masks = outputs.logits.squeeze(1)
            else:
                outputs = model(val_batch["pixel_values"].to(device))
                predicted_masks = outputs.squeeze(1)
            pred_mask = (predicted_masks.detach().cpu() > 0.0).numpy()
            print(pred_mask.shape)

            # Retrieve batch information
            image_names = val_batch["image_names"]
            raw_images = val_batch["raw_images"].float().cpu().numpy()
            image_with_mask = overlay_masks_on_images(raw_images, pred_mask)

            for i in range(len(image_names)):
                try:
                    # Record prediction results
                    prediction["image_names"].append(image_names[i])
                    prediction["preds"].append(classifier_output[i])
                    prediction["logits"].append(classifier_logits.cpu().numpy()[i])
                    seg_ratio = np.sum(pred_mask[i]) / (pred_mask[i].shape[0] * pred_mask[i].shape[1])
                    prediction["seg_ratio"].append(seg_ratio)

                    # save masks
                    save_path = os.path.join(save_dir, image_names[i])
                    if save_mask:
                        np.save(save_path + ".npy", pred_mask[i])
                    # Process positive predictions
                    if not enable_geo:
                      continue
                    if classifier_output[i, 0] == 1 and seg_ratio > seg_threshold:
                        save_path = os.path.join(save_dir, image_names[i])
                        image_with_mask[i].save(save_path)
                        if save_mask:
                            np.save(save_path + ".npy", pred_mask[i])

                        # Calculate mask center coordinates
                        y_indices, x_indices = np.where(pred_mask[i] == 1)
                        coord_x = np.mean(x_indices)
                        coord_y = np.mean(y_indices)

                        # Parse geographical information
                        geo_info = image_names[i].replace(".png", "").split("_")[1:]
                        if len(geo_info) != 3:
                            raise ValueError(f"Invalid geo_info format for image: {image_names[i]}")

                        center_lat = float(geo_info[0])
                        center_lon = float(geo_info[1])
                        quadrant = int(geo_info[2])  # Extract the quadrant

                        # Calculate step sizes in degrees per pixel
                        lat_step, lon_step = pixel_to_geo_step(center_lat, ZOOM_LEVEL)

                        # Convert pixel offsets to geographical offsets based on quadrant
                        if quadrant == 1:  # Right bottom corner
                            delta_lat = (mask_size-1-coord_y) * lat_step * COORD_SCALE
                            delta_lon = (coord_x-mask_size+1) * lon_step * COORD_SCALE
                        elif quadrant == 2:  # Left bottom corner
                            delta_lat = (mask_size-1-coord_y) * lat_step * COORD_SCALE
                            delta_lon = coord_x * lon_step * COORD_SCALE
                        elif quadrant == 3:  # Right top corner
                            delta_lat = -coord_y * lat_step * COORD_SCALE
                            delta_lon = (coord_x-mask_size+1) * lon_step * COORD_SCALE
                        elif quadrant == 4:  # Left top corner
                            delta_lat = -coord_y * lat_step * COORD_SCALE
                            delta_lon = coord_x * lon_step * COORD_SCALE
                        else:
                            raise ValueError(f"Invalid quadrant value: {quadrant}")

                        # Calculate the final latitude and longitude
                        final_lat = center_lat + delta_lat
                        final_lon = center_lon + delta_lon

                        # Combine latitude and longitude into a single field
                        lat_lon = f"{final_lat},{final_lon}"

                        # Calculate actual mask area in square meters
                        pixel_area = meters_per_pixel(center_lat, ZOOM_LEVEL) ** 2
                        actual_mask_area = np.sum(pred_mask[i]) * pixel_area * (COORD_SCALE ** 2)

                        # Append results to positive_stats
                        positive_stats["image_names"].append(image_names[i].replace(".png", ""))
                        positive_stats["latitude,longitude"].append(lat_lon)
                        positive_stats["mask_area"].append(actual_mask_area)

                except Exception as e:
                    print(f"Error processing image {image_names[i]}: {e}")
                    continue

    # Summary and save results
    print(
        "Number of positive images: {} in {} processed images".format(
            len(positive_stats["image_names"]), batch_idx * batch_size + len(image_names)
        )
    )

    # Save the stats
    positive_stats_df = pd.DataFrame(positive_stats)
    positive_stats_df.to_csv(os.path.join(save_dir, "positive_stats.csv"), index=False)
    prediction_df = pd.DataFrame(prediction)
    prediction_df.to_csv(os.path.join(save_dir, "prediction.csv"), index=False)

    # Return results
    return prediction_df, positive_stats_df

    



                    