import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import segmentation_models_pytorch as smp

from transformers import SamModel, SamProcessor


def load_model(model="sam", device="cuda"):
  if model == "sam":
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = model.to(device)
    image_size, mask_size = 1024, 256
    return model, processor, image_size, mask_size
  elif model == "deeplabv3-resnet101":
    model = smp.DeepLabV3Plus(
        encoder_name='resnet101', 
        encoder_weights='imagenet', 
        classes=1
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet101', 'imagenet')
    model = model.to(device)
    image_size, mask_size = 512, 512
    return model, preprocessing_fn, image_size, mask_size
  elif model == "unet":
    model = smp.Unet(
    encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet101', 'imagenet')
    model = model.to(device)
    image_size, mask_size = 512, 512
    return model, preprocessing_fn, image_size, mask_size
  elif model == "segformer-b1":
    processor = SegformerImageProcessor(do_resize=False)
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b1",
                                                            num_labels=1,
                                                            id2label={1: "solar_pv"},
                                                            label2id={"solar_pv": 1})
  elif model == "segformer-b5":
    processor = SegformerImageProcessor(do_resize=False)
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                            num_labels=1,
                                                            id2label={1: "solar_pv"},
                                                            label2id={"solar_pv": 1})
    model = model.to(device)
    image_size, mask_size = 640, 160
    return model, processor, image_size, mask_size
  else:
    raise NotImplementedError