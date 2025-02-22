import os
import urllib.request


from PIL import Image
import numpy
import rembg
import torch



## SAMAPI: Segment Anything Model API
class SAMAPI:
    predictor = None

    @staticmethod
    def get_instance(sam_checkpoint=None):
        if SAMAPI.predictor is None:
            if sam_checkpoint is None:
                sam_checkpoint = "tmp/sam_vit_h_4b8939.pth"
            if not os.path.exists(sam_checkpoint):
                os.makedirs('tmp', exist_ok=True)
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    sam_checkpoint
                )
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_type = "default"

            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            predictor = SamPredictor(sam)
            SAMAPI.predictor = predictor
        return SAMAPI.predictor
    
    @staticmethod
    def segment_api(rgb, mask=None, bbox=None, sam_checkpoint=None):
        """

        Parameters
        ----------
        rgb : np.ndarray h,w,3 uint8
        mask: np.ndarray h,w bool

        Returns
        -------

        """
        np = numpy
        predictor = SAMAPI.get_instance(sam_checkpoint)
        predictor.set_image(rgb)
        if mask is None and bbox is None:
            box_input = None
        else:
            # mask to bbox
            if bbox is None:
                y1, y2, x1, x2 = np.nonzero(mask)[0].min(), np.nonzero(mask)[0].max(), np.nonzero(mask)[1].min(), \
                                 np.nonzero(mask)[1].max()
            else:
                x1, y1, x2, y2 = bbox
            box_input = np.array([[x1, y1, x2, y2]])
        masks, scores, logits = predictor.predict(
            box=box_input,
            multimask_output=True,
            return_logits=False,
        )
        mask = masks[-1]
        return mask

class ImageUtils:
    @staticmethod
    def expand2square(pil_img, background_color):
        """
        Expand image to square with background color.

        Args:
            pil_img (Image): Image to expand.
            background_color (tuple): Background color in RGB format.
        """
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    @staticmethod
    def segment_6imgs(zero123pp_imgs):
        """
        segment 6 images from zero123pp_imgs.

        Args:
            zero123pp_imgs (Image): Image to segment. (960 x 640)
        """

        imgs = [zero123pp_imgs.crop([0, 0, 320, 320]),
                zero123pp_imgs.crop([320, 0, 640, 320]),
                zero123pp_imgs.crop([0, 320, 320, 640]),
                zero123pp_imgs.crop([320, 320, 640, 640]),
                zero123pp_imgs.crop([0, 640, 320, 960]),
                zero123pp_imgs.crop([320, 640, 640, 960])]
        segmented_imgs = []
        for i, img in enumerate(imgs):
            output = rembg.remove(img)
            mask = numpy.array(output)[:, :, 3]
            mask = SAMAPI.segment_api(numpy.array(img)[:, :, :3], mask)
            data = numpy.array(img)[:,:,:3]
            data[mask == 0] = [255, 255, 255]
            segmented_imgs.append(data)
        result = numpy.concatenate([
            numpy.concatenate([segmented_imgs[0], segmented_imgs[1]], axis=1),
            numpy.concatenate([segmented_imgs[2], segmented_imgs[3]], axis=1),
            numpy.concatenate([segmented_imgs[4], segmented_imgs[5]], axis=1)
        ])
        return Image.fromarray(result)
    @staticmethod
    def segment_img(img: Image):
        output = rembg.remove(img)
        mask = numpy.array(output)[:, :, 3] > 0
        sam_mask = SAMAPI.segment_api(numpy.array(img)[:, :, :3], mask)
        segmented_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
        segmented_img.paste(img, mask=Image.fromarray(sam_mask))
        return segmented_img


