import argparse
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from utils.image import ImageUtils
from PIL import Image

from modules.pcl_generator.main import PCL

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse. May be different from the original implementation')
parser.add_argument('--model_extractor', default='aliked', help='select extroctor model')
parser.add_argument('--model_matcher', default='diffglue', help='select matcher model')
parser.add_argument('--input_scans_path', help='directory of input scans')
parser.add_argument('--output_scans_path', help='directory of output scans')
# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])


@dataclass
class GeneralConfig:
    matching_strategy: str = "bruteforce"
    pair_file: str = None
    retrieval: bool = False
    overlap: float = 0.5
    db_path: str = None
    upright: bool = False
    verbose: bool = True


@dataclass
class ExtractorConfig:
    name: str = args.model_extractor
    max_keypoints: int = 4000

@dataclass
class MatcherConfig:
    name: str = args.model_matcher
    local_features: str = args.model_extractor
    features: str = args.model_extractor
    input_dim: int = 128 if local_features == 'aliked' else 256

@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
config = Config()
print(asdict(config))
# 

def viz():
    input_scans_path = Path(args.input_scans_path)
    output_scans_path = Path(args.output_scans_path)
    scan_list = [d for d in os.listdir(input_scans_path) if os.path.isdir(os.path.join(input_scans_path, d))]
    if not scan_list:
        print("No scans found in the input directory.")
        return

    print("Danh sách scan:", scan_list)


    for scan in scan_list:
        print("Vizualization for scan:", scan)
        # load image 
        images_path = input_scans_path / scan / 'image'    
        input_img = list(images_path.glob('*.png')) + list(images_path.glob('*.jpg')) + list(images_path.glob('*.jpeg'))
        output_path = output_scans_path / scan / 'viz'
        used_seg_dir = output_path / 'segmentation'
        if used_seg_dir.exists() and any(used_seg_dir.iterdir()):
            print(f"Scan '{scan}' đã được xử lý, bỏ qua.")
            continue

        os.makedirs(used_seg_dir, exist_ok=True)

        for i, img_tuple in enumerate(input_img):
            img = Image.open(img_tuple)
            img.save(output_path / f"input_{i}.png")
            #TODO call segmentation API
            seg_img = ImageUtils.segment_img(img)
            seg_img.save(os.path.join(used_seg_dir, f"seg_{i}.png"))

        
        pcl_gen = PCL(used_seg_dir, output_path)
        pcl_gen.generate(config=config)
        


        
    print("Vizualization")

if __name__ == '__main__':
    viz()