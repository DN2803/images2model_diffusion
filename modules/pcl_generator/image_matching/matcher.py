from pathlib import Path
from modules.pcl_generator.image_matching.pairs_generator import PairsGenerator

def get_pairs_from_file(pair_file: str) -> list:
    pairs = []
    with open(pair_file, "r") as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            im1, im2 = line.strip().split(" ", 1)
            pairs.append((im1, im2))
    return pairs

class ImageMatcher (): 
    def __init__(self, **kwargs):
        self.pairs = None
        self.pair_file = None
    def generate_pairs(self, **kwargs) -> Path:
        """
        Generates pairs of images for matching.

        Returns:
            Path: The path to the pair file containing the generated pairs of images.
        """
        if self.pair_file is not None and self.matching_strategy == "custom_pairs":
            if not self.pair_file.exists():
                raise FileExistsError(f"File {self.pair_file} does not exist")

            pairs = get_pairs_from_file(self.pair_file)
            self.pairs = [
                (self.image_dir / im1, self.image_dir / im2) for im1, im2 in pairs
            ]

        else:
            pairs_generator = PairsGenerator(
                **kwargs,
            )
            self.pairs = pairs_generator.run()

        return self.pair_file
