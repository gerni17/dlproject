from os import path


def assert_matching_images(source, segmentations):
    for x, y in zip(source, segmentations):
        if path.basename(x) != path.basename(y):
            print(f"{path.basename(x)} vs {path.basename(y)}")
            raise RuntimeError('RGB image did not receive correct segmentation image')
