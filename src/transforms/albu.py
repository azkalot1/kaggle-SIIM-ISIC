import albumentations as A
from .extra_transfroms import AdvancedHairAugmentation, GridMask, Microscope


def get_valid_transforms():
    return A.Compose(
        [
            A.Normalize()
        ],
        p=1.0)


def light_training_transforms():
    return A.Compose([
        A.OneOf(
            [
                A.Transpose(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.NoOp()
            ], p=1.0),
        A.Normalize()
    ])


def medium_training_transforms():
    return A.Compose([
        A.OneOf(
            [
                A.Transpose(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.NoOp()
            ], p=1.0),
        A.OneOf(
            [
                GridMask(num_grid=6),
                A.CoarseDropout(max_holes=16, max_height=16, max_width=16),
                A.NoOp()
            ], p=1.0),
        AdvancedHairAugmentation(),
        Microscope(),
        A.Normalize()
    ])


def heavy_training_transforms():
    return A.Compose([
        A.OneOf(
            [
                A.Transpose(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.NoOp()
            ], p=1.0),
        A.OneOf(
            [
                A.ElasticTransform(),
                A.GridDistortion(),
                A.OpticalDistortion(),
                A.NoOp(),
                A.ShiftScaleRotate(),
            ], p=1.0),
        A.OneOf(
            [
                A.GaussNoise(),
                A.GaussianBlur(),
                A.NoOp()
            ], p=1.0),
        A.OneOf(
            [
                A.CLAHE(),
                A.RGBShift(),
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
                A.HueSaturationValue(),
                A.NoOp()
            ], p=1.0),
        A.OneOf(
            [
                GridMask(num_grid=6),
                A.CoarseDropout(max_holes=16, max_height=16, max_width=16),
                A.NoOp()
            ], p=1.0),
        AdvancedHairAugmentation(),
        Microscope(),
        A.Normalize()
    ])


def get_training_trasnforms(transforms_type):
    if transforms_type == 'light':
        return(light_training_transforms())
    elif transforms_type == 'medium':
        return(medium_training_transforms())
    elif transforms_type == 'heavy':
        return(heavy_training_transforms())
    else:
        raise NotImplementedError("Not implemented transformation configuration")
