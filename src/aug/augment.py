import albumentations as album

class PairedSegAug:
    def __new__(cls, img_size: int, *args, **kwargs):
        return album.Compose([
            album.HorizontalFlip(p=0.6),
            album.VerticalFlip(p=0.5),
            album.RandomRotate90(p=0.5),
            album.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.7),
            album.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        ])
        

class VisualSegAug:
    def __new__(cls, *args, **kwargs):
        return album.Compose([
            album.RandomBrightnessContrast(p=0.5),
        ])