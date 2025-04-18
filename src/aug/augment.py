import albumentations as album

class PairedSegAug:
    def __new__(cls, img_size: int, *args, **kwargs):
        return album.Compose([
            album.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            album.HorizontalFlip(p=0.5),
            album.VerticalFlip(p=0.5),
            album.RandomRotate90(),
        ])
        

class VisualSegAug:
    def __new__(cls, *args, **kwargs):
        return album.Compose([
            album.RandomBrightnessContrast(p=0.5),
            album.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,p=0.5
            ),            
        ])