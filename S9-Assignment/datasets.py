from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Custom Class to work with albumentations lib
class CIFAR10Custom(datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        if(transform is None and train):
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
                    A.CoarseDropout (max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=0.4734, mask_fill_value = None),
                    A.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010),p =1.0),
                    ToTensorV2()
                ]
            )
        elif(transform is None):
            transform = A.Compose(
                [
                    A.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010),p =1.0),
                    ToTensorV2()
                ]
            )
        super().__init__(root=root, train=train, download=download, transform=transform)
        
    def __getitem__(self, index):
        image, lab = self.data[index], self.targets[index]
        if(self.transform is not None):
            image = self.transform(image=image)
        return image['image'], lab