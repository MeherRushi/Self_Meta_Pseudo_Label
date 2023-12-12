import torchvision.transforms as transforms

def get_transform():
    """
    Function to get data transformation for training images.
    """
    transform = transforms.Compose([
        transforms.AutoAugment(),
        transforms.RandAugment(),
        transforms.Resize((32, 32)),
        # transforms.RandomAutocontrast(p= 0.5),
        # transforms.ColorJitter(brightness=0.1, contrast=0.2),
        # transforms.RandomResizedCrop(size=(32,32)),
        # transforms.RandomEqualize(p= 0.5),
        # transforms.RandomInvert(p=0.5),
        # transforms.RandomAdjustSharpness(sharpness_factor=0.5),
        # transforms.RandomAffine(degrees=12.5,translate=[0.5,0.5],shear=[0.5,0.5]),
        # transforms.RandomRotation(degrees=12.5),
        # transforms.RandomPosterize(bits=8,p= 0.5),
        # transforms.RandomSolarize(threshold = 0.5,p= 0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def simple_transform():
    """
    Function to get a simple data transformation for images.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform
