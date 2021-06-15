import albumentations as A

transform = A.Compose([
    A.Resize(256, 256, always_apply=True),
    A.CenterCrop(224, 224, always_apply=True),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True)
])

with open('categories/category.txt', 'r') as f:
        class_names = f.readlines()
        f.close()