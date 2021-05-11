# Bachelor

#num_workers (-j)

Resolution of dataset in pixels (width x height):
- Jester: 176 x 100
- Something-Something-V1: 176 x 100
- Something-Something-V2: 178 x 100

Backbones:
ResNet: ResNet-18, ResNet-34, ResNet-50, ResNet-101
MobileNet-V2*
EfficientNetB0

*To train datasets with MobileNet-V2 architecture you should comment the lines 144-145 from:
temporal-shift-module-master/ops/models.py