# Bachelor

Resolution of dataset in pixels (width x height):
- Jester: variable_width x 100
- Something-Something-V1: variable_width x 100
- Something-Something-V2: 178 x 100
- UCF101: 340 x 256
- HMDB51: 320 x 240

Backbones:
ResNet: ResNet-18, ResNet-34, ResNet-50, ResNet-101
MobileNet-V2
EfficientNet

You can train on GPU or TPU*.

# Args
- num_workers (-j)
_____________________________________________________________________________________________

*To train on TPU you should remove ".cuda()" from:
- main.py (lines: 66, 162, 227, 285)
- test_models.py (line 188)