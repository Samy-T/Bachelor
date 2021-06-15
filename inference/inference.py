from typing import no_type_check
import torch
import torchvision
import argparse
import cv2
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
import utils
from ts import make_temporal_shift
#from efficientnet_pytorch import EfficientNet
#from ts import TemporalShift
#from efficientnet_pytorch.model import MBConvBlock

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-f', '--number_of_frames', dest='number_of_frames', default=8, type=int, help='number of frames to consider for each predicition')
args = vars(parser.parse_args())

class_names = utils.class_names
num_classes = len(class_names)
device = torch.device('cuda' if torch.cuda. is_available() else 'cpu')

print(f"Number of frames to consider for each prediction: {args['number_of_frames']}")

# Backbone
# ResNet
model = torchvision.models.resnet34(pretrained=True, progress=True)
make_temporal_shift(model, n_segment=args['number_of_frames'], n_div=args['number_of_frames'], place='blockres', temporal_pool=False)
model.fc = nn.Linear(512, num_classes) # Here you should change output corresponding to the number of action classes: e.g. ResNet-50: nn.Linear(2048, num_classes)

# Load your pretrained model
checkpoint = torch.load('ResNet34_Jester.pth.tar')
checkpoint = checkpoint['state_dict']

# EfficientNetB4
""" model = EfficientNet.from_pretrained('efficientnet-b4', num_classes) # Here you should change output corresponding to the number of action classes
for m in model._blocks:
    if isinstance(m, MBConvBlock):
        m._depthwise_conv = TemporalShift(m._depthwise_conv, n_segment=args['frames'], n_div=args['frames'])
# Load your pretrained model
checkpoint = torch.load('EfficientNet_Jester.pth.tar')
checkpoint = checkpoint['state_dict'] """

base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias',
                }
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)

# for ResNet
base_dict = {k.replace('new_fc', 'fc'): v for k, v in list(base_dict.items())}

# for EfficientNetB4
#base_dict = {k.replace('new_fc', '_fc'): v for k, v in list(base_dict.items())}

new_state_dict = {}
for k, v in base_dict.items():
    if k[:11] == 'base_model.':
        name = k[11:]  # remove `base_model.`
    else:
        name = k
    new_state_dict[name] = v

# Load weights from pretrained model
model.load_state_dict(new_state_dict)

model.avgpool = nn.AdaptiveAvgPool2d(1)

# Set model for inference
model=model.eval().to(device)

# Process frames
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# W x H of frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Save output video
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

frame_count = 0
total_fps = 0
frames = []

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        start_time = time.time()
        image = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert frame to RGB
        frame = utils.transform(image=frame)['image']
        frames.append(frame)
        if len(frames) == args['number_of_frames']:
            with torch.no_grad(): # Disabled gradient calculation
                input_frames = np.array(frames)
                input_frames = np.transpose(input_frames, (0, 3, 1, 2)) # (batch, color_channel, H, W)
                input_frames = torch.tensor(input_frames, dtype=torch.float32)                
                input_frames = input_frames.to(device)                
                outputs = model(input_frames)
                _, preds = torch.max(outputs.data, 1)
                probs = torch.nn.Softmax(dim=1)(outputs)
                pred = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

                label = class_names[pred].strip()
                print(label)
            end_time = time.time()

            fps = 1 / (end_time - start_time)

            total_fps += fps
            frame_count += 1

            wait_time = max(1, int(fps/4))
            cv2.putText(image, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            frames.pop(0)
            cv2.imshow('image', image)
            out.write(image)

            if cv2.waitKey(wait_time) &0xFF == ord('q'):
                break
    else:
        break

cap.release()
cv2.destroyAllWindows()

avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
