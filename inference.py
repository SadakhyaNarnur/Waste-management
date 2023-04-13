import os
import numpy as np
import cv2
import torch
import glob as glob
import matplotlib.pyplot as plt
from model import create_model
from pprint import pprint
import pandas as pd
import xml.etree.ElementTree as ETree
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
OUT_DIR = os.path.join(os.path.dirname(__file__),"../outputs")
model = create_model(num_classes=7).to(device)
model.load_state_dict(torch.load(
    os.path.join(OUT_DIR,'model26.pth'), map_location=device
))
model.eval()
DATA_PATH = os.path.join(os.path.dirname(__file__),"../data")
DIR_TEST_PRED = os.path.join(DATA_PATH, "test_predictions")
# directory where all the images are present
DIR_TEST = os.path.join(DATA_PATH,'test')
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")

# classes: 0 index is reserved for background
CLASSES = [
    'background', 'paper','glass','plastic','cardboard','metal','trash'
]
class_dict = {'paper':1,'glass':2,'plastic':3,'cardboard':4,'metal':5,'trash':6}
# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.8
pred_box = []
pred_score = []
pred_label = []
targ_box = []
targ_label = []
for i in range(len(test_images)):
    if test_images[i].endswith(".xml"):
        l = []
        xml_data = 'src/../data/test/plastic236_jpg.rf.213ae70bfd2e1ae5a45c47cbbd30eb1b.xml'
        prstree = ETree.parse(xml_data)
        root = prstree.getroot()
        l.extend([int(root[6][5][0].text),int(root[6][5][2].text),int(root[6][5][1].text),int(root[6][5][3].text)])
        targ_box.append(l)
        targ_label.append(class_dict[root[6][0].text])
        continue
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    print(test_images[i], os.path.exists(test_images[i]))
    image = cv2.imread(test_images[i],0)
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)
    
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    print("Prediction count",len(outputs[0]['boxes']))
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        pred_scores = scores[scores >= detection_threshold].astype(np.int32).copy()
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        print(draw_boxes)
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 255), 2)
            cv2.putText(orig_image, f'{pred_classes[j]}({scores[j]:.2f})', 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)
            pred_box.append([int(box[0]), int(box[1]),int(box[2]), int(box[3])])
            pred_score.append(scores[j])
            pred_label.append(class_dict[pred_classes[j]])
        plt.imshow(orig_image)
        dst_path = os.path.join(DIR_TEST_PRED, f"{image_name}.jpg")
        cv2.imwrite(dst_path, orig_image)
        print(dst_path)
        plt.show()
    print(f"Image {i+1} done...")
    print('-'*50)

preds = [
    dict(
        boxes=torch.tensor(pred_box),
        scores=torch.tensor(pred_score),
        labels=torch.tensor(pred_label)
    )
]
target = [
    dict(
        boxes=torch.tensor(targ_box),
        labels=torch.tensor(targ_label)
    )
]
metric = MeanAveragePrecision()
metric.update(preds, target)
pprint(metric.compute())
print('TEST PREDICTIONS COMPLETE')
# cv2.destroyAllWindows()