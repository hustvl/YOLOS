import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns
from numpy.lib.npyio import load
import pandas as pd
# Settings
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
DET_TOKEN_NUM=100

parser = argparse.ArgumentParser()
parser.add_argument('--visjson', default='', type=str, help="the json path to visualize")
parser.add_argument('--cococlsjson', default='vis_token_dist/coco-valsplit-cls-dist.json', type=str, help="the json path to visualize")

args, unparsed = parser.parse_known_args()

with open(args.visjson,'r') as load_f:
    load_dicts = json.load(load_f)

bot_thresh = 32*32
top_thresh = 96*96
size_colors = ['b','g','r']
tokens_list = [[] for j in range(DET_TOKEN_NUM)]
token_index = 0
img_id = load_dicts[0]['image_id']
for load_dict in load_dicts:
    if img_id == load_dict['image_id']:
        tempdic = {}
        tempdic['cx'] = load_dict['bbox'][0]
        tempdic['cy'] = load_dict['bbox'][1]
        area = load_dict['bbox'][2] * load_dict['bbox'][3]
        tempdic['area'] = area
        tempdic['category_id'] = load_dict['category_id']
        tempdic['category'] = CLASSES[int(load_dict['category_id'])]
        if area <= bot_thresh:
            tempdic['color_idx'] = 0 # small
        elif bot_thresh < area < top_thresh:
            tempdic['color_idx'] = 1 # medium
        elif area >= top_thresh:
            tempdic['color_idx'] = 2 # large
        tokens_list[token_index].append(tempdic)
        token_index = token_index + 1
    else:
        img_id = load_dict['image_id']
        token_index = 0
        tempdic = {}
        tempdic['cx'] = load_dict['bbox'][0]
        tempdic['cy'] = load_dict['bbox'][1]
        area = load_dict['bbox'][2] * load_dict['bbox'][3]
        tempdic['area'] = area
        tempdic['category_id'] = load_dict['category_id']
        tempdic['category'] = CLASSES[int(load_dict['category_id'])]
        if area <= bot_thresh:
            tempdic['color_idx'] = 0 # small b
        elif bot_thresh < area < top_thresh:
            tempdic['color_idx'] = 1 # medium g
        elif area >= top_thresh:
            tempdic['color_idx'] = 2 # large r 
        tokens_list[token_index].append(tempdic)
        token_index = token_index + 1
tokens_df = [None for j in range(DET_TOKEN_NUM) ]
for i in range(DET_TOKEN_NUM):
    tokens_df[i] = pd.DataFrame(tokens_list[i])

# draw bbox dist
fig, axs = plt.subplots(ncols=10, nrows=1, figsize=(22, 2), facecolor='white',tight_layout=True)
for index, ax in enumerate(axs):
    for index, row in tokens_df[index].iterrows():
        # ax.scatter(int(100*row['cx']), int(100*row['cy']), c=size_colors[int(row['color_idx'])], cmap='brg', s=40, alpha=0.2, marker='8', linewidth=0)
        # ax.scatter(int(100*row['cx']), int(100*row['cy']), c=size_colors[int(row['color_idx'])], cmap='brg', s=5, alpha=0.3, linewidth=0)  
        ax.scatter(100*row['cx'], 100*row['cy'], c=size_colors[int(row['color_idx'])], cmap='brg', s=5, alpha=0.3, linewidth=0)  
        # ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

filename = os.path.splitext(args.visjson)[0]
filename = filename + '-bbox.png'
fig.savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none')

print("%s done" % filename)
# draw category dist
plt.close(fig)
with open(args.cococlsjson,'r') as load_f:
    coco_load_dicts = json.load(load_f)

valcls_list=[]
for load_dict in coco_load_dicts:
    tempdic = {}
    tempdic['category_id'] = load_dict['category_id']
    tempdic['category'] = CLASSES[int(load_dict['category_id'])]
    valcls_list.append(tempdic)
valcls_df = pd.DataFrame(valcls_list)



cate_list=[]
for cls in CLASSES:
    if cls !='N/A':
        cate_list.append(cls)
# cate_list = valcls_df['category'].value_counts().keys().tolist()
nums_list=[[] for j in range(DET_TOKEN_NUM)]

for i in range(DET_TOKEN_NUM):
    for cate in cate_list:
        catekeys = tokens_df[i]['category'].value_counts().keys()
        if cate in catekeys:
            nums_list[i].append(tokens_df[i]['category'].value_counts()[cate])
        else:
            nums_list[i].append(0)
        # nums_list[i].append(tokens_df[i]['category'].value_counts()[cate])

tokens_cls_df = [None for j in range(DET_TOKEN_NUM) ]

for i in range(DET_TOKEN_NUM):
    temp_dic = {}
    temp_dic['category'] = cate_list
    # lognum_list = [math.log(num+1) for num in nums_list[i]]
    # temp_dic['num'] = lognum_list
    num_list = [ num+1 for num in nums_list[i]]
    # temp_dic['#objects'] = nums_list[i]
    temp_dic['#objects'] = num_list
    temp_str = "Det-Tok#%d" %(i)
    temp_dic['dettoken_index'] = [temp_str] * len(cate_list)
    tokens_cls_df[i] = pd.DataFrame(temp_dic)
merge_tokens_cls_df = pd.concat(tokens_cls_df)

coco_nums_list = []
for cate in cate_list:
    catekeys = valcls_df['category'].value_counts().keys()
    if cate in catekeys:
        coco_nums_list.append(valcls_df['category'].value_counts()[cate])
    else:
        coco_nums_list.append(0)

cocotemp_dic = {}
cocotemp_dic['category'] = cate_list
# lognum_list = [math.log(num+1) for num in nums_list[i]]
# temp_dic['num'] = lognum_list
coco_num_list = [ num+1 for num in coco_nums_list]
# temp_dic['#objects'] = nums_list[i]
cocotemp_dic['#objects'] = coco_num_list
vis_valcls_df = pd.DataFrame(cocotemp_dic)

fig2 = plt.figure(1,figsize=(22,6), facecolor='white',tight_layout=True)

g = sns.lineplot(data=merge_tokens_cls_df, x="category", y="#objects", label='Det-Tok dist')

val_1 = sns.lineplot(data=vis_valcls_df, x="category", y="#objects",palette=['red'],color = 'red',
                 markersize=40, markers='+', label='COCO val dist')
val_2 = sns.scatterplot(data=vis_valcls_df, x="category", y="#objects",palette=['red'], color = 'red')

g.set_ylim(1 , 13000)
# h.set_ylim(1 , 13000)
val_1.set_ylim(1 , 13000)
val_2.set_ylim(1 , 13000)
# g.set(ylim=(0, 2000))
g.set(yscale="log")
# h.set(yscale="log")
val_1.set(yscale="log")
val_2.set(yscale="log")
plt.xticks(rotation=-90)  

# # for j in range(5):
# sns.pointplot(data=tokens_df[0],x='category',)
filename = os.path.splitext(args.visjson)[0]
filename = filename + '-all-tokens-cls.png'
fig2.savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none')
print("%s done" % filename)