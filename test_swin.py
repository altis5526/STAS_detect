"""
指令: python train_swin.py
"""
import copy
import os.path as osp
import json
import mmcv
import numpy as np
import torch 
import os
import cv2
import bbox_visualizer as bbv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.datasets.builder import DATASETS
from mmcv import Config
from mmdet.datasets.custom import CustomDataset
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv_custom.checkpoint import load_checkpoint, load_state_dict

"""創建資料集"""
@DATASETS.register_module()
class STASDataset(CustomDataset):

    CLASSES = ("stas",)

    def load_annotations(self, ann_file):
        # load image list from file
        image_list = mmcv.list_from_file(self.ann_file)
    
        data_infos = []
        # convert annotations to middle format
        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}.jpg'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
    
            data_info = dict(filename=f'{image_id}.jpg', width=width, height=height)
    
            # load annotations

            lines = mmcv.list_from_file(osp.join('STAS_dataset/training/Train_Annotations', f'{image_id}.txt'))
            content = [line.strip().split(' ') for line in lines]
            bboxes = [[float(info) for info in x[1:]] for x in content]
            
            gt_bboxes = []
            gt_labels = []
    
            # filter 'DontCare'
            for bbox in bboxes:
                gt_labels.append(0)
                gt_bboxes.append(bbox)
            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long),
            )

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos

"""設定config"""
def procedure(test_batch_size=256, id=0):
    cfg = Config.fromfile('./configs/swin/cascade_base_900.py')


    # Modify dataset type and path
    cfg.dataset_type = 'STASDataset'
    cfg.data_root = 'STAS_dataset/'

    cfg.data.test.type = 'STASDataset'
    cfg.data.test.data_root = 'STAS_dataset/'
    cfg.data.test.ann_file = f'train{id}.txt'
    cfg.data.test.img_prefix = 'training/Train_Images'

    cfg.data.train.type = 'STASDataset'
    cfg.data.train.data_root = 'STAS_dataset/'
    cfg.data.train.ann_file = f'train{id}.txt'
    cfg.data.train.img_prefix = 'training/Train_Images'

    cfg.data.val.type = 'STASDataset'
    cfg.data.val.data_root = 'STAS_dataset/'
    cfg.data.val.ann_file = f'valid{id}.txt'
    cfg.data.val.img_prefix = 'training/Train_Images'

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    # cfg.load_from = 'checkpoints/cascade_mask_rcnn_swin_base_patch4_window7.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = f'./stas_base{id}'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    # cfg.optimizer.lr = 0.02 / 8
    # cfg.lr_config.warmup = None
    # cfg.log_config.interval = 10

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 1
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 1

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = [0]
    cfg.data.samples_per_gpu = test_batch_size
    cfg.runner.max_epochs=30
    cfg.device='cuda'

    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')



    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

        # Create work_dir
        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # train_detector(model, datasets, cfg, distributed=False, validate=True)

    return model, cfg

if __name__ == '__main__':
    
    test_batch_size = 32

    model, cfg = procedure(test_batch_size = test_batch_size)

    """testing"""
    model.cfg = cfg
    dir = "./STAS_dataset/Test_Images"
    files = os.listdir(dir)
    checkpoint = load_checkpoint(model, f"./stas_base_resize0/epoch_1.pth")
    load_state_dict(model, checkpoint)

    dictionary = {}
    images = []
    file_name_list = []
    image_count = 0

    

    for file in files:
        print(file)
        file_name = os.path.join(dir, file)
        img = mmcv.imread(file_name)
        images.append(img)
        file_name_list.append(file)
        if ((image_count % test_batch_size) == (test_batch_size - 1)) or file == files[len(files) - 1]:
            results = inference_detector(model, images)
            for i in range(len(results)):
                true_image_count = test_batch_size * (image_count // test_batch_size) + i
                print(true_image_count)
                detect_flag = 0
                count_stas = 0
                temp = list(results[i][0])
                bboxs = [[int(item) for item in L][:4] for L in temp]
                print(file_name_list[true_image_count])
                print(len(images))

                cv2.imwrite(f"result/{file_name_list[true_image_count]}", bbv.draw_multiple_rectangles(images[i], bboxs, (0,0,0)))
                for stas in results[i][0]:
                    xmin = int(stas[0])
                    ymin = int(stas[1])
                    xmax = int(stas[2])
                    ymax = int(stas[3])
                    confidence = float(stas[4])
                    found_stas = np.array([[xmin, ymin, xmax, ymax, round(confidence,5)]])
                    if confidence>=0.05:
                        if count_stas==0:
                            final_stas = found_stas
                        else:
                            final_stas = np.concatenate((final_stas, found_stas), axis=0)

                        count_stas += 1
                        detect_flag = 1
                    
                if detect_flag:
                    dictionary.update({f"{file_name_list[true_image_count]}": None})
                    # print(file_name_list[image_count - 1])
                    dictionary[file_name_list[true_image_count]] =  final_stas.tolist()

                for stas_ in range(count_stas):
                    dictionary[file_name_list[true_image_count]][stas_][0] = int(dictionary[file_name_list[true_image_count]][stas_][0])
                    dictionary[file_name_list[true_image_count]][stas_][1] = int(dictionary[file_name_list[true_image_count]][stas_][1])
                    dictionary[file_name_list[true_image_count]][stas_][2] = int(dictionary[file_name_list[true_image_count]][stas_][2])
                    dictionary[file_name_list[true_image_count]][stas_][3] = int(dictionary[file_name_list[true_image_count]][stas_][3])

            images = []
        
        image_count += 1
            
    json_object = json.dumps(dictionary)
    with open ("result_draw_bounding_box.json", 'w') as j:
        j.write(json_object)
