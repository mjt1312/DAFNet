
import os
from os.path import join, isdir
from tracker import *
import numpy as np
import json
import argparse
from scipy import io
import pickle

import math


def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)


    if set_type == 'RGB-T234' or 'VOT-RGBT':
        ############################################  have to refine #############################################

        img_list_v = sorted([seq_path + '/color/' + p for p in os.listdir(seq_path + '/color') if os.path.splitext(p)[1] == '.jpg'])
        img_list_t = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir') if os.path.splitext(p)[1] == '.jpg'])
        gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')

        ##polygon to rect
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    return img_list_v,img_list_t, gt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'VOT-RGBT' )

    parser.add_argument("-result_path", default = './result.npy')
    parser.add_argument("-visual_log",default=False, action= 'store_true')
    parser.add_argument("-visualize",default=True, action='store_true')
    parser.add_argument("-adaptive_align",default=True, action='store_false')
    parser.add_argument("-padding",default=1.2, type = float)
    parser.add_argument("-jitter",default=True, action='store_false')

    args = parser.parse_args()

    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ## option setting

    opts['result_path']=args.result_path
    opts['visual_log']=args.visual_log
    opts['set_type']=args.set_type
    opts['visualize'] = args.visualize
    opts['adaptive_align'] = args.adaptive_align
    opts['padding'] = args.padding
    opts['jitter'] = args.jitter
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    print opts


    ## path initialization
    dataset_path = '/DATA/gaoyuan/trackingdata/'


    seq_home = dataset_path + opts['set_type']
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
    #seq_list = ['elecbike']
    iou_list=[]
    fps_list=dict()
    bb_result = dict()
    result = dict()

    iou_list_nobb=[]
    bb_result_nobb = dict()
    for num,seq in enumerate(seq_list):
        np.random.seed(123)
        torch.manual_seed(456)
        torch.cuda.manual_seed(789)	
        if num<-1:
            continue
        seq_path = seq_home + '/' + seq
        img_list_v,img_list_t,gt=genConfig(seq_path,opts['set_type'])

        iou_result, result_bb, fps, result_nobb = run_mdnet(img_list_v, img_list_t,gt[0], gt, seq = seq, display=opts['visualize'])

        enable_frameNum = 0.
        for iidx in range(len(iou_result)):
            if (math.isnan(iou_result[iidx])==False): 
                enable_frameNum += 1.
            else:
                ## gt is not alowed
                iou_result[iidx] = 0.

        iou_list.append(iou_result.sum()/enable_frameNum)
        bb_result[seq] = result_bb
        fps_list[seq]=fps

        bb_result_nobb[seq] = result_nobb
        print '{} {} : {} , total mIoU:{}, fps:{}'.format(num,seq,iou_result.mean(), sum(iou_list)/len(iou_list),sum(fps_list.values())/len(fps_list))
		
        res = {}
        res['res'] = result_bb.round().tolist()
        res['type'] = 'rect'
        res['fps'] = fps
        #json.dump(res, open(result_path, 'w'), indent=2)
        io.savemat('./RT-MDNet+A/RT-MDNet+A-VOT-RGBT' + '_'+ seq + '.mat',res)
		
    result['bb_result']=bb_result
    result['fps']=fps_list
    result['bb_result_nobb']=bb_result_nobb
    np.save(opts['result_path'],result)

