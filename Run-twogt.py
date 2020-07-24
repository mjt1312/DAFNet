
import os
from os.path import join, isdir
from tracker import *
import numpy as np
import json
import argparse
from scipy import io
import pickle

import math
import pdb

def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)


    '''if set_type == 'RGB-T234' or 'VOT-RGBT':
        ############################################  have to refine #############################################

        img_list_v = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        img_list_t = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])
        gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')'''
    if set_type == 'RGB-T234':
        ############################################  have to refine #############################################

        img_list_v = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.bmp' or 'png'])
        img_list_t = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.bmp' or 'png'])
        gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        #gt_t = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')
        #gt = np.loadtxt(seq_path + '/init.txt')		
        #gt_v=np.concatenate((gt_v[:,0][:,None],gt_v[:,1][:,None],gt_v[:,2][:,None]-gt_v[:,0][:,None],gt_v[:,3][:,None]-gt_v[:,1][:,None]),axis=1)
        #gt_t=np.concatenate((gt_t[:,0][:,None],gt_t[:,1][:,None],gt_t[:,2][:,None]-gt_t[:,0][:,None],gt_t[:,3][:,None]-gt_t[:,1][:,None]),axis=1)		
        #gt=0.5*(gt_v+gt_t)
        #gt_i = np.loadtxt(seq_path + '/groundTruth_i.txt')		
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
    parser.add_argument("-set_type", default = 'RGB-T234' )

    parser.add_argument("-result_path", default = './result.npy')
    parser.add_argument("-visual_log",default=False, action= 'store_true')
    parser.add_argument("-visualize",default=False, action='store_true')
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
    dataset_path = '/home/zhuyabin/gaoyuan/'

    seq_home = dataset_path + opts['set_type']
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
    #seq_list = ['bus6']#twoelecbike1
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

        iou_result, result_bb, fps, result_nobb = run_mdnet(img_list_v, img_list_t,gt[0], gt, seq = seq, display=False) #opts['visualize']

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
        #save results
        loc_8 = []
        #if exists then remove old txt
        if os.path.exists(os.path.join('./RT-v-gt234/' + 'RT-v-gt234' + '_' + seq + '.txt')):
            os.remove(os.path.join('./RT-v-gt234/' + 'RT-v-gt234' + '_' + seq + '.txt'))
        for loc in res['res']:
            loc_8.append(loc[0])
            loc_8.append(loc[1])
            loc_8.append(loc[0]+loc[2])
            loc_8.append(loc[1])
            loc_8.append(loc[0]+loc[2])
            loc_8.append(loc[1] + loc[3])
            loc_8.append(loc[0])
            loc_8.append(loc[1] + loc[3])
            with open(os.path.join('./RT-v-gt234/' + 'RT-v-gt234'+'_' + seq + '.txt'), 'a') as f:
                count = 0
                for k in loc_8:
                    count +=1
                    f.write(str(k))
                    if count<8:
                        f.write(' ')
                f.write('\n')
            loc_8 = []


