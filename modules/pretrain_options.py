from collections import OrderedDict

pretrain_opts = OrderedDict()
pretrain_opts['use_gpu'] = True

pretrain_opts['init_model_path'] = './models/imagenet-vgg-m.mat'
#pretrain_opts['model_path'] = './models/rt_mdnet.pth'
pretrain_opts['model_path'] = './models/layers.pth'
pretrain_opts['model_path2'] = './models/layers2.pth'

pretrain_opts['model_path4'] = './models/conv4.pth'
pretrain_opts['model_path5'] = './models/conv5.pth'

#sk
pretrain_opts['model_path_sk_fc1'] = './models/sk_fc1.pth'
pretrain_opts['model_path_sk_fcs1'] = './models/sk_fcs1.pth'
pretrain_opts['model_path_sk_fc2'] = './models/sk_fc2.pth'
pretrain_opts['model_path_sk_fcs2'] = './models/sk_fcs2.pth'
pretrain_opts['model_path_sk_fc3'] = './models/sk_fc3.pth'
pretrain_opts['model_path_sk_fcs3'] = './models/sk_fcs3.pth'

pretrain_opts['batch_frames'] = 8
pretrain_opts['batch_pos'] = 64
pretrain_opts['batch_neg'] = 196

pretrain_opts['overlap_pos'] = [0.7, 1]
pretrain_opts['overlap_neg'] = [0, 0.5]

pretrain_opts['img_size'] = 107


pretrain_opts['lr'] = 0.0001
pretrain_opts['w_decay'] = 0.0005 #ori = 0.0005
pretrain_opts['momentum'] = 0.9
pretrain_opts['grad_clip'] = 10
pretrain_opts['ft_layers'] = ['conv','fc','sk_fc']
pretrain_opts['lr_mult'] = {'fc':10}
pretrain_opts['n_cycles'] = 1000


##################################### from RCNN #############################################
pretrain_opts['padding'] = 1.2
pretrain_opts['padding_ratio']=5.
pretrain_opts['padded_img_size'] = pretrain_opts['img_size']*int(pretrain_opts['padding_ratio'])
pretrain_opts['frame_interval'] = 2
