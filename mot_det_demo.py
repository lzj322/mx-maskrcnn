import mxnet as mx
import argparse
import os
from rcnn.config import config, default, generate_config
from rcnn.tools.demo_maskrcnn import demo_maskrcnn

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    print(args.dataset)
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    parser.add_argument('--result_path', help='result path', type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=default.rcnn_prefix, type=str)
    parser.add_argument('--epoch', help='model to test with', default=default.rcnn_epoch, type=int)
    parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)
    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=0.65, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--has_rpn', help='generate proposals on the fly', action='store_true')
    parser.add_argument('--proposal', help='can be ss for selective search or rpn', default='rpn', type=str)
    parser.add_argument('--det_save_file', help='file path to save detection results', default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    ctx = mx.gpu(args.gpu)
    dataset_types=['train','test']
    dataset_names=[[5],[6]]#[[10,11,13],[1,3,7,8,12,14]]
    for idx, dataset_type in enumerate(dataset_types):
        for dataset_name_idx in dataset_names[idx]:
            print "============= {}: MOT17-{} ================".format(dataset_type, str(dataset_name_idx).zfill(2))
            image_set = 'MOT17_{}'.format(str(dataset_name_idx).zfill(2))
            result_path='data/MOT/results/{}'.format(image_set)
            det_save_file = os.path.join('data/MOT/MOT_det_results/{}/MOT17-{}'.format(dataset_type,str(dataset_name_idx).zfill(2)))
            if not os.path.exists(det_save_file):
                os.makedirs(det_save_file)
            if os.path.exists(det_save_file+'/maskrcnn_det.txt'):
                os.remove(det_save_file+'/maskrcnn_det.txt')
            demo_maskrcnn(args.network, args.dataset, image_set, args.root_path, args.dataset_path, result_path,
                    ctx, args.prefix, args.epoch,
                    args.vis, args.shuffle, args.has_rpn, args.proposal, args.thresh, det_save_file=det_save_file)

if __name__ == '__main__':
    main()
