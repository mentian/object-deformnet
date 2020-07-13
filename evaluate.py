import os
import time
import argparse
import cv2
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.network import DeformNet
from lib.align import estimateSimilarityTransform
from lib.utils import load_depth, get_bbox, compute_mAP, plot_mAP


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='val', help='val, real_test')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
parser.add_argument('--model', type=str, default='results/camera/model_50.pth', help='resume from saved model')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
opt = parser.parse_args()

mean_shapes = np.load('assets/mean_points_emb.npy')

assert opt.data in ['val', 'real_test']
if opt.data == 'val':
    result_dir = 'results/eval_camera'
    file_path = 'CAMERA/val_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
else:
    result_dir = 'results/eval_real'
    file_path = 'Real/test_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0
norm_color = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def detect():
    # resume model
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    estimator = DeformNet(opt.n_cat, opt.nv_prior)
    estimator.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()
    # get test data list
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
                for line in open(os.path.join(opt.data_dir, file_path))]
    # frame by frame test
    t_inference = 0.0
    t_umeyama = 0.0
    inst_count = 0
    img_count = 0
    t_start = time.time()
    for path in tqdm(img_list):
        img_path = os.path.join(opt.data_dir, path)
        raw_rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        raw_rgb = raw_rgb[:, :, ::-1]
        raw_depth = load_depth(img_path)
        # load mask-rcnn detection results
        img_path_parsing = img_path.split('/')
        mrcnn_path = os.path.join('results/mrcnn_results', opt.data, 'results_{}_{}_{}.pkl'.format(
            opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)
        num_insts = len(mrcnn_result['class_ids'])
        f_sRT = np.zeros((num_insts, 4, 4), dtype=float)
        f_size = np.zeros((num_insts, 3), dtype=float)
        # prepare frame data
        f_points, f_rgb, f_choose, f_catId, f_prior = [], [], [], [], []
        valid_inst = []
        for i in range(num_insts):
            cat_id = mrcnn_result['class_ids'][i] - 1
            prior = mean_shapes[cat_id]
            rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][i])
            mask = np.logical_and(mrcnn_result['masks'][:, :, i], raw_depth > 0)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            # no depth observation for background in CAMERA dataset
            # beacuase of how we compute the bbox in function get_bbox
            # there might be a chance that no foreground points after cropping the mask
            # cuased by false positive of mask_rcnn, most of the regions are background
            if len(choose) < 32:
                f_sRT[i] = np.identity(4, dtype=float)
                f_size[i] = 2 * np.amax(np.abs(prior), axis=0)
                continue
            else:
                valid_inst.append(i)
            # process objects with valid depth observation
            if len(choose) > opt.n_pts:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:opt.n_pts] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, opt.n_pts-len(choose)), 'wrap')
            depth_masked = raw_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            pt2 = depth_masked / norm_scale
            pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
            points = np.concatenate((pt0, pt1, pt2), axis=1)
            rgb = raw_rgb[rmin:rmax, cmin:cmax, :]
            rgb = cv2.resize(rgb, (opt.img_size, opt.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = norm_color(rgb)
            crop_w = rmax - rmin
            ratio = opt.img_size / crop_w
            col_idx = choose % crop_w
            row_idx = choose // crop_w
            choose = (np.floor(row_idx * ratio) * opt.img_size + np.floor(col_idx * ratio)).astype(np.int64)
            # concatenate instances
            f_points.append(points)
            f_rgb.append(rgb)
            f_choose.append(choose)
            f_catId.append(cat_id)
            f_prior.append(prior)
        if len(valid_inst):
            f_points = torch.cuda.FloatTensor(f_points)
            f_rgb = torch.stack(f_rgb, dim=0).cuda()
            f_choose = torch.cuda.LongTensor(f_choose)
            f_catId = torch.cuda.LongTensor(f_catId)
            f_prior = torch.cuda.FloatTensor(f_prior)
            # inference
            torch.cuda.synchronize()
            t_now = time.time()
            assign_mat, deltas = estimator(f_points, f_rgb, f_choose, f_catId, f_prior)
            # assign_mat, deltas = estimator(f_rgb, f_choose, f_catId, f_prior)
            inst_shape = f_prior + deltas
            assign_mat = F.softmax(assign_mat, dim=2)
            f_coords = torch.bmm(assign_mat, inst_shape)  # bs x n_pts x 3
            torch.cuda.synchronize()
            t_inference += (time.time() - t_now)
            f_coords = f_coords.detach().cpu().numpy()
            f_points = f_points.cpu().numpy()
            f_choose = f_choose.cpu().numpy()
            f_insts = inst_shape.detach().cpu().numpy()
            t_now = time.time()
            for i in range(len(valid_inst)):
                inst_idx = valid_inst[i]
                choose = f_choose[i]
                _, choose = np.unique(choose, return_index=True)
                nocs_coords = f_coords[i, choose, :]
                f_size[inst_idx] = 2 * np.amax(np.abs(f_insts[i]), axis=0)
                points = f_points[i, choose, :]
                _, _, _, pred_sRT = estimateSimilarityTransform(nocs_coords, points)
                if pred_sRT is None:
                    pred_sRT = np.identity(4, dtype=float)
                f_sRT[inst_idx] = pred_sRT
            t_umeyama += (time.time() - t_now)
            img_count += 1
            inst_count += len(valid_inst)

        # save results
        result = {}
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        result['gt_class_ids'] = gts['class_ids']
        result['gt_bboxes'] = gts['bboxes']
        result['gt_RTs'] = gts['poses']
        result['gt_scales'] = gts['size']
        result['gt_handle_visibility'] = gts['handle_visibility']

        result['pred_class_ids'] = mrcnn_result['class_ids']
        result['pred_bboxes'] = mrcnn_result['rois']
        result['pred_scores'] = mrcnn_result['scores']
        result['pred_RTs'] = f_sRT
        result['pred_scales'] = f_size

        image_short_path = '_'.join(img_path_parsing[-3:])
        save_path = os.path.join(result_dir, 'results_{}.pkl'.format(image_short_path))
        with open(save_path, 'wb') as f:
            cPickle.dump(result, f)
    # write statistics
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'w')
    messages = []
    messages.append("Total images: {}".format(len(img_list)))
    messages.append("Valid images: {},  Total instances: {},  Average: {:.2f}/image".format(
        img_count, inst_count, inst_count/img_count))
    messages.append("Inference time: {:06f}  Average: {:06f}/image".format(t_inference, t_inference/img_count))
    messages.append("Umeyama time: {:06f}  Average: {:06f}/image".format(t_umeyama, t_umeyama/img_count))
    messages.append("Total time: {:06f}".format(time.time() - t_start))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()


def evaluate():
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    result_pkl_list = glob.glob(os.path.join(result_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False
    # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation
    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, result_dir, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)
    # metric
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    messages = []
    messages.append('mAP:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('Acc:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_acc[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_acc[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_acc[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_05_idx] * 100))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()
    # load NOCS results
    pkl_path = os.path.join('results/nocs_results', opt.data, 'mAP_Acc.pkl')
    with open(pkl_path, 'rb') as f:
        nocs_results = cPickle.load(f)
    nocs_iou_aps = nocs_results['iou_aps'][-1, :]
    nocs_pose_aps = nocs_results['pose_aps'][-1, :, :]
    iou_aps = np.concatenate((iou_aps, nocs_iou_aps[None, :]), axis=0)
    pose_aps = np.concatenate((pose_aps, nocs_pose_aps[None, :, :]), axis=0)
    # plot
    plot_mAP(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list)


if __name__ == '__main__':
    print('Detecting ...')
    detect()
    print('Evaluating ...')
    evaluate()
