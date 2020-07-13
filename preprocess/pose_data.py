import os
import sys
import glob
import cv2
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
sys.path.append('../lib')
from align import align_nocs_to_depth
from utils import load_depth


def create_img_list(data_dir):
    """ Create train/val/test data list for CAMERA and Real. """
    # CAMERA dataset
    for subset in ['train', 'val']:
        img_list = []
        img_dir = os.path.join(data_dir, 'CAMERA', subset)
        folder_list = [name for name in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, name))]
        for i in range(10*len(folder_list)):
            folder_id = int(i) // 10
            img_id = int(i) % 10
            img_path = os.path.join(subset, '{:05d}'.format(folder_id), '{:04d}'.format(img_id))
            img_list.append(img_path)
        with open(os.path.join(data_dir, 'CAMERA', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
    # Real dataset
    for subset in ['train', 'test']:
        img_list = []
        img_dir = os.path.join(data_dir, 'Real', subset)
        folder_list = [name for name in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name))]
        for folder in folder_list:
            img_paths = glob.glob(os.path.join(img_dir, folder, '*_color.png'))
            img_paths = sorted(img_paths)
            for img_full_path in img_paths:
                img_name = os.path.basename(img_full_path)
                img_ind = img_name.split('_')[0]
                img_path = os.path.join(subset, folder, img_ind)
                img_list.append(img_path)
        with open(os.path.join(data_dir, 'Real', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
    print('Write all data paths to file done!')


def process_data(img_path, depth):
    """ Load instance masks for the objects in the image. """
    mask_path = img_path + '_mask.png'
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    all_inst_ids = sorted(list(np.unique(mask)))
    assert all_inst_ids[-1] == 255
    del all_inst_ids[-1]    # remove background
    num_all_inst = len(all_inst_ids)
    h, w = mask.shape

    coord_path = img_path + '_coord.png'
    coord_map = cv2.imread(coord_path)[:, :, :3]
    coord_map = coord_map[:, :, (2, 1, 0)]
    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    class_ids = []
    instance_ids = []
    model_list = []
    masks = np.zeros([h, w, num_all_inst], dtype=np.uint8)
    coords = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)
    bboxes = np.zeros((num_all_inst, 4), dtype=np.int32)

    meta_path = img_path + '_meta.txt'
    with open(meta_path, 'r') as f:
        i = 0
        for line in f:
            line_info = line.strip().split(' ')
            inst_id = int(line_info[0])
            cls_id = int(line_info[1])
            # background objects and non-existing objects
            if cls_id == 0 or (inst_id not in all_inst_ids):
                continue
            if len(line_info) == 3:
                model_id = line_info[2]    # Real scanned objs
            else:
                model_id = line_info[3]    # CAMERA objs
            # remove one mug instance in CAMERA train due to improper model
            if model_id == 'b9be7cfe653740eb7633a2dd89cec754':
                continue
            # process foreground objects
            inst_mask = np.equal(mask, inst_id)
            # bounding box
            horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]
            vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]
            assert horizontal_indicies.shape[0], print(img_path)
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            # object occupies full image, rendering error, happens in CAMERA dataset
            if np.any(np.logical_or((x2-x1) > 600, (y2-y1) > 440)):
                return None, None, None, None, None, None
            # not enough valid depth observation
            final_mask = np.logical_and(inst_mask, depth > 0)
            if np.sum(final_mask) < 64:
                continue
            class_ids.append(cls_id)
            instance_ids.append(inst_id)
            model_list.append(model_id)
            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            bboxes[i] = np.array([y1, x1, y2, x2])
            i += 1
    # no valid foreground objects
    if i == 0:
        return None, None, None, None, None, None

    masks = masks[:, :, :i]
    coords = np.clip(coords[:, :, :i, :], 0, 1)
    bboxes = bboxes[:i, :]

    return masks, coords, class_ids, instance_ids, model_list, bboxes


def annotate_camera_train(data_dir):
    """ Generate gt labels for CAMERA train data. """
    camera_train = open(os.path.join(data_dir, 'CAMERA', 'train_list_all.txt')).read().splitlines()
    intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    valid_img_list = []
    for img_path in tqdm(camera_train):
        img_full_path = os.path.join(data_dir, 'CAMERA', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        depth = load_depth(img_full_path)
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        if instance_ids is None:
            continue
        # Umeyama alignment of GT NOCS map with depth image
        scales, rotations, translations, error_messages, _ = \
            align_nocs_to_depth(masks, coords, depth, intrinsics, instance_ids, img_path)
        if error_messages:
            continue
        # re-label for mug category
        for i in range(len(class_ids)):
            if class_ids[i] == 6:
                T0 = mug_meta[model_list[i]][0]
                s0 = mug_meta[model_list[i]][1]
                T = translations[i] - scales[i] * rotations[i] @ T0
                s = scales[i] / s0
                scales[i] = s
                translations[i] = T
        # write results
        gts = {}
        gts['class_ids'] = class_ids    # int list, 1 to 6
        gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
        gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        gts['rotations'] = rotations.astype(np.float32)    # np.array, R
        gts['translations'] = translations.astype(np.float32)  # np.array, T
        gts['instance_ids'] = instance_ids  # int list, start from 1
        gts['model_list'] = model_list  # str list, model id/name
        with open(img_full_path + '_label.pkl', 'wb') as f:
            cPickle.dump(gts, f)
        valid_img_list.append(img_path)
    # write valid img list to file
    with open(os.path.join(data_dir, 'CAMERA/train_list.txt'), 'w') as f:
        for img_path in valid_img_list:
            f.write("%s\n" % img_path)


def annotate_real_train(data_dir):
    """ Generate gt labels for Real train data through PnP. """
    real_train = open(os.path.join(data_dir, 'Real/train_list_all.txt')).read().splitlines()
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # scale factors for all instances
    scale_factors = {}
    path_to_size = glob.glob(os.path.join(data_dir, 'obj_models/real_train', '*_norm.txt'))
    for inst_path in sorted(path_to_size):
        instance = os.path.basename(inst_path).split('.')[0]
        bbox_dims = np.loadtxt(inst_path)
        scale_factors[instance] = np.linalg.norm(bbox_dims)
    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    valid_img_list = []
    for img_path in tqdm(real_train):
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        depth = load_depth(img_full_path)
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        if instance_ids is None:
            continue
        # compute pose
        num_insts = len(class_ids)
        scales = np.zeros(num_insts)
        rotations = np.zeros((num_insts, 3, 3))
        translations = np.zeros((num_insts, 3))
        for i in range(num_insts):
            s = scale_factors[model_list[i]]
            mask = masks[:, :, i]
            idxs = np.where(mask)
            coord = coords[:, :, i, :]
            coord_pts = s * (coord[idxs[0], idxs[1], :] - 0.5)
            coord_pts = coord_pts[:, :, None]
            img_pts = np.array([idxs[1], idxs[0]]).transpose()
            img_pts = img_pts[:, :, None].astype(float)
            distCoeffs = np.zeros((4, 1))    # no distoration
            retval, rvec, tvec = cv2.solvePnP(coord_pts, img_pts, intrinsics, distCoeffs)
            assert retval
            R, _ = cv2.Rodrigues(rvec)
            T = np.squeeze(tvec)
            # re-label for mug category
            if class_ids[i] == 6:
                T0 = mug_meta[model_list[i]][0]
                s0 = mug_meta[model_list[i]][1]
                T = T - s * R @ T0
                s = s / s0
            scales[i] = s
            rotations[i] = R
            translations[i] = T
        # write results
        gts = {}
        gts['class_ids'] = class_ids    # int list, 1 to 6
        gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
        gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        gts['rotations'] = rotations.astype(np.float32)    # np.array, R
        gts['translations'] = translations.astype(np.float32)  # np.array, T
        gts['instance_ids'] = instance_ids  # int list, start from 1
        gts['model_list'] = model_list  # str list, model id/name
        with open(img_full_path + '_label.pkl', 'wb') as f:
            cPickle.dump(gts, f)
        valid_img_list.append(img_path)
    # write valid img list to file
    with open(os.path.join(data_dir, 'Real/train_list.txt'), 'w') as f:
        for img_path in valid_img_list:
            f.write("%s\n" % img_path)


def annotate_test_data(data_dir):
    """ Generate gt labels for test data.
        Properly copy handle_visibility provided by NOCS gts.
    """
    # Statistics:
    # test_set    missing file     bad rendering    no (occluded) fg    occlusion (< 64 pts)
    #   val        3792 imgs        132 imgs         1856 (23) imgs      50 insts
    #   test       0 img            0 img            0 img               2 insts

    camera_val = open(os.path.join(data_dir, 'CAMERA', 'val_list_all.txt')).read().splitlines()
    real_test = open(os.path.join(data_dir, 'Real', 'test_list_all.txt')).read().splitlines()
    camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # compute model size
    model_file_path = ['obj_models/camera_val.pkl', 'obj_models/real_test.pkl']
    models = {}
    for path in model_file_path:
        with open(os.path.join(data_dir, path), 'rb') as f:
            models.update(cPickle.load(f))
    model_sizes = {}
    for key in models.keys():
        model_sizes[key] = 2 * np.amax(np.abs(models[key]), axis=0)
    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    subset_meta = [('CAMERA', camera_val, camera_intrinsics, 'val'), ('Real', real_test, real_intrinsics, 'test')]
    for source, img_list, intrinsics, subset in subset_meta:
        valid_img_list = []
        for img_path in tqdm(img_list):
            img_full_path = os.path.join(data_dir, source, img_path)
            all_exist = os.path.exists(img_full_path + '_color.png') and \
                        os.path.exists(img_full_path + '_coord.png') and \
                        os.path.exists(img_full_path + '_depth.png') and \
                        os.path.exists(img_full_path + '_mask.png') and \
                        os.path.exists(img_full_path + '_meta.txt')
            if not all_exist:
                continue
            depth = load_depth(img_full_path)
            masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
            if instance_ids is None:
                continue
            num_insts = len(instance_ids)
            # match each instance with NOCS ground truth to properly assign gt_handle_visibility
            nocs_dir = os.path.join(os.path.dirname(data_dir), 'results/nocs_results')
            if source == 'CAMERA':
                nocs_path = os.path.join(nocs_dir, 'val', 'results_val_{}_{}.pkl'.format(
                    img_path.split('/')[-2], img_path.split('/')[-1]))
            else:
                nocs_path = os.path.join(nocs_dir, 'real_test', 'results_test_{}_{}.pkl'.format(
                    img_path.split('/')[-2], img_path.split('/')[-1]))
            with open(nocs_path, 'rb') as f:
                nocs = cPickle.load(f)
            gt_class_ids = nocs['gt_class_ids']
            gt_bboxes = nocs['gt_bboxes']
            gt_sRT = nocs['gt_RTs']
            gt_handle_visibility = nocs['gt_handle_visibility']
            map_to_nocs = []
            for i in range(num_insts):
                gt_match = -1
                for j in range(len(gt_class_ids)):
                    if gt_class_ids[j] != class_ids[i]:
                        continue
                    if np.sum(np.abs(bboxes[i] - gt_bboxes[j])) > 5:
                        continue
                    # match found
                    gt_match = j
                    break
                # check match validity
                assert gt_match > -1, print(img_path, instance_ids[i], 'no match for instance')
                assert gt_match not in map_to_nocs, print(img_path, instance_ids[i], 'duplicate match')
                map_to_nocs.append(gt_match)
            # copy from ground truth, re-label for mug category
            handle_visibility = gt_handle_visibility[map_to_nocs]
            sizes = np.zeros((num_insts, 3))
            poses = np.zeros((num_insts, 4, 4))
            scales = np.zeros(num_insts)
            rotations = np.zeros((num_insts, 3, 3))
            translations = np.zeros((num_insts, 3))
            for i in range(num_insts):
                gt_idx = map_to_nocs[i]
                sizes[i] = model_sizes[model_list[i]]
                sRT = gt_sRT[gt_idx]
                s = np.cbrt(np.linalg.det(sRT[:3, :3]))
                R = sRT[:3, :3] / s
                T = sRT[:3, 3]
                # re-label mug category
                if class_ids[i] == 6:
                    T0 = mug_meta[model_list[i]][0]
                    s0 = mug_meta[model_list[i]][1]
                    T = T - s * R @ T0
                    s = s / s0
                # used for test during training
                scales[i] = s
                rotations[i] = R
                translations[i] = T
                # used for evaluation
                sRT = np.identity(4, dtype=np.float32)
                sRT[:3, :3] = s * R
                sRT[:3, 3] = T
                poses[i] = sRT
            # write results
            gts = {}
            gts['class_ids'] = np.array(class_ids)    # int list, 1 to 6
            gts['bboxes'] = bboxes    # np.array, [[y1, x1, y2, x2], ...]
            gts['instance_ids'] = instance_ids    # int list, start from 1
            gts['model_list'] = model_list    # str list, model id/name
            gts['size'] = sizes   # 3D size of NOCS model
            gts['scales'] = scales.astype(np.float32)    # np.array, scale factor from NOCS model to depth observation
            gts['rotations'] = rotations.astype(np.float32)    # np.array, R
            gts['translations'] = translations.astype(np.float32)    # np.array, T
            gts['poses'] = poses.astype(np.float32)    # np.array
            gts['handle_visibility'] = handle_visibility    # handle visibility of mug
            with open(img_full_path + '_label.pkl', 'wb') as f:
                cPickle.dump(gts, f)
            valid_img_list.append(img_path)
        # write valid img list to file
        with open(os.path.join(data_dir, source, subset+'_list.txt'), 'w') as f:
            for img_path in valid_img_list:
                f.write("%s\n" % img_path)


if __name__ == '__main__':
    data_dir = '/home/tianmeng/Documents/pose_ws/object-deformnet/data'
    # create list for all data
    create_img_list(data_dir)
    # annotate dataset and re-write valid data to list
    annotate_camera_train(data_dir)
    annotate_real_train(data_dir)
    annotate_test_data(data_dir)
