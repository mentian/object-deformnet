import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from lib.auto_encoder import PointCloudAE
from data.shape_dataset import ShapeDataset
from tools.tsne import tsne


def visualize_shape(name, shape_list, result_dir):
    """ Visualization and save image.

    Args:
        name: window name
        shape: list of geoemtries

    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=512, height=512, left=50, top=25)
    for shape in shape_list:
        vis.add_geometry(shape)
    ctr = vis.get_view_control()
    ctr.rotate(-300.0, 150.0)
    if name == 'camera':
        ctr.translate(20.0, -20.0)     # (horizontal right +, vertical down +)
    if name == 'laptop':
        ctr.translate(25.0, -60.0)
    vis.run()
    vis.capture_screen_image(os.path.join(result_dir, name+'.png'), False)
    vis.destroy_window()


parser = argparse.ArgumentParser()
parser.add_argument('--h5_file', type=str, default='data/obj_models/ShapeNetCore_2048.h5', help='h5py file')
parser.add_argument('--model', type=str, default='results/ae_points/model_50.pth',  help='resume model')
parser.add_argument('--result_dir', type=str, default='results/ae_points', help='directory to save mean shapes')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
opt = parser.parse_args()

opt.emb_dim = 512
opt.n_cat = 6
opt.n_pts = 1024

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

estimator = PointCloudAE(opt.emb_dim, opt.n_pts)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()
train_dataset = ShapeDataset(opt.h5_file, mode='train', augment=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

obj_models = []
embedding = []
catId = []  # zero-indexed
for i, data in enumerate(train_dataloader):
    batch_xyz, batch_label = data
    batch_xyz = batch_xyz[:, :, :3].cuda()
    batch_label = batch_label.cuda()
    emb, pred_points = estimator(batch_xyz)
    emb = emb.cpu().detach().numpy()
    inst_shape = batch_xyz.cpu().numpy()
    label = batch_label.cpu().numpy()
    embedding.append(emb)
    obj_models.append(inst_shape)
    catId.append(label)

embedding = np.squeeze(np.array(embedding).astype(np.float64), axis=1)
catId = np.squeeze((np.array(catId)), axis=1)
obj_models = np.squeeze(np.array(obj_models), axis=1)

# enbedding visualization
Y = tsne(embedding, 2, 50, 30.0)
y_bottle = Y[np.where(catId == 0)[0], :]
s_bottle = plt.scatter(y_bottle[:, 0], y_bottle[:, 1], s=20, marker='o', c='tab:orange')
y_bowl = Y[np.where(catId == 1)[0], :]
s_bowl = plt.scatter(y_bowl[:, 0], y_bowl[:, 1], s=20, marker='^', c='tab:blue')
y_camera = Y[np.where(catId == 2)[0], :]
s_camera = plt.scatter(y_camera[:, 0], y_camera[:, 1], s=20, marker='s', c='tab:olive')
y_can = Y[np.where(catId == 3)[0], :]
s_can = plt.scatter(y_can[:, 0], y_can[:, 1], s=20, marker='d', c='tab:gray')
y_laptop = Y[np.where(catId == 4)[0], :]
s_laptop = plt.scatter(y_laptop[:, 0], y_laptop[:, 1], s=20, marker='P', c='tab:purple')
y_mug = Y[np.where(catId == 5)[0], :]
s_mug = plt.scatter(y_mug[:, 0], y_mug[:, 1], s=20, marker='v', c='tab:brown')
plt.legend((s_bottle, s_bowl, s_camera, s_can, s_laptop, s_mug),
           ('bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'),
           loc='best', ncol=1, fontsize=8, frameon=False)
plt.xticks([])
plt.yticks([])
plt.savefig(os.path.join(opt.result_dir, 'visual_embedding.png'), bbox_inches='tight')

#  mean embedding and mean shape
mean_emb = np.empty((opt.n_cat, opt.emb_dim), dtype=np.float)
catId_to_name = {0: 'bottle', 1: 'bowl', 2: 'camera', 3: 'can', 4: 'laptop', 5: 'mug'}
mean_points = np.empty((opt.n_cat, opt.n_pts, 3), dtype=np.float)
for i in range(opt.n_cat):
    mean = np.mean(embedding[np.where(catId==i)[0], :], axis=0, keepdims=False)
    mean_emb[i] = mean
    assigned_emb = torch.cuda.FloatTensor(mean[None, :])
    _, mean_shape = estimator(None, assigned_emb)
    mean_shape = mean_shape.cpu().detach().numpy()[0]
    mean_points[i] = mean_shape
    # save point cloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mean_shape)
    visualize_shape(catId_to_name[i], [pcd], opt.result_dir)
# save results
np.save(os.path.join(opt.result_dir, 'mean_embedding'), mean_emb)
np.save(os.path.join(opt.result_dir, 'mean_points_emb'), mean_points)
