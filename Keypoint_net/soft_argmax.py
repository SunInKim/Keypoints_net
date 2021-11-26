import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_argmax(voxels):
	"""
	Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
	Return: 3D coordinates in shape (batch_size, channel, 3)
	"""
	assert voxels.dim()==4
	# alpha is here to make the largest element really big, so it
	# would become very close to 1 after softmax
	alpha = 1000.0 
	N,C,H,W = voxels.shape

	# soft_max = voxels.view(N,C,-1).type(torch.float64)
	soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2, dtype=torch.float64)
	soft_max = soft_max.view(voxels.shape)
	indices_kernel = torch.arange(start=0,end=H*W , dtype=torch.float64).unsqueeze(0)
	indices_kernel = indices_kernel.view((H,W)).cuda()
	conv = soft_max*indices_kernel
	indices = conv.sum(2).sum(2)
	y = indices%W
	x = (indices/W)%H
	# x = (((indices/D).floor())/W).floor()%H
	coords = torch.stack([x,y],dim=2)
	return coords


def soft_argmax2(heatmaps, joint_num):
    assert isinstance(heatmaps, torch.Tensor)

    # N,C,H,W = heatmaps.shape

    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim*cfg.output_shape[0]*cfg.output_shape[1]))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim, cfg.output_shape[0], cfg.output_shape[1]))

    accu_x = heatmaps.sum(dim=(2,3))
    accu_y = heatmaps.sum(dim=(2,4))
    accu_z = heatmaps.sum(dim=(3,4))

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(1,cfg.output_shape[1]+1).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(1,cfg.output_shape[0]+1).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(1,cfg.depth_dim+1).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True) -1
    accu_y = accu_y.sum(dim=2, keepdim=True) -1
    accu_z = accu_z.sum(dim=2, keepdim=True) -1

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out


def generate_2d_integral_preds_tensor(heatmaps, key_num, x_dim, y_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], key_num, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    accu_y = heatmaps.sum(dim=3)

    # accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.FloatTensor), devices=[accu_x.device.index])[0]
    # accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.FloatTensor), devices=[accu_y.device.index])[0]
    
    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.FloatTensor), devices=[accu_y.device.index])[0]


    # accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(z_dim).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    return accu_x, accu_y

def softmax_integral_tensor(heatmaps):
    # global soft max
    N,key_num,H,W = heatmaps.shape
    heatmaps = heatmaps.reshape((heatmaps.shape[0], key_num, -1))
    heatmaps = F.softmax(heatmaps, 2)
    
    # integrate heatmap into joint location    
    x, y = generate_2d_integral_preds_tensor(heatmaps, key_num, W, H)
    
    u = x / float(W) - 0.5
    v = y / float(H) - 0.5
    # z = z / float(hm_depth) - 0.5
    heatmaps = torch.cat((v, u), dim=2)
    heatmaps = heatmaps.type(torch.float64)
    # print(heatmaps.shape)
    # heatmaps = heatmaps.reshape((heatmaps.shape[0], key_num * 2))
    return heatmaps