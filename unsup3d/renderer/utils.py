import numpy as np
import torch
import torch.nn as nn
import pytorch3d
import pytorch3d.loss
import pytorch3d.renderer
import pytorch3d.structures
import pytorch3d.io
import pytorch3d.transforms

def get_printer(msg):
    """This function returns a printer function, that prints information about a  tensor's
    gradient. Used by register_hook in the backward pass.
    """
    def printer(tensor):
        if tensor.nelement() == 1:
            print(f"{msg} {tensor}")
        else:
            print(f"{msg} shape: {tensor.shape}"
                  f" max: {tensor.max()} min: {tensor.min()}"
                  f" mean: {tensor.mean()}")
    return printer


def register_hook(tensor, msg):
    """Utility function to call retain_grad and Pytorch's register_hook
    in a single line
    """
    tensor.retain_grad()
    tensor.register_hook(get_printer(msg))

def create_meshes_from_grid_3d(grid_3d, device):
    ## Vertices
    vertices = grid_3d 
    b, h, w, _ = vertices.shape
    vertices_center = torch.nn.functional.avg_pool2d(vertices.permute(0,3,1,2), 2, stride=1).permute(0,2,3,1)
    vertices = torch.cat([vertices.view(b,h*w,3), vertices_center.view(b,(h-1)*(w-1),3)], 1)

    tmp_vertices = torch.zeros_like(vertices).detach()


    if vertices.requires_grad:
        register_hook(vertices, "vertices")
    # register_hook(vertices_center, "vertices_center")
    # register_hook(vertices, "vertices")

    ## Faces
    idx_map = torch.arange(h*w).reshape(h,w)
    idx_map_center = torch.arange((h-1)*(w-1)).reshape(h-1,w-1)
    faces1 = torch.stack([idx_map[:h-1,:w-1], idx_map[1:,:w-1], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces2 = torch.stack([idx_map[1:,:w-1], idx_map[1:,1:], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces3 = torch.stack([idx_map[1:,1:], idx_map[:h-1,1:], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces4 = torch.stack([idx_map[:h-1,1:], idx_map[:h-1,:w-1], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces = torch.cat([faces1, faces2, faces3, faces4], 1)

    mesh_debug = pytorch3d.structures.Meshes(verts=tmp_vertices, faces=faces).to(device)
    # mesh_debug = mesh_debug.offset_verts(vertices_test)
    mesh_debug = mesh_debug.update_padded(vertices)

    # meshes = pytorch3d.structures.Meshes(verts=vertices_test.to(device), faces=faces.to(device))
    return mesh_debug

def get_grid(b, H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b,1,1,1).flip(3).float() # flip h,w to x,y
    return grid 

def depth_to_3d_grid(depth, inv_K):
    b, h, w = depth.shape
    grid_2d = get_grid(b, h, w, normalize=False).to(depth.device)  # Nxhxwx2
    depth = depth.unsqueeze(-1)
    grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
    grid_3d = grid_3d.matmul(inv_K.transpose(2,1)) * depth
    return grid_3d