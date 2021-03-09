import numpy as np
import math
import torch
import torch.nn as nn
import pytorch3d
import pytorch3d.loss
import pytorch3d.renderer
import pytorch3d.structures
import pytorch3d.io
import pytorch3d.transforms
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    SoftPhongShader,
    MeshRasterizer,
    TexturesUV,
    DirectionalLights
)
from . import utils

def get_printer(msg):
    """This function returns a printer function, that prints information about a  tensor's
    gradient. Used by register_hook in the backward pass.
    """
    def printer(tensor):
        print("Was geht?")
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


class Renderer(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 64)
        self.fov = cfgs.get('fov', 10)
        self.blur_radius = cfgs.get('blur_radius', np.log(1. / 1e-4 - 1.)*1e-4)
        self.cameras = pytorch3d.renderer.FoVPerspectiveCameras(znear=0.9, zfar=1.1,fov=self.fov, device=self.device)
        self.image_renderer = self._create_image_renderer()
        init_verts, init_faces, init_aux = pytorch3d.io.load_obj(cfgs['init_shape_obj_path'], device=self.device)
        self.tex_faces_uv = init_faces.textures_idx.unsqueeze(0)
        self.tex_verts_uv = init_aux.verts_uvs.unsqueeze(0)
        fx = (self.image_size-1)/2/(math.tan(self.fov/2 *math.pi/180))
        fy = (self.image_size-1)/2/(math.tan(self.fov/2 *math.pi/180))
        cx = (self.image_size-1)/2
        cy = (self.image_size-1)/2
        K = [[fx, 0., cx],
             [0., fy, cy],
             [0., 0., 1.]]
        K = torch.FloatTensor(K).to(self.device)
        self.inv_K = torch.inverse(K).unsqueeze(0)

    def _create_image_renderer(self):
        raster_settings = self._get_rasterization_settings()
        renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
                                shader=SoftPhongShader(device=self.device, cameras=self.cameras))
        return renderer

    def _get_rasterization_settings(self):
        raster_settings = RasterizationSettings(image_size=self.image_size, blur_radius=self.blur_radius, faces_per_pixel=32, perspective_correct=False)
        return raster_settings

    def _get_textures(self, tex_im):
        # tex_im is a tensor that contains the non-flipped and flipped albedo_map
        tex_im = tex_im.permute(0,2,3,1)/2.+0.5
        b, h, w, c = tex_im.shape
        assert w == self.image_size and h == self.image_size, "Texture image has the wrong resolution."
        textures = TexturesUV(maps=tex_im,  # texture maps are BxHxWx3
                                    faces_uvs=self.tex_faces_uv.repeat(b, 1, 1),
                                    verts_uvs=self.tex_verts_uv.repeat(b, 1, 1))
        return textures

    
    def _get_lights(self, lighting):
        ambient = lighting["ambient"]/2.+0.5
        diffuse = lighting["diffuse"]/2.+0.5
        direction = lighting["direction"]
        ambient_color = ambient.repeat(1,3)
        diffuse_color = diffuse.repeat(1,3)
        b, _  = ambient.shape
        specular_color=torch.zeros((b,3))
        lights = DirectionalLights(ambient_color=ambient_color, diffuse_color=diffuse_color, specular_color=specular_color, direction=direction)
        return lights.to(self.device)
    

    def _get_transformed_meshes(self, meshes, view):
        # rotate mesh with pytorch functions
        # rotate non flipped mesh in one direction and flipped version in other direction

        R = pytorch3d.transforms.euler_angles_to_matrix(view[:,:3], convention="XYZ")
        T = view[:,3:]

        object_center = torch.tensor([[0.0,0.0,1.0]])
        shift_center = pytorch3d.transforms.Translate(-object_center, device=self.device)
        rotate = pytorch3d.transforms.Rotate(R, device=self.device)
        shift_back = pytorch3d.transforms.Translate(object_center, device=self.device)
        translate = pytorch3d.transforms.Translate(T, device=self.device)
        tsf = shift_center.compose(rotate, shift_back, translate)

        meshes_verts = meshes.verts_padded()
        new_mesh_verts = tsf.transform_points(meshes_verts)
        transformed_meshes = meshes.update_padded(new_mesh_verts) 
        
        return transformed_meshes.to(self.device)

    def create_meshes_from_depth_map(self,depth_map):
        grid_3d = utils.depth_to_3d_grid(depth_map, self.inv_K)
        meshes = utils.create_meshes_from_grid_3d(grid_3d, self.device)
        return meshes


    def forward(self, meshes, albedo_maps, view, lighting):
        textures = self._get_textures(albedo_maps)
        lights = self._get_lights(lighting)
        transformed_meshes = self._get_transformed_meshes(meshes, view)

        # replace texture at mesh
        transformed_meshes.textures = textures

        images = self.image_renderer(meshes_world=transformed_meshes,lights=lights)

        return images