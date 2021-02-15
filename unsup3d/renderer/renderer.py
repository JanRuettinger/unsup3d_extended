import numpy as np
import torch
import torch.nn as nn
import pytorch3d
import pytorch3d.loss
import pytorch3d.renderer
import pytorch3d.structures
import pytorch3d.io
import pytorch3d.transforms
​
​
class Renderer(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.device = cfgs.get('device', 'cpu')
        self.render_resolution = cfgs.get('render_resolution', 64)
        self.fov = cfgs.get('fov', 10)
        self.blur_raduis = cfgs.get('blend_radius', np.log(1. / 1e-4 - 1.)*1e-4)
        self.cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=self.fov, device=self.device)
        self.image_renderer = self._create_image_renderer()
        init_verts, init_faces, init_aux = pytorch3d.io.load_obj(cfgs['init_shape.obj'], device=self.device)
        self.tex_faces_uv = init_faces.textures_idx.unsqueeze(0)
        self.tex_verts_uv = init_aux.verts_uvs.unsqueeze(0)
​

    def _create_image_renderer(self):
        raster_settings = _get_rasterization_settings()
        renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
                                shader=SoftPhongShader(device=self.device, cameras=self.cameras))
        return renderer


    def _get_rasterization_settings(self):
        raster_settings = RasterizationSettings(image_size=self.render_resolution, blur_radius=self.blur_radius, faces_per_pixel=1)
        return raster_settings


    def _get_textures(self, tex_im)
        # tex_im is a tensor that contains the non-flipped and flipped albedo_map
        b, h, w, c = tex_im.shape
        assert w == self.render_resolution and h == self.render_resolution, "Texture image has the wrong resolution."
        # tex_maps = tex_im.permute(0,2,3,1)
        textures = pytorch3d.renderer.TexturesUV(maps=tex_im,  # texture maps are BxHxWx3
                                                 faces_uvs=self.tex_faces_uv.repeat(b, 1, 1),
                                                 verts_uvs=self.tex_verts_uv.repeat(b, 1, 1))
        return textures

    
    def _get_lights(self, light_direction, ambient_color, diffuse_color, specular_color=[0,0,0]):
        ambient_color = torch.cat(3*[canon_light_a]).view(1,3)
        diffuse_color = torch.cat(3*[canon_light_b]).view(1,3)
        specular_color=torch.zeros((1,3))
        lights = DirectionalLights(ambient_color=ambient_color, diffuse_color=diffuse_color, specular_color=specular_color, direction=direction)
        return lights
    

    def _get_transformed_mesh(self, meshes, view):
        # rotate mesh with pytorch functions
        # rotate non flipped mesh in one direction and flipped version in other direction

        R = axis_angle_to_matrix(view[:,:3])
        T = view[:,3:

        object_center = torch.tensor([[0.0,0.0,1.0]])
        shift_center = pytorch3d.transforms.Translate(-object_center, device=device)
        rotate = pytorch3d.transforms.Rotate(R, device=device)
        shift_back = pytorch3d.transforms.Translate(object_center, device=device)
        translate = pytorch3d.transforms.Translate(T, device=device)
        tsf = shift_center.compose(rotate, shift_back, translate)

        meshes_verts = meshes.verts_padded()
        new_mesh_verts = tsf.transform_points(meshes_verts)
        transformed_meshes = meshes.update_padded(new_mesh_verts) 
        return meshes

    def forward(self, meshes, albedo_maps, view, lightning):
        # Can both images (flipped and not flipped be rendered in one go?)

        textures = self.get_textures(albedo_maps)
        lights = self.get_lights(lightning)
        transformed_mesh = self.get_transformed_meshes(meshes, view)

        # replace texture at mesh
        self.transformed_mesh.textures = textures

        images = self.image_renderer(meshes_world=transformed_mesh,lights=lights)
        return images