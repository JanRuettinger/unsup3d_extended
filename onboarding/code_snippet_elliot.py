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
        self.image_size = cfgs.get('image_size', 64)
        self.fov = cfgs.get('fov', 30)
        self.camp_pos_z_offset = cfgs.get('camp_pos_z_offset', 15)
        self.max_range = np.tan(self.fov/2 /180 * np.pi) * self.camp_pos_z_offset
        self.cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=self.fov, device=self.device)
        self.image_renderer = self._create_image_renderer()
        init_verts, init_faces, init_aux = pytorch3d.io.load_obj(cfgs['init_shape_obj'], device=self.device)
        self.meshes = pytorch3d.structures.Meshes(verts=[init_verts], faces=[init_faces.verts_idx]).to(self.device)
        self.tex_faces_uv = init_faces.textures_idx.unsqueeze(0)
        self.tex_verts_uv = init_aux.verts_uvs.unsqueeze(0)
​
        self.num_shape_vertices = init_verts.size(0)
        cam_pos = torch.FloatTensor([[0, 0, self.camp_pos_z_offset]]).to(self.device)
        cam_at = torch.FloatTensor([[0, 0, 0]]).to(self.device)
        self.update_camera_pose(position=cam_pos, at=cam_at)
​
    def _get_soft_rasterizer_settings(self):
        blend_params = pytorch3d.renderer.BlendParams(sigma=1e-4, gamma=1e-4)
        settings = pytorch3d.renderer.RasterizationSettings(
            image_size=self.image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100,
        )
        return settings, blend_params
​
    def _create_silhouette_renderer(self):
        settings, blend_params = self._get_soft_rasterizer_settings()
        return pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=self.cameras, raster_settings=settings),
            shader=pytorch3d.renderer.SoftSilhouetteShader(blend_params=blend_params)
        )
​
    def _create_image_renderer(self):
        settings, blend_params = self._get_soft_rasterizer_settings()
        lights = pytorch3d.renderer.DirectionalLights(device=self.device,
                                                      ambient_color=((1., 1., 1.),),
                                                      diffuse_color=((0., 0., 0.),),
                                                      specular_color=((0., 0., 0.),),
                                                      direction=((0, 1, 0),))
        return pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=self.cameras, raster_settings=settings),
            shader=pytorch3d.renderer.SoftPhongShader(device=self.device, lights=lights, blend_params=blend_params)
        )
​
    def update_camera_pose(self, position, at):
        self.cameras.R = pytorch3d.renderer.look_at_rotation(position, at, device=self.device)
        self.cameras.T = -torch.bmm(self.cameras.R.transpose(1, 2), position[:, :, None])[:, :, 0]
​
    def transform_mesh(self, mesh, pose):
        b, f, _ = pose.shape
        rot_mat = pytorch3d.transforms.euler_angles_to_matrix(pose[...,:3].view(-1,3), convention='XYZ')
        tsf = pytorch3d.transforms.Rotate(rot_mat, device=pose.device)
        trans_xyz = pose[...,3:].tanh() * self.max_range
        tsf = tsf.compose(pytorch3d.transforms.Translate(trans_xyz.view(-1,3), device=pose.device))
​
        mesh_verts = mesh.verts_padded()
        new_mesh_verts = tsf.transform_points(mesh_verts)
        new_mesh = mesh.update_padded(new_mesh_verts)
        return new_mesh
​
    def get_deformed_mesh(self, shape, deform, pose):
        b, f, _, _ = shape.shape
        mesh = self.meshes.extend(b*f)
        mesh = mesh.offset_verts(shape.view(-1, 3))
​
        # per frame deformation
        deform_type = self.cfgs.get("frame_deform", "none")
        if deform_type == 'offset':
            mesh = mesh.update_padded(mesh.verts_padded() + deform)
        elif deform_type == 'none':
            pass
        else:
            raise ValueError('Invalid frame_deform value.')
​
        # rigid rotation + translation
        mesh = self.transform_mesh(mesh, pose)
​
        return mesh
​
    def get_textures(self, tex_im):
        b, f, c, h, w = tex_im.shape
​
        ## pad top half with zeros for ico sphere with bad texture uv
        tex_im = torch.cat([torch.zeros_like(tex_im), tex_im], 3)
        # tex_im = nn.functional.interpolate(tex_im, (h, w), mode='bilinear', align_corners=False)
        textures = pytorch3d.renderer.TexturesUV(maps=tex_im.view(b*f, c, h*2, w).permute(0, 2, 3, 1),  # texture maps are BxHxWx3
                                                 faces_uvs=self.tex_faces_uv.repeat(b*f, 1, 1),
                                                 verts_uvs=self.tex_verts_uv.repeat(b*f, 1, 1))
        return textures
​
    def forward(self, pose, texture, shape, deform):
        b, f, _ = pose.shape
        mesh = self.get_deformed_mesh(shape, deform, pose)
        flow = self.render_flow(mesh, b, f)  # Bx(F-1)xHxWx2
​
        mesh.textures = self.get_textures(texture)
        image = self.image_renderer(meshes_world=mesh, cameras=self.cameras)
        image = image.view(b, f, *image.shape[1:])
        return image, flow, mesh