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


def create_meshes_from_vertices_and_faces(vertices, faces):
    meshes = pytorch3d.structures.Meshes(verts=vertices.to(self.device), faces=faces.to(self.device), textures=textures.to(self.device))
    return meshes