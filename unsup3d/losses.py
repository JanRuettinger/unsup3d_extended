from torch import autograd
import torch.nn.functional as F

EPS = 1e-7

def compute_bce(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    loss = F.binary_cross_entropy_with_logits(d_out, targets)
    return loss

def compute_lse(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    d_out = F.sigmoid(d_out)
    loss = F.mse_loss(d_out, targets, reduction='mean')
    return loss
    
def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.reshape(batch_size, -1).sum(1)
    return reg


def photometric_loss(im1, im2, mask=None, conf_sigma=None):
		loss = (im1-im2).abs()
		if conf_sigma is not None:
				loss = loss *2**0.5 / (conf_sigma +EPS) + (conf_sigma +EPS).log()
		if mask is not None:
				mask = mask.expand_as(loss)
				loss = (loss * mask).sum() / mask.sum()
		else:
				loss = loss.mean()
		return loss