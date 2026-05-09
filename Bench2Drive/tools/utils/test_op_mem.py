from projects.mmdet3d_plugin.ops import deformable_aggregation_func, deformable_format
import torch

def get_max_memory():
    device = "cuda:0"
    mem = torch.cuda.max_memory_allocated(device=device)
    mem_mb = torch.tensor([mem / (1024 * 1024)],
        dtype=torch.int,
        device=device)
    return mem_mb.item()

bs = 8
N = 6
H = 256
W = 704
C = 256

num_anchor = 900
num_pts = 13
num_all_pts = num_anchor * num_pts

stride = [4,8,16,32]
stride = [8,16,32,64]
num_scale = len(stride)
num_group = 8
num_feat = sum([H//s*W//s for s in stride])
num_depth = 45

feature_maps = [torch.rand((bs, N, C, H//s, W//s)).cuda() for s in stride]
feature_maps = deformable_format(feature_maps)
points_2d = torch.rand((bs, num_all_pts, N, 2)).cuda()
weights = torch.rand((bs, num_all_pts, N, num_scale, num_group)).cuda()
depth_prob = torch.rand((bs, N, num_feat, num_depth)).cuda()
depth = torch.rand((bs, num_all_pts, N, 1)).cuda()


# features = deformable_aggregation_func(
#     *feature_maps, points_2d, weights, depth_prob, depth
# )
features_ = deformable_aggregation_func(
    *feature_maps, points_2d, weights
)

print(get_max_memory())
import ipdb; ipdb.set_trace()