# model.py (修正版)
import time
import torch
import torch.nn as nn

try:
    from torch_points import knn
except (ModuleNotFoundError, ImportError):
    from torch_points_kernels import knn

def _safe_knn(src, query, k, device):
    """
    Safe wrapper for knn:
    - Try calling knn on whatever device src/query are on.
    - If knn implementation raises (e.g. "CUDA version not implemented"), fall back to CPU call
      and then move results to `device`.
    Returns (idx, dist) both on `device`.
    """
    try:
        idx, dist = knn(src.contiguous(), query.contiguous(), k)
        # Move to desired device if necessary
        if idx.device != device:
            idx = idx.to(device)
        if dist.device != device:
            dist = dist.to(device)
        return idx, dist
    except Exception:
        # fallback to CPU knn then move results to device
        idx, dist = knn(src.cpu().contiguous(), query.cpu().contiguous(), k)
        return idx.to(device), dist.to(device)


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        """
            input: (B, d_in, N, K)
            output: (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())
        self.device = device

    def forward(self, coords, features, knn_output):
        """
            coords: (B, N, 3)  (should be on some device)
            features: (B, d_feat, N, 1)  (on same device as coords)
            knn_output: tuple (idx, dist), idx (B, N, K), dist (B, N, K)
            returns: (B, 2*d, N, K)
        """
        idx, dist = knn_output  # idx, dist are already moved to desired device by caller (_safe_knn)
        B, N, K = idx.size()

        # Build neighbors coordinates: want (B, 3, N, K)
        # extended_idx for gather must be same device as extended_coords
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)  # (B,3,N,K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)  # (B,3,N,K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)  # (B,3,N,K)

        # concat: [center_coords, neighbors, center - neighbors, dist]
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)  # shape -> (B,1,N,K)
        ), dim=-3)  # result (B, 10, N, K)

        # ensure concat and features are on same device
        if concat.device != self.device:
            concat = concat.to(self.device)
        if features.device != self.device:
            features = features.to(self.device)

        mlp_out = self.mlp(concat)  # (B, d, N, K)
        feat_expanded = features.expand(B, -1, N, K)  # (B, d, N, K) where features has d equal to mlp_out channels
        return torch.cat((mlp_out, feat_expanded), dim=-3)  # (B, 2*d, N, K)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        # Linear will be applied to last dim (channels)
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)  # after permute, this will be softmax over neighbor dim
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        """
            x: (B, d_in, N, K)
            returns: (B, d_out, N, 1)
        """
        # compute attention scores
        # permute to (B, N, K, d) so Linear acts on last dim
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # back to (B, d_in, N, K)

        # weighted sum over neighbors (K)
        features = torch.sum(scores * x, dim=-1, keepdim=True)  # (B, d_in, N, 1)

        return self.mlp(features)  # (B, d_out, N, 1)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2 * d_out)
        self.shortcut = SharedMLP(d_in, 2 * d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out // 2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out // 2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()
        self.device = device

    def forward(self, coords, features):
        """
            coords: (B, N, 3)  -- device could be GPU or CPU
            features: (B, d_in, N, 1) -- same device as coords ideally
            returns: (B, 2*d_out, N, 1)
        """
        # use safe knn that falls back to CPU if needed
        knn_idx, knn_dist = _safe_knn(coords, coords, self.num_neighbors, device=coords.device)

        knn_output = (knn_idx, knn_dist)
        x = self.mlp1(features)  # (B, d_out//2, N, 1)

        x = self.lse1(coords, x, knn_output)  # (B, d_out, N, K)
        x = self.pool1(x)  # (B, d_out//2, N, 1)

        x = self.lse2(coords, x, knn_output)  # (B, d_out, N, K)
        x = self.pool2(x)  # (B, d_out, N, 1)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))  # (B, 2*d_out, N, 1)


class RandLANet(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=32, decimation=4, device=torch.device('cpu')):
        super(RandLANet, self).__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        )

        # store desired device, but DO NOT force .to(device) here;
        # leave device placement to training script (mytrain.py calls model.to(device))
        self.device = device

    def forward(self, input):
        """
            input: (B, N, d_in)
            returns: (B, num_classes, N)
        """
        N = input.size(1)
        d = self.decimation

        # coordinates keep same device as input
        coords = input[..., :3].clone()  # (B, N, 3) on input.device

        # initial feature projection: (B, N, d_in) -> Linear -> (B, N, 8) -> transpose -> (B, 8, N) -> unsqueeze -> (B, 8, N, 1)
        x = self.fc_start(input).transpose(-2, -1).unsqueeze(-1)
        x = self.bn_start(x)  # (B, 8, N, 1)

        decimation_ratio = 1
        x_stack = []

        # permutation must be on same device as tensors we index
        permutation = torch.randperm(N, device=input.device)
        coords = coords[:, permutation]
        x = x[:, :, permutation]

        for lfa in self.encoder:
            # coords for current resolution: coords[:, :N//decimation_ratio]
            cur_coords = coords[:, : (N // decimation_ratio)]
            x = lfa(cur_coords, x)  # x shape (B, 2*d_out, N//decimation_ratio, 1)
            x_stack.append(x.clone())
            decimation_ratio *= d
            # downsample x for next level (keep first N//decimation_ratio points)
            x = x[:, :, : (N // decimation_ratio)]

        x = self.mlp(x)

        # DECODER: upsample using knn and skip connections
        for mlp in self.decoder:
            # current resolution coords: original coords at resolution N//decimation_ratio
            src_coords = coords[:, : (N // decimation_ratio)]
            target_coords = coords[:, : (d * N // decimation_ratio)]
            # find nearest neighbor indices from src_coords -> target_coords
            neighbors, _ = _safe_knn(src_coords, target_coords, 1, device=input.device)  # neighbors: (B, tgt_N, 1)
            # neighbors currently map target points to src indices; ensure long dtype
            neighbors = neighbors.long().to(input.device)

            # gather features from x (low-res) to upsampled points
            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)  # (B, C, tgt_N, 1)
            x_neighbors = torch.gather(x, -2, extended_neighbors)  # (B, C, tgt_N, 1)

            # pop skip connection from stack and concat
            skip = x_stack.pop()
            # ensure skip and x_neighbors are on same device
            if skip.device != x_neighbors.device:
                skip = skip.to(x_neighbors.device)
            x = torch.cat((x_neighbors, skip), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        # inverse permutation to restore original point order
        inv_perm = torch.argsort(permutation)
        x = x[:, :, inv_perm]

        scores = self.fc_end(x)  # (B, num_classes, N, 1)
        return scores.squeeze(-1)  # (B, num_classes, N)


if __name__ == '__main__':
    import time
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    d_in = 7
    # create a reasonable sized cloud for quick test (use smaller than full)
    cloud = 1000 * torch.randn(1, 2**12, d_in).to(device)
    model = RandLANet(d_in, 6, 16, 4, device)
    model = model.to(device)
    model.eval()

    t0 = time.time()
    pred = model(cloud)
    t1 = time.time()
    print("forward time:", t1 - t0)
    print("output shape:", pred.shape)


# import time

# import torch
# import torch.nn as nn

# try:
#     from torch_points import knn
# except (ModuleNotFoundError, ImportError):
#     from torch_points_kernels import knn

# class SharedMLP(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size=1,
#         stride=1,
#         transpose=False,
#         padding_mode='zeros',
#         bn=False,
#         activation_fn=None
#     ):
#         super(SharedMLP, self).__init__()

#         conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

#         self.conv = conv_fn(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding_mode=padding_mode
#         )
#         self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
#         self.activation_fn = activation_fn

#     def forward(self, input):
#         r"""
#             Forward pass of the network

#             Parameters
#             ----------
#             input: torch.Tensor, shape (B, d_in, N, K)

#             Returns
#             -------
#             torch.Tensor, shape (B, d_out, N, K)
#         """
#         x = self.conv(input)
#         if self.batch_norm:
#             x = self.batch_norm(x)
#         if self.activation_fn:
#             x = self.activation_fn(x)
#         return x


# class LocalSpatialEncoding(nn.Module):
#     def __init__(self, d, num_neighbors, device):
#         super(LocalSpatialEncoding, self).__init__()

#         self.num_neighbors = num_neighbors
#         self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

#         self.device = device

#     def forward(self, coords, features, knn_output):
#         r"""
#             Forward pass

#             Parameters
#             ----------
#             coords: torch.Tensor, shape (B, N, 3)
#                 coordinates of the point cloud
#             features: torch.Tensor, shape (B, d, N, 1)
#                 features of the point cloud
#             neighbors: tuple

#             Returns
#             -------
#             torch.Tensor, shape (B, 2*d, N, K)
#         """
#         # finding neighboring points
#         idx, dist = knn_output
#         B, N, K = idx.size()
#         # idx(B, N, K), coords(B, N, 3)
#         # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
#         extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
#         extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
#         neighbors = torch.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)
#         # if USE_CUDA:
#         #     neighbors = neighbors.cuda()

#         # relative point position encoding
#         concat = torch.cat((
#             extended_coords,
#             neighbors,
#             extended_coords - neighbors,
#             dist.unsqueeze(-3)
#         ), dim=-3).to(self.device)
#         return torch.cat((
#             self.mlp(concat),
#             features.expand(B, -1, N, K)
#         ), dim=-3)



# class AttentivePooling(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(AttentivePooling, self).__init__()

#         self.score_fn = nn.Sequential(
#             nn.Linear(in_channels, in_channels, bias=False),
#             nn.Softmax(dim=-2)
#         )
#         self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

#     def forward(self, x):
#         r"""
#             Forward pass

#             Parameters
#             ----------
#             x: torch.Tensor, shape (B, d_in, N, K)

#             Returns
#             -------
#             torch.Tensor, shape (B, d_out, N, 1)
#         """
#         # computing attention scores
#         scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

#         # sum over the neighbors
#         features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

#         return self.mlp(features)



# class LocalFeatureAggregation(nn.Module):
#     def __init__(self, d_in, d_out, num_neighbors, device):
#         super(LocalFeatureAggregation, self).__init__()

#         self.num_neighbors = num_neighbors

#         self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
#         self.mlp2 = SharedMLP(d_out, 2*d_out)
#         self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

#         self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
#         self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device)

#         self.pool1 = AttentivePooling(d_out, d_out//2)
#         self.pool2 = AttentivePooling(d_out, d_out)

#         self.lrelu = nn.LeakyReLU()

#     def forward(self, coords, features):
#         r"""
#             Forward pass

#             Parameters
#             ----------
#             coords: torch.Tensor, shape (B, N, 3)
#                 coordinates of the point cloud
#             features: torch.Tensor, shape (B, d_in, N, 1)
#                 features of the point cloud

#             Returns
#             -------
#             torch.Tensor, shape (B, 2*d_out, N, 1)
#         """
#         knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)

#         x = self.mlp1(features)

#         x = self.lse1(coords, x, knn_output)
#         x = self.pool1(x)

#         x = self.lse2(coords, x, knn_output)
#         x = self.pool2(x)

#         return self.lrelu(self.mlp2(x) + self.shortcut(features))



# class RandLANet(nn.Module):
#     def __init__(self, d_in, num_classes, num_neighbors=32, decimation=4, device=torch.device('cpu')):
#         super(RandLANet, self).__init__()
#         # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.num_neighbors = num_neighbors
#         self.decimation = decimation

#         self.fc_start = nn.Linear(d_in, 8)
#         self.bn_start = nn.Sequential(
#             nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
#             nn.LeakyReLU(0.2)
#         )

#         # encoding layers
#         self.encoder = nn.ModuleList([
#             LocalFeatureAggregation(8, 16, num_neighbors, device),
#             LocalFeatureAggregation(32, 64, num_neighbors, device),
#             LocalFeatureAggregation(128, 128, num_neighbors, device),
#             LocalFeatureAggregation(256, 256, num_neighbors, device)
#         ])

#         self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

#         # decoding layers
#         decoder_kwargs = dict(
#             transpose=True,
#             bn=True,
#             activation_fn=nn.ReLU()
#         )
#         self.decoder = nn.ModuleList([
#             SharedMLP(1024, 256, **decoder_kwargs),
#             SharedMLP(512, 128, **decoder_kwargs),
#             SharedMLP(256, 32, **decoder_kwargs),
#             SharedMLP(64, 8, **decoder_kwargs)
#         ])

#         # final semantic prediction
#         self.fc_end = nn.Sequential(
#             SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
#             SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
#             nn.Dropout(),
#             SharedMLP(32, num_classes)
#         )
#         self.device = device

#         self = self.to(device)

#     def forward(self, input):
#         r"""
#             Forward pass

#             Parameters
#             ----------
#             input: torch.Tensor, shape (B, N, d_in)
#                 input points

#             Returns
#             -------
#             torch.Tensor, shape (B, num_classes, N)
#                 segmentation scores for each point
#         """
#         N = input.size(1)
#         d = self.decimation

#         coords = input[...,:3].clone().cpu()
#         x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
#         x = self.bn_start(x) # shape (B, d, N, 1)

#         decimation_ratio = 1

#         # <<<<<<<<<< ENCODER
#         x_stack = []

#         permutation = torch.randperm(N)
#         coords = coords[:,permutation]
#         x = x[:,:,permutation]

#         for lfa in self.encoder:
#             # at iteration i, x.shape = (B, N//(d**i), d_in)
#             x = lfa(coords[:,:N//decimation_ratio], x)
#             x_stack.append(x.clone())
#             decimation_ratio *= d
#             x = x[:,:,:N//decimation_ratio]


#         # # >>>>>>>>>> ENCODER

#         x = self.mlp(x)

#         # <<<<<<<<<< DECODER
#         for mlp in self.decoder:
#             neighbors, _ = knn(
#                 coords[:,:N//decimation_ratio].cpu().contiguous(), # original set
#                 coords[:,:d*N//decimation_ratio].cpu().contiguous(), # upsampled set
#                 1
#             ) # shape (B, N, 1)
#             neighbors = neighbors.to(self.device)

#             extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

#             x_neighbors = torch.gather(x, -2, extended_neighbors)

#             x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

#             x = mlp(x)

#             decimation_ratio //= d

#         # >>>>>>>>>> DECODER
#         # inverse permutation
#         x = x[:,:,torch.argsort(permutation)]

#         scores = self.fc_end(x)

#         return scores.squeeze(-1)


# if __name__ == '__main__':
#     import time
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     d_in = 7
#     cloud = 1000*torch.randn(1, 2**16, d_in).to(device)
#     model = RandLANet(d_in, 6, 16, 4, device)
#     # model.load_state_dict(torch.load('checkpoints/checkpoint_100.pth'))
#     model.eval()

#     t0 = time.time()
#     pred = model(cloud)
#     t1 = time.time()
#     # print(pred)
#     print(t1-t0)
