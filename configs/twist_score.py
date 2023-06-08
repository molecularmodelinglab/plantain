
import torch
import torch.nn.functional as F
import dgl.backend as dF

from models.force_field import ScaleOutput, cdist_diff, rbf_encode
from models.twister_v2 import AttentionContract, FullNorm, NormAndLinear, PredEnergy, TwistBlock, TwistEncoder, TwistFFCoef, TwistIndex, TwistModule
from terrace import Batch, LazyLinear

class FastScore(TwistModule):

    def forward(self, x, coef, lig_pose, inference=False):
        self.start_forward()

        rbf_start = self.cfg.rbf_start
        rbf_end = self.cfg.rbf_end
        rbf_steps = self.cfg.rbf_steps
        full_rec_data = x.full_rec_data
        l_rf_coef = coef.l_rf_coef
        ll_coef = coef.ll_coef

        lig_coord = torch.cat(lig_pose.coord, -2)
        if lig_coord.dim() == 3:
            lig_coord = lig_coord.transpose(0,1)
        rec_coord = full_rec_data.ndata.coord
        if inference:
            lig_coord = lig_coord.unsqueeze(0)
            rec_coord = rec_coord.unsqueeze(0)
        else:
            lig_coord = dF.pad_packed_tensor(lig_coord, x.lig_graph.dgl().batch_num_nodes(), 0.0)
            rec_coord = dF.pad_packed_tensor(rec_coord, full_rec_data.dgl().batch_num_nodes(), 0.0)
        if lig_coord.dim() == 4:
            lig_coord = lig_coord.transpose(1,2)
            rec_coord = rec_coord.unsqueeze(1)
            l_rf_coef = l_rf_coef.unsqueeze(1)
            ll_coef = ll_coef.unsqueeze(1)

        l_rf_dist = cdist_diff(lig_coord, rec_coord)
        ll_dist = cdist_diff(lig_coord, lig_coord)

        l_rf_mask = (lig_coord[...,0] != 0.0).unsqueeze(-1) & (rec_coord[...,0] != 0.0).unsqueeze(-2)
        ll_mask =  (lig_coord[...,0] != 0.0).unsqueeze(-1) & (lig_coord[...,0] != 0.0).unsqueeze(-2)

        extra_ll = ~torch.eye(ll_mask.shape[-1], dtype=bool, device=ll_mask.device).unsqueeze(0)
        if lig_coord.dim() == 4:
            extra_ll = extra_ll.unsqueeze(1)
        ll_mask = ll_mask & extra_ll

        l_rf_rbfs = rbf_encode(l_rf_dist, rbf_start,rbf_end,rbf_steps)
        ll_rbfs = rbf_encode(ll_dist, rbf_start,rbf_end,rbf_steps)

        l_rf_data = l_rf_rbfs*l_rf_coef
        ll_data = ll_rbfs*ll_coef

        ll_col = self.make(AttentionContract, self.cfg)(ll_data, self.cfg.score.ll_heads, self.cfg.score.ll_feat, ll_mask.unsqueeze(-1))
        l_rf_col = self.make(AttentionContract, self.cfg)(l_rf_data, self.cfg.score.l_rf_heads, self.cfg.score.l_rf_feat, l_rf_mask.unsqueeze(-1))
        all_collapsed =torch.cat((ll_col, l_rf_col), -1)

        pred_dist = self.make(LazyLinear, 1)(all_collapsed).squeeze(-1).transpose(-1,-2).contiguous()

        single_hid = self.make(AttentionContract, self.cfg)(all_collapsed,
                                                            self.cfg.score.single_heads, 
                                                            self.cfg.score.single_feat, 
                                                            ll_mask[...,0].unsqueeze(-1))
        single_hid = self.make(NormAndLinear, self.cfg, self.cfg.score.single_hid)(single_hid)

        single_out = self.make(LazyLinear, 2)(F.leaky_relu(single_hid))

        pred_noise = single_out[...,0]
        pred_rmsd = single_out[...,1]

        energy = pred_dist.mean(-2)

        return Batch(PredEnergy, dist=pred_dist, rmsd=pred_rmsd, noise=pred_noise, energy=energy)

class TwistScore(TwistModule):

    def get_input_feats(self):
        return [ "lig_graph", "rec_graph", "full_rec_data" ]

    def get_hidden_feat(self, x):
        self.start_forward()

        twist_index = TwistIndex(x)

        td = self.make(TwistEncoder, self.cfg)(x, twist_index)
        for i in range(self.cfg.num_blocks):
            td = td + self.make(TwistBlock, self.cfg)(x, twist_index, td)
            if self.cfg.get("norm_after_add", False):
                td = self.make(FullNorm, self.cfg)(td)

        return Batch(TwistFFCoef,
            l_rf_coef = self.make(LazyLinear, self.cfg.rbf_steps)(F.leaky_relu(td.l_rf_feat)),
            ll_coef = self.make(LazyLinear, self.cfg.rbf_steps)(F.leaky_relu(td.ll_feat)),
            inv_dist_mat = self.make(LazyLinear, 1)(F.leaky_relu(td.l_rf_feat))[...,0]
        )
    
    def get_energy(self, x, coef, lig_pose, inference=False):
        return self.make(FastScore, self.cfg)(x, coef, lig_pose, inference)