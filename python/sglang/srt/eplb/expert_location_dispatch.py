# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.server_args import get_global_server_args

@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "random"]
    # (num_logical_experts,)
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    # (num_logical_experts, X)
    partial_logical_to_all_physical_map: torch.Tensor
    # (num_logical_experts,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int

    @classmethod
    def init_new(cls, layer_id: int):
        ep_dispatch_algorithm = get_global_server_args().ep_dispatch_algorithm
        expert_location_metadata = get_global_expert_location_metadata()
        assert expert_location_metadata is not None

        if ep_dispatch_algorithm is None:
            return None

        return cls(
            ep_dispatch_algorithm=ep_dispatch_algorithm,
            partial_logical_to_rank_dispatch_physical_map=(
                expert_location_metadata.logical_to_rank_dispatch_physical_map[
                    layer_id, :
                ]
                if expert_location_metadata.logical_to_rank_dispatch_physical_map
                is not None
                else None
            ),
            partial_logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map[
                layer_id, :
            ],
            partial_logical_to_all_physical_map_num_valid=expert_location_metadata.logical_to_all_physical_map_num_valid[
                layer_id, :
            ],
            num_physical_experts=expert_location_metadata.num_physical_experts,
        )


def transform_select_experts_inputs(
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    info: Optional[ExpertLocationDispatchInfo],
):
    if (info is not None) and (info.ep_dispatch_algorithm == "fake"):
        router_logits.uniform_(5, 10)
        if correction_bias is not None:
            correction_bias = torch.zeros_like(correction_bias)
    return router_logits, correction_bias


def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        phy = _topk_ids_logical_to_physical_static(topk_ids, info)
    elif info.ep_dispatch_algorithm in ["dynamic", "fake"]:
        phy = _topk_ids_logical_to_physical_dynamic(topk_ids, info)
    else:
        raise NotImplementedError(f"Unknown algorithm {info.ep_dispatch_algorithm}")
    
    elastic_ep_state = ElasticEPStateManager.instance()
    el_metadata = get_global_expert_location_metadata()
    if (elastic_ep_state is None) or (el_metadata is None):
        return phy
    
    # Remap physical experts in fault rank
    active_ranks = elastic_ep_state.active_ranks
    num_local_phy = el_metadata.num_local_physical_experts
    ranks = phy.div(num_local_phy, rounding_mode='floor')
    inactive_phyical_ids = (active_ranks[ranks] == 0)
    # Gather candidates
    candidates = info.partial_logical_to_all_physical_map[topk_ids] # (n tokens, top-k logical, n replicas)
    cand_ranks = candidates.div(num_local_phy, rounding_mode='floor')
    cand_is_active = (active_ranks[cand_ranks] == 1) & (candidates >= 0) # filter out -1 padding
    # Select active replications in them
    selected_replica_indices = torch.argmax(cand_is_active.to(torch.int8), dim=-1)
    active_phy_ids = candidates.gather(-1, selected_replica_indices.unsqueeze(-1)).squeeze(-1)
    # Remap inactive physical experts
    phy = torch.where(inactive_phyical_ids, active_phy_ids, phy)

    return phy

def _topk_ids_logical_to_physical_static(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    return info.partial_logical_to_rank_dispatch_physical_map[topk_ids]


def _topk_ids_logical_to_physical_dynamic(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    chosen_dispatch_index = (
        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
        % info.partial_logical_to_all_physical_map_num_valid[topk_ids]
    )
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]

    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids
