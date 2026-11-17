# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .manip_loco.manip_loco import ManipLoco
from .manip_loco.b1z1_config import B1Z1RoughCfg, B1Z1RoughCfgPPO

import os

from legged_gym.utils.task_registry import task_registry

from .manip_loco.as2piper_config import AS2PiPERPlateCfg, AS2PiPERPlateCfgPPO
from .manip_loco.as2piper_forward import AS2PiPERForward
task_registry.register('as2piper_plate_forward', AS2PiPERForward, AS2PiPERPlateCfg(), AS2PiPERPlateCfgPPO(), 'manip_loco')
from .manip_loco.as2piper_hw import AS2PiPERHardware
from .manip_loco.as2piper_hw_config import AS2PiPERHardwareCfg,AS2PiPERHardwareCfgPPO
task_registry.register('as2piper_plate_hw',AS2PiPERHardware,AS2PiPERHardwareCfg(),AS2PiPERHardwareCfgPPO(),'manip_loco')
from .manip_loco.as2piper_gait import AS2PiPERGait,AS2PiPERGaitCfg,AS2PiPERGaitCfgPPO
task_registry.register('as2piper_plate_gait',AS2PiPERGait,AS2PiPERGaitCfg(),AS2PiPERGaitCfgPPO(),'manip_loco')
from .manip_loco.as2piper_walk_wbc import AS2PiPERWalkWBC,AS2PiPERWalkWBCCfg,AS2PiPERWalkWBCCfgPPO
task_registry.register('as2piper_plate_walk_wbc',AS2PiPERWalkWBC,AS2PiPERWalkWBCCfg(),AS2PiPERWalkWBCCfgPPO(),'manip_loco')
from .manip_loco.as2piper_walk_wbc import AS2PiPERWalkWBCNominalCfg
task_registry.register('as2piper_plate_walk_wbc_nominal',AS2PiPERWalkWBC,AS2PiPERWalkWBCNominalCfg(),AS2PiPERWalkWBCCfgPPO(),'manip_loco')
from .manip_loco.as2piper_walk_wbc_vbc import AS2PiPERWalkWBCVBC,AS2PiPERWalkWBCVBCCfg,AS2PiPERWalkWBCVBCCfgPPO
task_registry.register('as2piper_plate_walk_wbc_vbc',AS2PiPERWalkWBCVBC,AS2PiPERWalkWBCVBCCfg(),AS2PiPERWalkWBCVBCCfgPPO(),'manip_loco')
from .manip_loco.as2piper_vbc_port import AS2PiPERVBCPort,AS2PiPERVBCPortCfg,AS2PiPERVBCPortCfgPPO
task_registry.register('as2piper_plate_vbc_port',AS2PiPERVBCPort,AS2PiPERVBCPortCfg(),AS2PiPERVBCPortCfgPPO(),'manip_loco')
