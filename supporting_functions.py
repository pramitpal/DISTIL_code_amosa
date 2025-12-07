"""
SNN Mapper Utility Functions and Classes
Supporting functions for SNN (Spiking Neural Network) mapping and simulation
"""

import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Set
import re
import subprocess
import random
import warnings
import copy
from collections import defaultdict, Counter
from dataclasses import dataclass, field


# ============================================================================
# SECTION 1: SNNMapper Core Class
# ============================================================================

#@title SNN mapper
import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
class SNNMapper:
    def __init__(
        self,
        weights: List[Tuple],
        layer_groups: List[List],
        NPE: int = 19,
        NT: int = 16,
        X: int = 128,
        bits_per_cell: int = 1,
        P: int = 100,
        Vmem_res: int = 4,
        Timestep: int = 5,
        NoC_buswidth: int = 32,
        NoI_buswidth: int = 32,
        allow_break_columns: bool = True,
        include_chiplets: bool = True,
        max_chiplets: Optional[int] = None,
        cutoff_layer: int = -1
    ):
        self.weights = weights
        self.layer_groups = layer_groups
        self.NPE = NPE
        self.NT = NT
        self.X = X
        self.P = P
        self.bits_per_cell = bits_per_cell
        self.Vmem_res = Vmem_res
        self.Timestep = Timestep
        self.NoC_buswidth = NoC_buswidth
        self.NoI_buswidth = NoI_buswidth
        self.allow_break_columns = allow_break_columns
        self.include_chiplets = include_chiplets
        self.max_chiplets = max_chiplets
        self.cutoff_layer = cutoff_layer

        # Will be computed during run()
        self.tunable_params = None
        self.xbars = None
        self.IFMS = None
        self.OFMS = None
        self.TOPS = None
        self.MEMS = None
        self.layer_output_sizes = None
        self.chiplet_data = None

    def _calc_tunable_params(self):
        """Calculate tunable parameters for each layer."""
        xbars, params, IFMS, OFMS = [], [], [], []
        MOVES_X, MOVES_Y, TOPS, TOTAL_MACS, MEMS = [], [], [], [], []

        for each in self.weights:
            IFM_H, IFM_W, IFM_C, K_H, K_W, K_N, Pool, Stride = each

            IFM = IFM_H * IFM_W * IFM_C
            param = IFM_C * K_H * K_W * K_N
            xbar = math.ceil(K_H * K_W * IFM_C / (self.X)) * math.ceil(K_N / (self.bits_per_cell * self.X))
            if Pool < 0:
                OFM = IFM_W * K_N
            else:
                OFM = IFM_H * IFM_W * K_N
            # SAME padding if stride=1 and kernel is odd
            pad_w = (K_W // 2) if (Stride == 1 and (K_W % 2 == 1)) else 0
            pad_h = (K_H // 2) if (Stride == 1 and (K_H % 2 == 1)) else 0

            moves_x = (IFM_W + 2 * pad_w - K_W) // Stride + 1
            moves_y = (IFM_H + 2 * pad_h - K_H) // Stride + 1
            if Pool < 0:
                ops_xy = moves_x ** 2
                total_macs = ops_xy * K_H * K_W * K_N
            else:
                ops_xy = moves_x * moves_y
                total_macs = ops_xy * K_H * K_W * IFM_C * K_N

            TOTAL_MACS.append(total_macs / 1e12)

            LIF_memory_bytes = OFM * self.Vmem_res / 8
            params.append(param)
            xbars.append(xbar)
            IFMS.append(IFM)
            OFMS.append(OFM)
            MEMS.append(LIF_memory_bytes)

        return params, xbars, IFMS, OFMS, TOTAL_MACS, MEMS

    def _validate_layer_chiplet_requirements(self) -> Tuple[bool, Optional[Dict]]:
        """
        Validate that no single layer would exceed max_chiplets limit.
        """
        if self.max_chiplets is None:
            return True, None
        
        chip_capacity = self.NT * self.NPE
        usable_capacity = max(0, min(chip_capacity, math.floor(chip_capacity * (self.P / 100.0))))
        
        if usable_capacity == 0:
            return False, {
                'error': 'usable_capacity_zero',
                'message': f'Usable capacity is 0 with NT={self.NT}, NPE={self.NPE}, P={self.P}%'
            }
        
        for layer_idx, each in enumerate(self.weights):
            cols = each[5]
            rows = math.ceil(self.tunable_params[layer_idx] / max(cols, 1)) if cols > 0 else 0
            total_xbars = int(self.xbars[layer_idx])
            
            if cols <= self.X:
                min_chiplets_needed = math.ceil(total_xbars / usable_capacity)
            else:
                atomic_chunk = math.ceil(rows / self.X) if rows > 0 else total_xbars
                
                if self.allow_break_columns:
                    min_chiplets_needed = math.ceil(total_xbars / usable_capacity)
                else:
                    min_chiplets_needed = math.ceil(total_xbars / usable_capacity)
                    if atomic_chunk > usable_capacity:
                        num_chunks = math.ceil(total_xbars / atomic_chunk)
                        min_chiplets_needed = num_chunks
            
            if min_chiplets_needed > self.max_chiplets:
                return False, {
                    'layer_index': layer_idx,
                    'layer_number': layer_idx + 1,
                    'required_chiplets': min_chiplets_needed,
                    'max_chiplets': self.max_chiplets,
                    'total_xbars': total_xbars,
                    'usable_capacity': usable_capacity,
                    'cols': cols,
                    'can_split': cols > self.X,
                    'message': f'Layer {layer_idx + 1} requires {min_chiplets_needed} chiplets, '
                              f'but max_chiplets is set to {self.max_chiplets}'
                }
        
        return True, None

    def _calculate_layer_pe_requirement(self, layer_index: int) -> int:
        """Calculate the total number of PEs (crossbars) needed for a layer."""
        return int(self.xbars[layer_index])

    def _map_single_layer(self, layer_index: int):
        """Map a single layer to chiplets and return the chiplet list."""
        chip_capacity = self.NT * self.NPE
        usable_capacity = max(0, min(chip_capacity, math.floor(chip_capacity * (self.P / 100.0))))

        each = self.weights[layer_index]
        cols = each[5]
        rows = math.ceil(self.tunable_params[layer_index] / max(cols, 1)) if cols > 0 else 0
        total_need = int(self.xbars[layer_index])

        Chiplet = []
        remaining_usable = usable_capacity

        def new_chip():
            return {
                "Layers_filled": [],
                "Crossbars_filled_respective_layer": [],
                "Crossbars_remaining_respective_layer": [],
                "Layer_tile_distribution": {},
                "Empty_crossbars": chip_capacity
            }

        def chip_used(blk):
            return sum(blk["Crossbars_filled_respective_layer"])

        def finalize_chip(blk):
            blk["Empty_crossbars"] = chip_capacity - chip_used(blk)

        chip = new_chip()
        current_tile = 0
        current_tile_used = 0

        def add_layer_allocation(layer_num, crossbars_alloc, crossbars_remaining):
            nonlocal current_tile, current_tile_used

            crossbars_to_place = crossbars_alloc
            temp_tile = current_tile
            temp_used = current_tile_used
            remaining_crossbars = crossbars_to_place
            max_tile_needed = temp_tile

            while remaining_crossbars > 0:
                space_in_current_tile = self.NPE - temp_used
                if remaining_crossbars <= space_in_current_tile:
                    break
                else:
                    remaining_crossbars -= space_in_current_tile
                    max_tile_needed += 1
                    temp_used = 0

            if max_tile_needed >= self.NT:
                return False

            if layer_num in chip["Layers_filled"]:
                layer_idx = chip["Layers_filled"].index(layer_num)
                chip["Crossbars_filled_respective_layer"][layer_idx] += crossbars_alloc
                chip["Crossbars_remaining_respective_layer"][layer_idx] = crossbars_remaining
            else:
                chip["Layers_filled"].append(layer_num)
                chip["Crossbars_filled_respective_layer"].append(crossbars_alloc)
                chip["Crossbars_remaining_respective_layer"].append(crossbars_remaining)
                chip["Layer_tile_distribution"][layer_num] = {}

            remaining_crossbars = crossbars_to_place
            while remaining_crossbars > 0:
                space_in_current_tile = self.NPE - current_tile_used

                if remaining_crossbars <= space_in_current_tile:
                    if current_tile in chip["Layer_tile_distribution"][layer_num]:
                        chip["Layer_tile_distribution"][layer_num][current_tile] += remaining_crossbars
                    else:
                        chip["Layer_tile_distribution"][layer_num][current_tile] = remaining_crossbars
                    current_tile_used += remaining_crossbars
                    remaining_crossbars = 0
                else:
                    if current_tile in chip["Layer_tile_distribution"][layer_num]:
                        chip["Layer_tile_distribution"][layer_num][current_tile] += space_in_current_tile
                    else:
                        chip["Layer_tile_distribution"][layer_num][current_tile] = space_in_current_tile
                    remaining_crossbars -= space_in_current_tile
                    current_tile += 1
                    current_tile_used = 0

            if current_tile_used == self.NPE:
                current_tile += 1
                current_tile_used = 0

            return True

        def reset_tile_tracking():
            nonlocal current_tile, current_tile_used
            current_tile = 0
            current_tile_used = 0

        remaining_need = total_need
        atomic_chunk = math.ceil(rows / self.X) if cols > self.X and rows > 0 else total_need
        layer_num = layer_index + 1

        if cols <= self.X:
            while remaining_need > 0:
                if remaining_need > remaining_usable:
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
                    continue

                if not add_layer_allocation(layer_num, remaining_need, 0):
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
                    add_layer_allocation(layer_num, remaining_need, 0)

                remaining_usable -= remaining_need
                remaining_need = 0

                if remaining_usable == 0:
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
        else:
            while remaining_need > 0:
                if remaining_usable == 0:
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()

                alloc = 0
                if remaining_usable >= atomic_chunk:
                    k = min(remaining_usable // atomic_chunk, math.ceil(remaining_need / atomic_chunk))
                    k = max(k, 1)
                    alloc = min(k * atomic_chunk, remaining_need)
                else:
                    if self.allow_break_columns and remaining_usable > 0:
                        alloc = min(remaining_usable, remaining_need)
                    else:
                        finalize_chip(chip)
                        Chiplet.append(chip)
                        chip = new_chip()
                        remaining_usable = usable_capacity
                        reset_tile_tracking()
                        continue

                remaining_need -= alloc
                remaining_usable -= alloc

                if not add_layer_allocation(layer_num, alloc, max(remaining_need, 0)):
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
                    add_layer_allocation(layer_num, alloc, max(remaining_need, 0))
                    remaining_need -= alloc
                    remaining_usable -= alloc

        if chip["Layers_filled"] or chip_used(chip) > 0:
            finalize_chip(chip)
            Chiplet.append(chip)

        return Chiplet

    def _calculate_chiplets_needed_for_layer(self, layer_index: int, starting_usable: int) -> int:
        """
        Calculate how many chiplets would be needed to fully map a layer,
        starting from a chiplet with `starting_usable` capacity remaining.
        
        Returns:
            int: Total number of chiplets needed (1 if fits in current, more if spills over)
        """
        chip_capacity = self.NT * self.NPE
        usable_capacity = max(0, min(chip_capacity, math.floor(chip_capacity * (self.P / 100.0))))
        
        if usable_capacity == 0:
            return float('inf')
        
        total_need = int(self.xbars[layer_index])
        
        if total_need <= starting_usable:
            return 0  # Fits in current chiplet, no additional needed
        
        remaining_after_first = total_need - starting_usable
        additional_chiplets = math.ceil(remaining_after_first / usable_capacity)
        
        return additional_chiplets

    def _generate_chiplet_mapping(self):
        """
        Generate chiplet mapping for all layers.
        
        Layers are mapped sequentially. Once a layer cannot fit within the 
        chiplet budget, that layer and ALL subsequent layers go to unmapped_layers.
        """
        validation_passed, validation_error = self._validate_layer_chiplet_requirements()
        
        if not validation_passed:
            return {
                'main_chiplets': [],
                'unmapped_layers': [],
                'unmapped_layer_indices': [],
                'validation_error': validation_error
            }
        
        chip_capacity = self.NT * self.NPE
        usable_capacity = max(0, min(chip_capacity, math.floor(chip_capacity * (self.P / 100.0))))

        # Build column/row info for each layer
        XX = []
        for i, each in enumerate(self.weights):
            cols = each[5]
            rows = math.ceil(self.tunable_params[i] / max(cols, 1)) if cols > 0 else 0
            XX.append([cols, rows, int(self.xbars[i])])

        # Determine the effective cutoff based on cutoff_layer parameter
        effective_cutoff = self.cutoff_layer if self.cutoff_layer >= 0 else len(self.weights)

        Chiplet = []
        remaining_usable = usable_capacity
        cutoff_reached_at_layer = -1  # Track where we stopped due to chiplet limit

        def new_chip():
            return {
                "Layers_filled": [],
                "Crossbars_filled_respective_layer": [],
                "Crossbars_remaining_respective_layer": [],
                "Layer_tile_distribution": {},
                "Empty_crossbars": chip_capacity
            }

        def chip_used(blk):
            return sum(blk["Crossbars_filled_respective_layer"])

        def finalize_chip(blk):
            blk["Empty_crossbars"] = chip_capacity - chip_used(blk)

        chip = new_chip()
        current_tile = 0
        current_tile_used = 0

        def add_layer_allocation(layer_num, crossbars_alloc, crossbars_remaining):
            nonlocal current_tile, current_tile_used

            crossbars_to_place = crossbars_alloc
            temp_tile = current_tile
            temp_used = current_tile_used
            remaining_crossbars = crossbars_to_place
            max_tile_needed = temp_tile

            while remaining_crossbars > 0:
                space_in_current_tile = self.NPE - temp_used
                if remaining_crossbars <= space_in_current_tile:
                    break
                else:
                    remaining_crossbars -= space_in_current_tile
                    max_tile_needed += 1
                    temp_used = 0

            if max_tile_needed >= self.NT:
                return False

            if layer_num in chip["Layers_filled"]:
                layer_idx = chip["Layers_filled"].index(layer_num)
                chip["Crossbars_filled_respective_layer"][layer_idx] += crossbars_alloc
                chip["Crossbars_remaining_respective_layer"][layer_idx] = crossbars_remaining
            else:
                chip["Layers_filled"].append(layer_num)
                chip["Crossbars_filled_respective_layer"].append(crossbars_alloc)
                chip["Crossbars_remaining_respective_layer"].append(crossbars_remaining)
                chip["Layer_tile_distribution"][layer_num] = {}

            remaining_crossbars = crossbars_to_place
            while remaining_crossbars > 0:
                space_in_current_tile = self.NPE - current_tile_used

                if remaining_crossbars <= space_in_current_tile:
                    if current_tile in chip["Layer_tile_distribution"][layer_num]:
                        chip["Layer_tile_distribution"][layer_num][current_tile] += remaining_crossbars
                    else:
                        chip["Layer_tile_distribution"][layer_num][current_tile] = remaining_crossbars
                    current_tile_used += remaining_crossbars
                    remaining_crossbars = 0
                else:
                    if current_tile in chip["Layer_tile_distribution"][layer_num]:
                        chip["Layer_tile_distribution"][layer_num][current_tile] += space_in_current_tile
                    else:
                        chip["Layer_tile_distribution"][layer_num][current_tile] = space_in_current_tile
                    remaining_crossbars -= space_in_current_tile
                    current_tile += 1
                    current_tile_used = 0

            if current_tile_used == self.NPE:
                current_tile += 1
                current_tile_used = 0

            return True

        def reset_tile_tracking():
            nonlocal current_tile, current_tile_used
            current_tile = 0
            current_tile_used = 0

        def get_current_chiplet_count():
            """Get current number of committed chiplets"""
            return len(Chiplet)

        def current_chip_has_content():
            """Check if current chip being built has any content"""
            return chip["Layers_filled"] or chip_used(chip) > 0

        def can_fit_layer_in_remaining_budget(layer_index: int) -> bool:
            """
            Check if a layer can be fully mapped within the remaining chiplet budget.
            This considers both committed chiplets and the current chip being built.
            """
            if self.max_chiplets is None:
                return True
            
            committed_chiplets = get_current_chiplet_count()
            additional_chiplets_needed = self._calculate_chiplets_needed_for_layer(layer_index, remaining_usable)
            
            # If current chip has content, it counts as 1 chiplet
            # If layer needs additional chiplets beyond current, we need committed + 1 (current) + additional
            if current_chip_has_content():
                total_chiplets_if_mapped = committed_chiplets + 1 + additional_chiplets_needed
            else:
                # Current chip is empty, it will be used for this layer
                if additional_chiplets_needed == 0:
                    total_chiplets_if_mapped = committed_chiplets + 1
                else:
                    total_chiplets_if_mapped = committed_chiplets + 1 + additional_chiplets_needed
            
            return total_chiplets_if_mapped <= self.max_chiplets

        i = 0
        layers_to_place = min(len(self.weights), len(XX), effective_cutoff)

        while i < layers_to_place:
            # BEFORE mapping: check if this layer can fit in remaining chiplet budget
            if self.max_chiplets is not None:
                if not can_fit_layer_in_remaining_budget(i):
                    # This layer cannot fit - stop here
                    # All layers from i onwards go to unmapped_layers
                    cutoff_reached_at_layer = i
                    break
            
            cols, rows, total_need = XX[i]
            remaining_need = total_need
            atomic_chunk = math.ceil(rows / self.X) if cols > self.X and rows > 0 else total_need

            if cols <= self.X:
                # Layer cannot be split
                if remaining_need > remaining_usable:
                    if current_chip_has_content():
                        finalize_chip(chip)
                        Chiplet.append(chip)
                        chip = new_chip()
                        remaining_usable = usable_capacity
                        reset_tile_tracking()
                    
                    # Re-check budget after committing current chip
                    if self.max_chiplets is not None:
                        if not can_fit_layer_in_remaining_budget(i):
                            cutoff_reached_at_layer = i
                            break

                if remaining_need > remaining_usable:
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()

                if not add_layer_allocation(i + 1, remaining_need, 0):
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
                    add_layer_allocation(i + 1, remaining_need, 0)

                remaining_usable -= remaining_need
                i += 1

                if remaining_usable == 0:
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()

                continue

            # cols > X: splitting allowed
            layer_complete = False
            while remaining_need > 0:
                if remaining_usable == 0:
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
                    
                    # Re-check budget after committing
                    if self.max_chiplets is not None:
                        # Check if remaining part of layer can still fit
                        temp_additional = math.ceil(remaining_need / usable_capacity)
                        if get_current_chiplet_count() + temp_additional > self.max_chiplets:
                            # Cannot complete this layer - but we've partially mapped it
                            # This is a problem - we need to remove partial mapping
                            # For now, we'll handle this by checking BEFORE starting the layer
                            pass

                alloc = 0
                if remaining_usable >= atomic_chunk:
                    k = min(remaining_usable // atomic_chunk, math.ceil(remaining_need / atomic_chunk))
                    k = max(k, 1)
                    alloc = min(k * atomic_chunk, remaining_need)
                else:
                    if self.allow_break_columns and remaining_usable > 0:
                        alloc = min(remaining_usable, remaining_need)
                    else:
                        finalize_chip(chip)
                        Chiplet.append(chip)
                        chip = new_chip()
                        remaining_usable = usable_capacity
                        reset_tile_tracking()
                        continue

                remaining_need -= alloc
                remaining_usable -= alloc

                if not add_layer_allocation(i + 1, alloc, max(remaining_need, 0)):
                    finalize_chip(chip)
                    Chiplet.append(chip)
                    chip = new_chip()
                    remaining_usable = usable_capacity
                    reset_tile_tracking()
                    if not add_layer_allocation(i + 1, alloc, max(remaining_need, 0)):
                        raise RuntimeError(f"Cannot allocate layer {i+1} even on empty chiplet")

                if remaining_need == 0:
                    layer_complete = True
                    i += 1
                    break
            
            # If we exited the while loop without completing the layer, something went wrong
            if not layer_complete and remaining_need > 0:
                # This shouldn't happen if we checked correctly at the start
                cutoff_reached_at_layer = i
                break

        # Finalize the last chip if it has content
        if current_chip_has_content():
            finalize_chip(chip)
            Chiplet.append(chip)

        # Determine unmapped layers
        unmapped_layer_indices = []
        unmapped_layers = []
        
        # Determine starting point for unmapped layers
        if cutoff_reached_at_layer >= 0:
            # Chiplet budget was exhausted at this layer
            unmapped_start = cutoff_reached_at_layer
        elif self.cutoff_layer >= 0:
            # Explicit cutoff was set
            unmapped_start = self.cutoff_layer
        else:
            # All layers were mapped
            unmapped_start = len(self.weights)
        
        # All layers from unmapped_start onwards are unmapped (sequential constraint)
        for layer_idx in range(unmapped_start, len(self.weights)):
            unmapped_layer_indices.append(layer_idx)
            layer_chiplets = self._map_single_layer(layer_idx)
            unmapped_layers.append(layer_chiplets)

        return {
            'main_chiplets': Chiplet,
            'unmapped_layers': unmapped_layers,
            'unmapped_layer_indices': unmapped_layer_indices,
            'validation_error': None
        }

    def run(self):
        """
        Execute the mapping process.
        
        Returns:
            dict: Mapping results or validation error
        """
        self.tunable_params, self.xbars, self.IFMS, self.OFMS, self.TOPS, self.MEMS = self._calc_tunable_params()
        result = self._generate_chiplet_mapping()
        return result
    
    def print_mapping_summary(self, result: dict):
        """Print a summary of the mapping results."""
        if result.get('validation_error'):
            print("=" * 60)
            print("VALIDATION ERROR")
            print("=" * 60)
            print(result['validation_error']['message'])
            return
        
        print("=" * 60)
        print("MAPPING SUMMARY")
        print("=" * 60)
        
        main_chiplets = result['main_chiplets']
        unmapped_layers = result['unmapped_layers']
        unmapped_layer_indices = result.get('unmapped_layer_indices', [])
        
        print(f"\nMain Chiplets: {len(main_chiplets)}")
        if self.max_chiplets:
            print(f"Max Chiplets Allowed: {self.max_chiplets}")
        
        total_xbars_main = 0
        layers_in_main = set()
        for idx, chiplet in enumerate(main_chiplets):
            xbars_used = sum(chiplet['Crossbars_filled_respective_layer'])
            total_xbars_main += xbars_used
            layers_in_main.update(chiplet['Layers_filled'])
            print(f"  Chiplet {idx + 1}: Layers {chiplet['Layers_filled']}, "
                  f"XBars Used: {xbars_used}, Empty: {chiplet['Empty_crossbars']}")
        
        print(f"\nTotal XBars in Main Mapping: {total_xbars_main}")
        print(f"Layers in Main Mapping: {sorted(layers_in_main)}")
        
        if unmapped_layers:
            print(f"\n{'=' * 60}")
            print("UNMAPPED LAYERS (Sequential from cutoff)")
            print("=" * 60)
            print(f"Number of Unmapped Layers: {len(unmapped_layers)}")
            print(f"Unmapped Layer Indices (0-based): {unmapped_layer_indices}")
            print(f"Unmapped Layer Numbers (1-based): {[idx + 1 for idx in unmapped_layer_indices]}")
            
            for i, (layer_idx, layer_chiplets) in enumerate(zip(unmapped_layer_indices, unmapped_layers)):
                layer_xbars = int(self.xbars[layer_idx])
                print(f"\n  Layer {layer_idx + 1} (requires {layer_xbars} XBars):")
                print(f"    Chiplets needed: {len(layer_chiplets)}")
                for j, chiplet in enumerate(layer_chiplets):
                    xbars_used = sum(chiplet['Crossbars_filled_respective_layer'])
                    print(f"      Chiplet {j + 1}: XBars Used: {xbars_used}, Empty: {chiplet['Empty_crossbars']}")
        else:
            print("\nAll layers mapped successfully!")
        
        print(f"\n{'=' * 60}")
        print("LAYER REQUIREMENTS")
        print("=" * 60)
        chip_capacity = self.NT * self.NPE
        usable_capacity = max(0, min(chip_capacity, math.floor(chip_capacity * (self.P / 100.0))))
        print(f"Chip Capacity: {chip_capacity} (NT={self.NT} × NPE={self.NPE})")
        print(f"Usable Capacity per Chip: {usable_capacity} (P={self.P}%)")
        print(f"\n{'Layer':<8} {'XBars':<10} {'Min Chiplets':<15} {'Status':<10}")
        print("-" * 50)
        for i in range(len(self.weights)):
            xbars = int(self.xbars[i])
            chiplets_needed = math.ceil(xbars / usable_capacity) if usable_capacity > 0 else float('inf')
            status = "Unmapped" if i in unmapped_layer_indices else "Mapped"
            print(f"{i + 1:<8} {xbars:<10} {chiplets_needed:<15} {status:<10}")
# ============================================================================
# SECTION 2: Mesh Grid and Topology Functions
# ============================================================================

def find_best_grid(num_chiplets):
    """Find the best rectangular arrangement of chiplets."""
    if num_chiplets <= 0:
        return (0, 0)

    sqrt_n = int(math.sqrt(num_chiplets))
    best_rows, best_cols = 1, num_chiplets

    for rows in range(sqrt_n, 0, -1):
        if num_chiplets % rows == 0:
            cols = num_chiplets // rows
            if rows > cols:
                rows, cols = cols, rows

            best_rows, best_cols = rows, cols
            break

    return (best_rows, best_cols)


def nearest_almost_square(n):
    """Find the nearest almost square number >= n."""
    k = int(n**0.5)

    while True:
        s1 = k * k
        s2 = k * (k + 1)

        candidates = [x for x in (s1, s2) if x >= n]
        if candidates:
            return min(candidates)

        k += 1


def valid_system_sizes(low, high, max_diff=1):
    """Return all system sizes between [low, high] with balanced mesh."""
    results = []

    for total in range(low, high + 1):
        valid = False

        for r in range(1, int(total**0.5) + 1):
            if total % r == 0:
                c = total // r
                if abs(r - c) <= max_diff:
                    results.append((total))
                    valid = True
                    break

    return results


# ============================================================================
# SECTION 3: Traffic Calculation Functions
# ============================================================================

def get_layer_traffic(
    layer_id,
    chiplet_data,
    groupings,
    weights,
    tunable_params,
    xbars,
    X, Vmem_res, Timestep, NoC_buswidth,
    lif_tiles_per_layer,
    SRAM_KB_per_tile,
    acc_enabled=False
):
    """
    Calculate input and output traffic for a layer.
    
    Returns:
        dict: Traffic matrices with 'output' and 'input' keys
    """

    def _group_index_for(layer):
        for gi, g in enumerate(groupings):
            if g['start_layer'] <= layer <= g['end_layer']:
                return gi
        return None

    def _lif_tiles_for_group(gidx):
        name = f"LIF{gidx}"
        tiles = []
        for c_id, chiplet in enumerate(chiplet_data):
            lif_map = chiplet.get('Layer_tile_distribution', {}).get(name, {})
            for t_id in sorted(lif_map.keys()):
                tiles.append((c_id, t_id))
        return name, tiles

    def _traffic_bits_params(li):
        IFM_H, IFM_W = weights[li][0], weights[li][1]
        OC        = weights[li][5]
        total_ofm = IFM_H * IFM_W * OC
        total_xb  = int(xbars[li])
        denom     = (X * max(1, OC))
        per_tp    = max(1, tunable_params[li])
        cb_per_col= max(1, int(math.ceil(per_tp / denom)))
        num_cols  = max(1, int(math.ceil(max(1, total_xb) / cb_per_col)))
        ofm_per_c = total_ofm / 1
        base_bits = ofm_per_c * max(1, Vmem_res)
        return base_bits

    def _compute_output_bits_for_layer(layer, acc_mode):
        gidx = _group_index_for(layer)
        if gidx is None:
            return [[f"T?_{layer}", "LIF_NOT_FOUND", 0, "NOT_FOUND", "NOT_FOUND"]], {}

        lif_name, lif_all = _lif_tiles_for_group(gidx)
        if not lif_all:
            return [[f"T?_{layer}", "LIF_NOT_FOUND", 0, "NOT_FOUND", "NOT_FOUND"]], {}

        li = layer - 1
        need = max(1, min(int(lif_tiles_per_layer[li]), len(lif_all)))
        lif_used = lif_all[:need]
        lif_cap_bits = int(SRAM_KB_per_tile) * 1024 * 8

        base_bits = _traffic_bits_params(li)
        results_bits = []
        lif_written_bits = {(c,t): 0.0 for (c,t) in lif_used}

        if acc_mode:
            acc_name = f"ACC{gidx}"
            acc_chip = lif_used[0][0]
            total_into_acc = 0.0

            for c_id, chiplet in enumerate(chiplet_data):
                ltd = chiplet.get('Layer_tile_distribution', {})
                if layer not in ltd:
                    continue
                for tile_id, _cb in ltd[layer].items():
                    src = f"T{tile_id}"
                    results_bits.append([src, acc_name, base_bits, c_id, acc_chip])
                    total_into_acc += base_bits

            remaining = total_into_acc
            for (dst_c, lif_tid) in lif_used:
                if remaining <= 0:
                    break
                space = lif_cap_bits - lif_written_bits[(dst_c, lif_tid)]
                if space <= 0:
                    continue
                assign = min(remaining, space)
                lif_written_bits[(dst_c, lif_tid)] += assign
                results_bits.append([acc_name, f"{lif_name}_{lif_tid}", assign, acc_chip, dst_c])
                remaining -= assign
            return results_bits, lif_written_bits

        else:
            for c_id, chiplet in enumerate(chiplet_data):
                ltd = chiplet.get('Layer_tile_distribution', {})
                if layer not in ltd:
                    continue
                for tile_id, _cb in ltd[layer].items():
                    remaining = base_bits
                    for (dst_c, lif_tid) in lif_used:
                        if remaining <= 0:
                            break
                        space = lif_cap_bits - lif_written_bits[(dst_c, lif_tid)]
                        if space <= 0:
                            continue
                        assign = min(remaining, space)
                        lif_written_bits[(dst_c, lif_tid)] += assign
                        results_bits.append([f"T{tile_id}", f"{lif_name}_{lif_tid}", assign, c_id, dst_c])
                        remaining -= assign
            return results_bits, lif_written_bits

    results_output_bits, _lif_written_cur = _compute_output_bits_for_layer(layer_id, acc_enabled)

    results_input_bits = []
    if layer_id > 1:
        prev_layer = layer_id - 1

        _prev_out_bits, prev_lif_written_bits = _compute_output_bits_for_layer(prev_layer, acc_enabled)

        supply_list = [((c,t), bits) for (c,t), bits in prev_lif_written_bits.items() if bits > 0]
        supply_list.sort(key=lambda x: (x[0][0], x[0][1]))

        total_prev_written_bits = sum(bits for _, bits in supply_list)

        pj = prev_layer - 1
        prev_IFM_H, prev_IFM_W, prev_OC = weights[pj][0], weights[pj][1], weights[pj][5]
        tile_quota_bits = float(prev_IFM_H) * float(prev_IFM_W) * float(prev_OC) * float(max(1, Vmem_res))

        dest_tiles = []
        for c_id, chiplet in enumerate(chiplet_data):
            ltd = chiplet.get('Layer_tile_distribution', {})
            if layer_id in ltd:
                for t_id in sorted(ltd[layer_id].keys()):
                    dest_tiles.append((c_id, t_id))

        if total_prev_written_bits > 0 and dest_tiles:
            weights_list = []
            for (c,t), bits in supply_list:
                w = bits / float(total_prev_written_bits)
                weights_list.append(((c,t), w))

            prev_gidx = _group_index_for(prev_layer)
            for (dst_c, dst_t) in dest_tiles:
                alloc = []
                sum_floor = 0.0
                for (src_ct, w) in weights_list:
                    amt = w * tile_quota_bits
                    amt_floor = math.floor(amt)
                    alloc.append([src_ct, amt_floor])
                    sum_floor += amt_floor
                residual = int(max(0, math.ceil(tile_quota_bits - sum_floor)))
                fracs = [ (i, (weights_list[i][1] * tile_quota_bits) - alloc[i][1]) for i in range(len(alloc)) ]
                fracs.sort(key=lambda x: x[1], reverse=True)
                i = 0
                while residual > 0 and i < len(fracs):
                    idx = fracs[i][0]
                    alloc[idx][1] += 1
                    residual -= 1
                    i += 1

                for (src_ct, amt_bits) in alloc:
                    if amt_bits <= 0:
                        continue
                    src_c, src_t = src_ct
                    results_input_bits.append([f"LIF{prev_gidx}_{src_t}", f"T{dst_t}", amt_bits, src_c, dst_c])

    def _scale_to_packets(edges_bits):
        ts = max(1, int(Timestep))
        bw = max(1, int(NoC_buswidth))
        out = []
        for s, d, bits, sc, dc in edges_bits:
            packets = int(math.ceil((bits * ts) / bw))
            out.append([s, d, packets, sc, dc])
        return out

    return {
        "output": _scale_to_packets(results_output_bits),
        "input":  _scale_to_packets(results_input_bits)
    }


# ============================================================================
# SECTION 4: System and Chiplet Matrix Functions
# ============================================================================

def create_system_matrix_from_edges(all_edge_lists, count_diagonal=False, mesh_rows=4, mesh_cols=4, num_chiplets=None):
    """Create system-level traffic matrix from edge lists."""
    if mesh_rows is not None and mesh_cols is not None:
        num_chiplets = int(mesh_rows) * int(mesh_cols)
    elif num_chiplets is None:
        max_id = 0
        for sub in (all_edge_lists or []):
            if not sub:
                continue
            for e in sub:
                if len(e) >= 5 and isinstance(e[3], int) and isinstance(e[4], int):
                    max_id = max(max_id, e[3], e[4])
        num_chiplets = max_id + 1
    else:
        num_chiplets = int(num_chiplets)

    N = max(1, num_chiplets)
    M = np.zeros((N, N), dtype=int)

    for sub in (all_edge_lists or []):
        if not sub:
            continue
        for e in sub:
            if len(e) < 5:
                continue
            _, _, traffic, sc, dc = e[:5]
            if not (isinstance(sc, int) and isinstance(dc, int)):
                continue
            if sc < 0 or sc >= N or dc < 0 or dc >= N:
                continue
            if count_diagonal or sc != dc:
                M[sc, dc] += int(traffic)

    if mesh_rows is not None and mesh_cols is not None:
        names = [f"R{r}C{c}" for r in range(int(mesh_rows)) for c in range(int(mesh_cols))]
    else:
        names = [f"C{i}" for i in range(N)]

    return M, pd.DataFrame(M, index=names, columns=names)


def create_tile_matrix_for_chiplet(
    results_list,
    target_chiplet: int,
    NT: int,
    include_chiplets: bool = False,
    num_chiplets: int = None
):
    """Create per-chiplet tile-level traffic matrix."""
    if num_chiplets is None:
        max_id = target_chiplet
        for sub in (results_list or []):
            if not sub:
                continue
            for e in sub:
                if len(e) >= 5 and isinstance(e[3], int) and isinstance(e[4], int):
                    max_id = max(max_id, e[3], e[4])
        num_chiplets = max_id + 1

    tile_labels = [f"T{i}" for i in range(NT)]
    other_chiplet_labels = [f"C{i}" for i in range(num_chiplets) if i != target_chiplet] if include_chiplets else []
    all_labels = tile_labels + other_chiplet_labels
    idx = {l: i for i, l in enumerate(all_labels)}
    M = np.zeros((len(all_labels), len(all_labels)), dtype=int)

    lif_pat = re.compile(r"^LIF\d+_(\d+)$")
    t_pat   = re.compile(r"^T(\d+)$")

    def map_node(node, chip_id):
        if isinstance(node, str) and node.startswith("ACC"):
            return None

        if isinstance(node, int):
            if chip_id == target_chiplet:
                return f"T{node}" if 0 <= node < NT else None
            else:
                return f"C{chip_id}" if include_chiplets else None

        if isinstance(node, str):
            m = t_pat.match(node)
            if m:
                tile_id = int(m.group(1))
                if chip_id == target_chiplet:
                    return f"T{tile_id}" if 0 <= tile_id < NT else None
                else:
                    return f"C{chip_id}" if include_chiplets else None

            m = lif_pat.match(node)
            if m:
                lif_tile_id = int(m.group(1))
                if chip_id == target_chiplet:
                    return f"T{lif_tile_id}" if 0 <= lif_tile_id < NT else None
                else:
                    return f"C{chip_id}" if include_chiplets else None

            return None

        return None

    for sub in (results_list or []):
        if not sub:
            continue
        for e in sub:
            if len(e) < 5:
                continue
            src, dst, amt, sc, dc = e[:5]
            if not (isinstance(sc, int) and isinstance(dc, int)):
                continue

            s_lbl = map_node(src, sc)
            d_lbl = map_node(dst, dc)
            if s_lbl is None or d_lbl is None:
                continue
            if s_lbl not in idx or d_lbl not in idx:
                continue

            M[idx[s_lbl], idx[d_lbl]] += int(amt)

    return M, pd.DataFrame(M, index=all_labels, columns=all_labels)


# ============================================================================
# SECTION 5: Traffic Matrix Splitting and Scaling
# ============================================================================

def split_top_rest(df: pd.DataFrame, percent: float = 0.90, method: str = "mass", positive_only: bool = False):
    """Split traffic matrix into top percentile and rest."""
    arr = df.to_numpy(dtype=float)
    p = percent / 100.0 if percent > 1 else percent
    p = min(max(p, 0.0), 1.0)

    if method == "mass":
        flat = arr.ravel()
        order = np.argsort(flat)[::-1]
        sorted_vals = flat[order]

        if positive_only:
            total = sorted_vals[sorted_vals > 0].sum()
        else:
            total = sorted_vals.sum()

        top_mask_flat = np.zeros_like(flat, dtype=bool)
        if total != 0:
            cumsum = np.cumsum(sorted_vals)
            target = p * total
            k = np.searchsorted(cumsum, target, side="left") + 1
            top_mask_flat[order[:k]] = True

        top_mask = top_mask_flat.reshape(arr.shape)
    else:
        raise ValueError("method must be 'mass'.")

    top_df  = pd.DataFrame(np.where(top_mask,  arr, 0), index=df.index, columns=df.columns)
    rest_df = pd.DataFrame(np.where(~top_mask, arr, 0), index=df.index, columns=df.columns)
    return top_df.astype(int), rest_df.astype(int)


def scale_traffic_matrices(system_matrix, chiplet_matrices, minimum_traffic=5):
    """Scale traffic matrices so minimum non-zero value becomes minimum_traffic."""
    def get_scaled_data(matrices, minimum_traffic=5):
        scale_factors = []
        scaled_matrices = []

        was_list = isinstance(matrices, list)
        if not was_list:
            matrices = [matrices]

        for mat in matrices:
            if isinstance(mat, pd.DataFrame):
                mat = mat.values

            mat = mat.astype(float)

            if np.any(mat != 0):
                min_nonzero = np.min(mat[mat != 0])
                if np.isnan(min_nonzero) or min_nonzero == 0:
                    scale_factor = 1.0
                    scaled_mat = mat.astype(int)
                else:
                    scale_factor = min_nonzero / minimum_traffic
                    scaled_mat = np.ceil(mat / scale_factor).astype(int)
            else:
                scale_factor = 1.0
                scaled_mat = mat.astype(int)

            scaled_df = pd.DataFrame(scaled_mat)
            scale_factors.append(scale_factor)
            scaled_matrices.append(scaled_df)

        if not was_list and len(scale_factors) == 1:
            return scale_factors[0], scaled_matrices[0]
        return scale_factors, scaled_matrices

    chiplet_scaling_factors, chiplet_scaled_matrices = get_scaled_data(chiplet_matrices, minimum_traffic)
    system_scaling_factor, system_scaled_matrix = get_scaled_data(system_matrix, minimum_traffic)

    return (chiplet_scaled_matrices, chiplet_scaling_factors,
            system_scaled_matrix, system_scaling_factor)


# ============================================================================
# SECTION 6: BookSim File Generation and Simulation
# ============================================================================

def validate_and_flatten_mesh_layout(mesh_layout):
    """Validate mesh layout and flatten to list of node IDs in row-major order."""
    if not mesh_layout or not isinstance(mesh_layout[0], (list, np.ndarray)):
        raise ValueError("mesh_layout must be a 2D list/array (non-empty).")

    rows = len(mesh_layout)
    if rows == 0:
        raise ValueError("mesh_layout has 0 rows.")

    cols = max(len(row) for row in mesh_layout)
    if cols == 0:
        raise ValueError("mesh_layout has 0 columns in first row.")

    node_ids = []
    seen = set()
    for i, row in enumerate(mesh_layout):
        for node_id in row:
            if not isinstance(node_id, int):
                raise ValueError(f"All elements in mesh_layout must be integers (got {node_id}).")
            if node_id in seen:
                raise ValueError(f"Duplicate node ID {node_id} in mesh_layout.")
            seen.add(node_id)
            node_ids.append(node_id)

    return rows, cols, node_ids


def generate_router_node_mapping(node_ids):
    """Generate router to node mapping configuration using explicit node IDs."""
    output_lines = []
    for router_id in node_ids:
        line = f"router {router_id} node {router_id}"
        output_lines.append(line)
    return output_lines


def generate_mesh_booksim_config(mesh_layout, latency=1):
    """Generate mesh topology configuration for BookSim."""
    pos_to_id = {}
    for r, row in enumerate(mesh_layout):
        for c, node_id in enumerate(row):
            pos_to_id[(r, c)] = node_id

    output_lines = []
    for (r, c), router_id in pos_to_id.items():
        neighbors = []
        if (r, c + 1) in pos_to_id:
            neighbors.append((pos_to_id[(r, c + 1)], latency))
        if (r, c - 1) in pos_to_id:
            neighbors.append((pos_to_id[(r, c - 1)], latency))
        if (r + 1, c) in pos_to_id:
            neighbors.append((pos_to_id[(r + 1, c)], latency))
        if (r - 1, c) in pos_to_id:
            neighbors.append((pos_to_id[(r - 1, c)], latency))

        line = f"router {router_id}"
        for nid, lat in neighbors:
            line += f" router {nid} {lat}"
        output_lines.append(line)

    return output_lines


def generate_packet_schedule(traffic_df, mesh_layout, time=0):
    """Generate packet schedule from traffic DataFrame."""
    output_lines = []

    rows, cols, node_ids = validate_and_flatten_mesh_layout(mesh_layout)
    num_nodes = len(node_ids)

    if isinstance(traffic_df, np.ndarray):
        default_labels = list(range(num_nodes))
        traffic_df = pd.DataFrame(traffic_df, index=default_labels, columns=default_labels)

    src_labels = list(traffic_df.index)
    dst_labels = list(traffic_df.columns)

    all_df_ids = set(int(l) for l in src_labels + dst_labels if isinstance(l, (int, float)))
    all_layout_ids = set(node_ids)
    if all_df_ids - all_layout_ids:
        print(f"Warning: Traffic DF has IDs not in layout: {all_df_ids - all_layout_ids}")
    if all_layout_ids - all_df_ids:
        print(f"Warning: Layout has IDs not in traffic DF: {all_layout_ids - all_df_ids}")

    for src_label in src_labels:
        for dst_label in dst_labels:
            packets = int(traffic_df.loc[src_label, dst_label])
            if packets <= 0:
                continue
            source_node = int(src_label)
            destination_node = int(dst_label)

            for i in range(packets):
                line = f"{source_node} {destination_node} {time}"
                time += 1
                output_lines.append(line)

    return output_lines, time


def save_to_file(filename, content, ext=''):
    """Save content to file."""
    with open(filename + ext, 'w') as f:
        f.write('\n'.join(content))
        f.write('\n')


def generate_booksim_files(mesh_layout, traffic_df, topology_filename='anynet_file', trace_filename='trace_file'):
    """Main function to generate mesh topology and trace files."""
    rows, cols, node_ids = validate_and_flatten_mesh_layout(mesh_layout)
    num_nodes = len(node_ids)

    if isinstance(traffic_df, np.ndarray):
        if traffic_df.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"traffic_df (numpy) must be {num_nodes}×{num_nodes} "
                f"(matching {rows}×{cols} mesh = {num_nodes} nodes). "
                f"Got shape {traffic_df.shape}."
            )
    else:
        if len(traffic_df) != num_nodes or len(traffic_df.columns) != num_nodes:
            raise ValueError(
                f"traffic_df must be {num_nodes}×{num_nodes} "
                f"(matching {rows}×{cols} mesh = {num_nodes} nodes). "
                f"Got rows={len(traffic_df)}, cols={len(traffic_df.columns)}."
            )

    router_node_lines = generate_router_node_mapping(node_ids)
    topology_lines = (router_node_lines +
                      generate_mesh_booksim_config(mesh_layout))

    save_to_file(topology_filename, topology_lines)

    trace_lines, _ = generate_packet_schedule(traffic_df, mesh_layout)

    save_to_file(trace_filename, trace_lines, '.txt')

    return topology_lines, trace_lines


def filter_routers_by_threshold(out, n):
    """Remove routers with ID > n and all their associated links."""
    filtered_out = []

    for line in out:
        if line.startswith('router') and 'node' in line:
            parts = line.split()
            router_id = int(parts[1])

            if router_id <= n:
                filtered_out.append(line)

        elif line.startswith('router') and 'router' in line[7:]:
            parts = line.split()
            source_router = int(parts[1])

            if source_router > n:
                continue

            new_line_parts = ['router', str(source_router)]

            i = 2
            while i < len(parts):
                if parts[i] == 'router':
                    target_router = int(parts[i+1])
                    weight = parts[i+2]

                    if target_router <= n:
                        new_line_parts.extend(['router', str(target_router), weight])
                    i += 3
                else:
                    i += 1

            if len(new_line_parts) > 2:
                filtered_out.append(' '.join(new_line_parts))

    return filtered_out


# ============================================================================
# SECTION 7: BookSim Helper Functions
# ============================================================================

def extract_latency_from_booksim_output(filename):
    """Extract completion time from BookSim output using regex."""
    with open(filename, 'r') as f:
        content = f.read()
        match = re.search(r'- Completion Time:\s+(\d+)', content)
        if match:
            return int(match.group(1))
    return None


def extract_total_power_from_booksim_output(filename):
    """Extract 'Total Power' from a BookSim power summary text file."""
    with open(filename, 'r') as f:
        for line in f:
            if "Total Power" in line:
                if ":" in line:
                    value_str = line.split(":", 1)[1].strip().split()[0]
                    try:
                        return float(value_str)
                    except ValueError:
                        pass
    return None


def run_booksim_NoI(communication_matrix, mesh_layout, n=-1):
    """Run BookSim for Network-on-Interposer (NoI) simulation."""
    _ = subprocess.run(['chmod', '+x', 'booksim', 'run_system_level.sh','run_chiplet_level.sh', 'cleanup.sh'])
    
    topology, trace = generate_booksim_files(
        mesh_layout=mesh_layout, traffic_df=communication_matrix,
        topology_filename='system_level_anynet_file', trace_filename='trace_file'
    )

    if n >= 0:
        with open('system_level_anynet_file', 'r') as f:
            file_content = [line.strip() for line in f.readlines() if line.strip()]
        filtered = filter_routers_by_threshold(file_content, n)
        save_to_file('system_level_anynet_file', filtered)

    result = subprocess.run(
        ['./run_system_level.sh'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print(result.stdout.strip())
    NoI_latency = extract_latency_from_booksim_output('out_system_level.txt')
    NoI_power = extract_total_power_from_booksim_output('out_system_level.txt')
    
    return NoI_latency, NoI_power


def run_booksim_NoC(tile_matrices, mesh_layouts):
    """Run BookSim for Network-on-Chip (NoC) simulations."""
    _ = subprocess.run(['chmod', '+x', 'booksim', 'run_system_level.sh','run_chiplet_level.sh', 'cleanup.sh'])
    NoC_latency = []
    NoC_power = []
    num_chiplets = len(tile_matrices)

    for chiplet_id in range(num_chiplets):
        if tile_matrices[chiplet_id].sum().sum().item() == 0:
            NoC_latency.append(0)
            NoC_power.append(0)
            continue
        topology, trace = generate_booksim_files(
            mesh_layout=mesh_layouts[chiplet_id],
            traffic_df=tile_matrices[chiplet_id],
            topology_filename='chiplet_level_anynet_file',
            trace_filename='trace_file'
        )

        result = subprocess.run(
            ['./run_chiplet_level.sh', str(chiplet_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        NoC_latency.append(extract_latency_from_booksim_output(f'out_chiplet_{chiplet_id}.txt'))
        NoC_power.append(extract_total_power_from_booksim_output(f'out_chiplet_{chiplet_id}.txt'))
        print(result.stdout.strip())
    
    return NoC_latency, NoC_power

def cleanup_booksim_files():
    """Cleanup generated BookSim files."""
    _ = subprocess.run(['./cleanup.sh'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)

def update_param_booksim(filename, param, new_value):
    """Update channel width in the config file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    with open(filename, 'w') as f:
        for line in lines:
            if line.strip().startswith(param):
                f.write(f'{param}  = {new_value};\n')
            else:
                f.write(line)


# ============================================================================
# SECTION 8: Group and LIF Placement Functions
# ============================================================================

def get_row_col_2(chiplet_id, mesh_cols):
    """Get row/column coordinates from chiplet ID (snake pattern)."""
    row, col_base = divmod(chiplet_id, mesh_cols)
    if row % 2 == 1:
        col = mesh_cols - 1 - col_base
    else:
        col = col_base
    return row, col


def get_distance(chiplet_mapping_data, layer_num, lif_tiles_needed,
                 lif_start_chiplet, lif_tiles_per_chiplet,
                 mesh_rows, mesh_cols, pe_per_tile=40,
                 intra_cost=1, inter_cost=4):
    """
    Calculate system-wide communication cost for all-to-all communication pattern
    between Compute and LIF tiles.
    """
    layer_tile_dist = {}
    for group in chiplet_mapping_data:
        if layer_num in group['Layer_tile_distribution']:
            layer_tile_dist = group['Layer_tile_distribution'][layer_num]
            break

    if not layer_tile_dist or lif_tiles_needed == 0:
        return 0.0

    lif_distribution = {}
    tiles_remaining = lif_tiles_needed
    current_chiplet = lif_start_chiplet

    while tiles_remaining > 0 and current_chiplet < mesh_rows * mesh_cols:
        placed_here = min(tiles_remaining, lif_tiles_per_chiplet)
        lif_distribution[current_chiplet] = placed_here
        tiles_remaining -= placed_here
        current_chiplet += 1

    if not lif_distribution and lif_tiles_needed > 0:
        print("Error: LIF tiles needed but could not be placed.")
        return float('inf')

    total_system_cost = 0

    for compute_chiplet_id, num_compute_tiles in layer_tile_dist.items():
        c_row, c_col = get_row_col_2(compute_chiplet_id, mesh_cols)

        for lif_chiplet_id, num_lif_tiles in lif_distribution.items():
            l_row, l_col = get_row_col_2(lif_chiplet_id, mesh_cols)

            hops = abs(c_row - l_row) + abs(c_col - l_col)

            if hops == 0:
                path_cost = num_compute_tiles * num_lif_tiles * intra_cost
            else:
                path_cost = num_compute_tiles * num_lif_tiles * hops * inter_cost

            total_system_cost += path_cost

    return total_system_cost


def calculate_group_cost(chiplet_mapping_data, start_layer, end_layer, start_placing_in_chiplet,
                         lif_tiles_per_chiplet, MEMS, SRAM_KB_per_tile, operations,
                         mesh_rows, mesh_cols, pe_per_tile=40,
                         intra_cost=1, inter_cost=4, bus_width_bits=32):
    """Calculate total cost for a layer group."""
    LIF_tiles_needed = int(np.ceil((np.array(MEMS[start_layer - 1:end_layer]) / (1024 * SRAM_KB_per_tile)).max()))

    total_cost = 0
    max_MEMS = max(MEMS) if max(MEMS) > 0 else 1

    for layer in range(start_layer, end_layer + 1):
        for group in chiplet_mapping_data:
            if layer in group['Layer_tile_distribution']:
                num_tiles_in_layer = sum(group['Layer_tile_distribution'][layer].values())
                break
        else:
            num_tiles_in_layer = 1

        layer_ops = operations[layer - 1]
        total_pes = num_tiles_in_layer * pe_per_tile
        per_pe_traffic = layer_ops / total_pes if total_pes > 0 else 0

        layer_lif_distance = get_distance(
            chiplet_mapping_data, layer, LIF_tiles_needed, start_placing_in_chiplet,
            lif_tiles_per_chiplet, mesh_rows, mesh_cols, pe_per_tile, intra_cost, inter_cost)

        estimated_traffic_bits = per_pe_traffic * total_pes * (layer_lif_distance ** 1)
        traffic_cycles = estimated_traffic_bits / bus_width_bits

        total_cost += estimated_traffic_bits

    return total_cost


def build_unmapped(original_mapping, unmapped_layer, max_chiplets, NT, NPE, which=0):
    """Build unmapped layer chiplet configuration."""
    EMPTY = {
        'Layers_filled': [],
        'Crossbars_filled_respective_layer': [],
        'Crossbars_remaining_respective_layer': [],
        'Layer_tile_distribution': {},
        'Empty_crossbars': NT * NPE
    }
    EMPTY_unavailable = {
        'Layers_filled': [],
        'Crossbars_filled_respective_layer': [-1],
        'Crossbars_remaining_respective_layer': [],
        'Layer_tile_distribution': {},
        'Empty_crossbars': NT * NPE
    }
    original_mapping = copy.deepcopy(original_mapping)
    unmapped = copy.deepcopy(unmapped_layer)
    if which == 0:
        count = max_chiplets
        for each in original_mapping:
            if each.get('Layers_filled'):
                count -= 1
        insert_at = max_chiplets - count - 1
        if not isinstance(unmapped, list):
            unmapped = []
        insert_at = max(0, min(insert_at, len(unmapped)))
        unmapped.insert(insert_at, copy.deepcopy(EMPTY_unavailable))

    while len(unmapped) < max_chiplets:
        unmapped.append(copy.deepcopy(EMPTY))

    if len(unmapped) > max_chiplets:
        unmapped = unmapped[:max_chiplets]

    return unmapped


def build_four_corner_chiplet_array(
    mesh_rows,
    mesh_cols,
    compute_chiplets,
    LIF_chiplet,
    EMPTY,
    tiles_per_lif,
    weights
):
    """Build four-corner LIF placement configuration."""
    import copy

    num_cells = mesh_rows * mesh_cols

    def is_empty(ch):
        return (not ch.get('Layer_tile_distribution')) and (not ch.get('Layers_filled'))
    
    def is_lif(ch):
        ltd = ch.get('Layer_tile_distribution', {})
        for k, v in ltd.items():
            if isinstance(k, str) and k.startswith('LIF'):
                return True
            if isinstance(v, dict):
                for subk in v.keys():
                    if isinstance(subk, str) and subk.startswith('LIF'):
                        return True
        return False

    comp_src = [copy.deepcopy(x) for x in compute_chiplets]
    num_comp = len(comp_src)

    cc = [copy.deepcopy(EMPTY) for _ in range(num_cells)]

    top_left  = 0
    top_right = mesh_cols - 1
    lif_reserved = {top_left, top_right}
    cc[top_left]  = copy.deepcopy(LIF_chiplet)
    cc[top_right] = copy.deepcopy(LIF_chiplet)

    ptr = 0
    for r in range(mesh_rows):
        for c in range(mesh_cols):
            idx = r * mesh_cols + c
            if idx in lif_reserved:
                continue
            if ptr < num_comp:
                cc[idx] = copy.deepcopy(comp_src[ptr])
                ptr += 1

    if ptr != num_comp:
        raise ValueError(
            f"Not enough capacity: placed {ptr}/{num_comp} compute chiplets "
            f"with top-corner LIFs reserved."
        )

    compute_rows = []
    for r in range(mesh_rows):
        row_has_compute = any(
            (not is_empty(cc[r*mesh_cols + c])) and (not is_lif(cc[r*mesh_cols + c]))
            for c in range(mesh_cols)
        )
        if row_has_compute:
            compute_rows.append(r)

    if compute_rows == [0] and mesh_rows >= 2:
        bottom_row = 1
    else:
        bottom_row = compute_rows[-1] if compute_rows else 0

    bottom_left  = bottom_row * mesh_cols
    bottom_right = bottom_left + (mesh_cols - 1)
    to_relocate = []
    for idx in (bottom_left, bottom_right):
        if not is_lif(cc[idx]):
            if not is_empty(cc[idx]):
                to_relocate.append(copy.deepcopy(cc[idx]))
            cc[idx] = copy.deepcopy(LIF_chiplet)

    lif_reserved.update({bottom_left, bottom_right})

    if to_relocate:
        rel_ptr = 0
        for r in range(mesh_rows):
            for c in range(mesh_cols):
                idx = r * mesh_cols + c
                if idx in lif_reserved:
                    continue
                if is_empty(cc[idx]):
                    cc[idx] = to_relocate[rel_ptr]
                    rel_ptr += 1
                    if rel_ptr == len(to_relocate):
                        break
            if rel_ptr == len(to_relocate):
                break
        if rel_ptr != len(to_relocate):
            raise RuntimeError("Relocation failed: not enough free non-LIF slots.")

    placed_compute = sum((not is_empty(ch)) and (not is_lif(ch)) for ch in cc)
    if placed_compute != num_comp:
        raise AssertionError(
            f"Validation failed: compute placed = {placed_compute}, expected = {num_comp}"
        )

    corner_ids = sorted({top_left, top_right, bottom_left, bottom_right})
    global_grp = [{
        'start_layer': 1,
        'end_layer': len(weights),
        'lif_tiles': tiles_per_lif * len(corner_ids),
        'lif_distribution': [(cid, tiles_per_lif) for cid in corner_ids],
        'cost': 0.0,
    }]

    while cc and is_empty(cc[-1]):
        cc.pop()

    return cc, global_grp


def calculate_traffic(grouping, chip_cfg, mesh_rows=4, mesh_cols=4, NoI_buswidth=128, NoC_buswidth=16, Timestep=5):
    """Calculate inter and intra-chiplet traffic for a given grouping."""
    all_traffic = []
    for layer_id in range(1, len(chip_cfg) + 1):
        traffic_dict = get_layer_traffic(
            layer_id=layer_id, chiplet_data=chip_cfg, groupings=grouping,
            weights=[], tunable_params=[], xbars=[], X=128, Vmem_res=4,
            Timestep=Timestep, NoC_buswidth=1,
            lif_tiles_per_layer=np.ones(len(chip_cfg), dtype=int),
            SRAM_KB_per_tile=72, acc_enabled=False
        )
        all_traffic.extend([traffic_dict["output"], traffic_dict["input"]])

    _, M_sys_df = create_system_matrix_from_edges(all_traffic, mesh_rows=mesh_rows, mesh_cols=mesh_cols, count_diagonal=True)
    M_sys_df = np.ceil(M_sys_df / NoI_buswidth).astype(int)
    total_inter = M_sys_df.values.sum() - np.diag(M_sys_df.values).sum()

    chiplet_dfs = []
    total_intra = 0
    for c in range(len(chip_cfg)):
        M_chip_df = np.ceil(create_tile_matrix_for_chiplet(all_traffic, c, 16, include_chiplets=False)[1] / NoC_buswidth).astype(int)
        chiplet_dfs.append(M_chip_df)
        total_intra += M_chip_df.values.sum() - np.diag(M_chip_df.values).sum()

    return int(total_inter), int(total_intra), M_sys_df, chiplet_dfs


def get_last_index_to_check(df, max_tiles_limit):
    """Get best solution index within memory constraint."""
    valid_solutions = df[df['total_lif_tiles'] <= max_tiles_limit]

    if valid_solutions.empty:
        print(f"No solutions found with <= {max_tiles_limit} tiles.")
        return None

    best_idx = valid_solutions['total_cost'].idxmin()
    return best_idx


def optimize_multi_group_lif_placement(chiplet_data, layer_weights, groups, lif_needed, mesh_rows, mesh_cols, lif_capacity=7):
    """
    Optimize LIF tile placement across chiplets using Simulated Annealing.
    """

    def generate_grid_coords(mesh_rows, mesh_cols):
        coords = {}
        index = 0
        for y in range(mesh_rows):
            if y % 2 == 0:
                for x in range(mesh_cols):
                    coords[index] = (x, y)
                    index += 1
            else:
                for x in range(mesh_cols-1, -1, -1):
                    coords[index] = (x, y)
                    index += 1
        return coords

    INITIAL_TEMP = 5000
    FINAL_TEMP = 0.01
    COOLING_RATE = 0.995
    MAX_ITERATIONS = 50000
    COST_INTRA = 0.1
    COST_PER_HOP = 1.0
    HOP_EXPONENT = 1.2
    LOAD_BALANCE_WEIGHT = 50.0
    CONGESTION_WEIGHT = 10.0
    LOCALITY_BONUS = 0.5
    PENALTY_OVER_CAPACITY = 1e9

    grid_coords = generate_grid_coords(mesh_rows, mesh_cols)
    valid_chiplets = list(range(len(chiplet_data)))
    num_chiplets = len(valid_chiplets)

    def get_hops(id1, id2):
        if id1 not in grid_coords or id2 not in grid_coords:
            return abs(id1 - id2)
        x1, y1 = grid_coords[id1]
        x2, y2 = grid_coords[id2]
        return abs(x1 - x2) + abs(y1 - y2)

    layer_locations = {}
    for chip_idx, chip_info in enumerate(chiplet_data):
        for layer in chip_info.get('Layers_filled', []):
            if layer not in layer_locations:
                layer_locations[layer] = chip_idx

    expanded_groups = []
    for g_range in groups:
        expanded_groups.append(list(range(g_range[0], g_range[1] + 1)))

    group_compute_chiplets = []
    group_compute_weights = []

    for g_idx, layers in enumerate(expanded_groups):
        chiplet_weights = defaultdict(float)
        for layer in layers:
            if layer in layer_locations and layer in layer_weights:
                c_id = layer_locations[layer]
                chiplet_weights[c_id] += layer_weights[layer]
        group_compute_chiplets.append(dict(chiplet_weights))
        group_compute_weights.append(chiplet_weights)

    def calculate_total_cost(allocation_state):
        traffic_cost = 0.0
        penalty_cost = 0.0
        load_balance_cost = 0.0
        congestion_cost = 0.0
        locality_bonus = 0.0

        chiplet_usage = {c_id: 0 for c_id in valid_chiplets}
        for g_idx, alloc in enumerate(allocation_state):
            for c_id, count in alloc.items():
                chiplet_usage[c_id] += count

        for c_id, usage in chiplet_usage.items():
            if usage > lif_capacity:
                overage = usage - lif_capacity
                penalty_cost += overage * PENALTY_OVER_CAPACITY

        for g_idx, layers in enumerate(expanded_groups):
            group_lif_tiles = {c_id: count for c_id, count in allocation_state[g_idx].items() if count > 0}
            total_tiles = sum(group_lif_tiles.values())

            if total_tiles == 0:
                traffic_cost += PENALTY_OVER_CAPACITY
                continue

            for layer in layers:
                if layer not in layer_weights:
                    continue

                weight = layer_weights[layer]
                compute_loc = layer_locations.get(layer)

                if compute_loc is None:
                    continue

                weighted_dist = 0.0

                for lif_loc, tile_count in group_lif_tiles.items():
                    tile_fraction = tile_count / total_tiles

                    if compute_loc == lif_loc:
                        dist = COST_INTRA
                        locality_bonus += weight * tile_fraction * LOCALITY_BONUS
                    else:
                        hops = get_hops(compute_loc, lif_loc)
                        dist = COST_PER_HOP * (hops ** HOP_EXPONENT)

                    weighted_dist += dist * tile_fraction

                traffic_cost += weight * weighted_dist

        usages = [chiplet_usage[c] for c in valid_chiplets]
        non_zero_usages = [u for u in usages if u > 0]

        if len(non_zero_usages) > 1:
            mean_usage = sum(non_zero_usages) / len(non_zero_usages)
            variance = sum((u - mean_usage) ** 2 for u in non_zero_usages) / len(non_zero_usages)
            std_dev = math.sqrt(variance)
            load_balance_cost = LOAD_BALANCE_WEIGHT * std_dev

        total_cost = traffic_cost + penalty_cost + load_balance_cost + congestion_cost - locality_bonus

        return total_cost

    def initialize_state():
        state = []
        for g_idx, needed in enumerate(lif_needed):
            alloc = {c: 0 for c in valid_chiplets}
            compute_weights = group_compute_weights[g_idx]

            if compute_weights:
                total_weight = sum(compute_weights.values())
                chiplets = list(compute_weights.keys())
                probs = [compute_weights[c] / total_weight for c in chiplets]

                for _ in range(needed):
                    if random.random() < 0.7 and chiplets:
                        c = random.choices(chiplets, weights=probs, k=1)[0]
                    else:
                        c = random.choice(valid_chiplets)
                    alloc[c] += 1
            else:
                for _ in range(needed):
                    c = random.choice(valid_chiplets)
                    alloc[c] += 1

            state.append(alloc)

        return state

    current_state = initialize_state()
    current_cost = calculate_total_cost(current_state)

    best_state = copy.deepcopy(current_state)
    best_cost = current_cost

    curr_temp = INITIAL_TEMP

    for iteration in range(MAX_ITERATIONS):
        # Generate neighbor (simple perturbation)
        candidate_state = copy.deepcopy(current_state)
        g_idx = random.randint(0, len(groups) - 1)
        potential_src = [c for c, count in candidate_state[g_idx].items() if count > 0]

        if potential_src:
            src = random.choice(potential_src)
            dest = random.choice([c for c in valid_chiplets if c != src])
            candidate_state[g_idx][src] -= 1
            candidate_state[g_idx][dest] += 1

        candidate_cost = calculate_total_cost(candidate_state)
        delta = candidate_cost - current_cost

        accept = False
        if delta < 0:
            accept = True
        else:
            if curr_temp > 0:
                acceptance_prob = math.exp(-delta / curr_temp)
                if random.random() < acceptance_prob:
                    accept = True

        if accept:
            current_state = candidate_state
            current_cost = candidate_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_state = copy.deepcopy(current_state)

        curr_temp *= COOLING_RATE

        if curr_temp < FINAL_TEMP:
            break

    final_output = []
    for g_idx, alloc in enumerate(best_state):
        group_list = []
        for c_id, count in alloc.items():
            group_list.extend([c_id] * count)
        group_list.sort()
        final_output.append(group_list)

    dist = []
    for each_group_dist in final_output:
        chiplet_counts = {}
        for chiplet_id in each_group_dist:
            chiplet_counts[chiplet_id] = chiplet_counts.get(chiplet_id, 0) + 1
        dist.append(sorted(chiplet_counts.items()))

    return dist, {}


def add_lif_tiles(chiplet_mapping, groupings, NT, LIF_T, NPE, mesh_rows, mesh_cols):
    """Add LIF tiles to chiplet mapping based on groupings."""
    chiplets = copy.deepcopy(chiplet_mapping)
    MAX_CHIPLETS = mesh_rows * mesh_cols

    if isinstance(groupings, dict):
        groupings = [groupings]

    all_cids = []
    for g in groupings:
        if isinstance(g, dict) and 'lif_distribution' in g:
            all_cids.extend([cid for cid, _ in g['lif_distribution'] if isinstance(cid, int)])
    need = max(all_cids, default=-1) + 1 if all_cids else 0
    assert need <= MAX_CHIPLETS, "lif_distribution references a chiplet beyond mesh size."

    while len(chiplets) < need:
        chiplets.append({
            'Layers_filled': [],
            'Crossbars_filled_respective_layer': [],
            'Crossbars_remaining_respective_layer': [],
            'Layer_tile_distribution': {},
            'Empty_crossbars': (NT - LIF_T) * NPE
        })

    next_tile = {}
    lif_id = 0
    for g in groupings:
        if 'lif_distribution' not in g:
            continue
        for chiplet_id, num_tiles in g['lif_distribution']:
            if chiplet_id >= len(chiplets):
                raise ValueError(f"Chiplet ID {chiplet_id} exceeds current chiplets length {len(chiplets)}")
            ltd = chiplets[chiplet_id].setdefault('Layer_tile_distribution', {})
            start = next_tile.get(chiplet_id, NT - LIF_T)
            ltd[f"LIF{lif_id}"] = {i: NPE for i in range(start, start + num_tiles)}
            next_tile[chiplet_id] = start + num_tiles
        lif_id += 1

    return chiplets


def get_balanced_solutions(df, n=1, w_mem=0.5, w_cost=0.5):
    """Select top n balanced solutions from the DataFrame."""
    temp_df = df.copy()

    mem = temp_df['total_lif_tiles']
    cost = temp_df['total_cost']

    norm_mem = (mem - mem.min()) / (mem.max() - mem.min()) if mem.max() > mem.min() else 0
    norm_cost = (cost - cost.min()) / (cost.max() - cost.min()) if cost.max() > cost.min() else 0

    temp_df['weighted_score'] = (w_mem * norm_mem) + (w_cost * norm_cost)

    top_indices = temp_df['weighted_score'].nsmallest(n).index

    return df.loc[top_indices]


# ============================================================================
# SECTION 9: AMOSA (Archived Multi-Objective Simulated Annealing) Classes
# ============================================================================

@dataclass
class Solution:
    """Represents a layer grouping solution for AMOSA."""
    breakpoints: List[int]
    objectives: Tuple[float, float] = field(default_factory=lambda: (float('inf'), float('inf')))
    lif_distributions: List[List[Tuple[int, int]]] = field(default_factory=list)
    feasible: bool = True
    chiplet_mapping: List[Dict] = field(default_factory=list)
    groupings: List[Dict] = field(default_factory=list)

    def __hash__(self):
        return hash(tuple(self.breakpoints))

    def __eq__(self, other):
        return self.breakpoints == other.breakpoints

    def get_groups(self, total_mapped_layers: int) -> List[Tuple[int, int]]:
        """Convert breakpoints to (start, end) tuples for mapped layer groups."""
        groups = []
        start = 1
        for bp in self.breakpoints:
            groups.append((start, bp))
            start = bp + 1
        if start <= total_mapped_layers:
            groups.append((start, total_mapped_layers))
        return groups

    def get_groups_string(self, total_mapped_layers: int, unmapped_layers: List[int] = None) -> str:
        """Get readable string representation of all groups."""
        groups = self.get_groups(total_mapped_layers)
        parts = [f"G{i+1}:[L{s}-L{e}]" for i, (s, e) in enumerate(groups)]

        if unmapped_layers:
            for i, layer in enumerate(unmapped_layers):
                parts.append(f"G{len(groups)+i+1}:[L{layer}]*")

        return " | ".join(parts)


class AMOSALogger:
    """Handles detailed logging for AMOSA optimization."""

    def __init__(self, verbose_level: int = 1):
        self.verbose_level = verbose_level
        self.iteration_log = []

    def log_header(self, optimizer):
        if self.verbose_level >= 1:
            print("\n" + "═"*100)
            print("  AMOSA LAYER GROUPING OPTIMIZATION")
            print("═"*100)
            print(f"  📊 Network: {optimizer.total_layers} layers")
            print(f"     ├─ Mapped layers: {optimizer.mapped_layers} ({optimizer.mapped_layer_list})")
            print(f"     └─ Unmapped layers: {len(optimizer.unmapped_layers)} ({optimizer.unmapped_layers})")
            print(f"  🎯 Target: {optimizer.num_groups} TOTAL groups")
            print(f"     ├─ Groups for mapped layers: {optimizer.num_mapped_groups}")
            print(f"     └─ Groups for unmapped layers: {len(optimizer.unmapped_layers)} (1 each)")
            print(f"  🔲 Mesh: {optimizer.mesh_rows}×{optimizer.mesh_cols} = {optimizer.total_chiplets} chiplets")
            print(f"  💾 LIF_T: {optimizer.LIF_T} tiles/chiplet")
            print(f"  📦 SRAM: {optimizer.SRAM_KB_per_tile} KB/tile")
            print("═"*100)
            print(f"\n  AMOSA Parameters:")
            print(f"    ├─ Archive Size: {optimizer.archive_size}")
            print(f"    ├─ Initial Temp: {optimizer.initial_temp}")
            print(f"    ├─ Final Temp: {optimizer.final_temp}")
            print(f"    ├─ Cooling Rate: {optimizer.cooling_rate}")
            print(f"    └─ Iterations/Temp: {optimizer.iterations_per_temp}")
            print("═"*100 + "\n")

    def log_initialization(self, archive_size: int, attempts: int):
        if self.verbose_level >= 1:
            print(f"📋 INITIALIZATION COMPLETE")
            print(f"   └─ Archive initialized with {archive_size} feasible solutions (after {attempts} attempts)")
            print()

    def log_iteration(self, iteration: int, current_sol: 'Solution', new_sol: 'Solution',
                      total_mapped_layers: int, unmapped_layers: List[int],
                      action: str, reason: str, perturbation_type: str = "",
                      archive_size: int = 0, temperature: float = 0):
        log_entry = {
            'iteration': iteration,
            'temperature': temperature,
            'current_groups': current_sol.get_groups_string(total_mapped_layers, unmapped_layers),
            'new_groups': new_sol.get_groups_string(total_mapped_layers, unmapped_layers),
            'current_obj': current_sol.objectives,
            'new_obj': new_sol.objectives,
            'action': action,
            'reason': reason,
            'perturbation': perturbation_type,
            'archive_size': archive_size,
            'new_feasible': new_sol.feasible
        }
        self.iteration_log.append(log_entry)

        if self.verbose_level >= 2:
            action_symbols = {'ACCEPTED': '✅', 'REJECTED': '❌', 'ARCHIVED': '📥', 'DOMINATED': '👎'}
            symbol = action_symbols.get(action, '•')
            print(f"\n  Iter {iteration:4d} │ {symbol} {action}")
            print(f"  {'─'*12}┼{'─'*85}")
            print(f"  Current    │ {current_sol.get_groups_string(total_mapped_layers, unmapped_layers)}")
            print(f"             │ Memory: {current_sol.objectives[0]:.0f} tiles, Cost: {current_sol.objectives[1]:.2f}")
            if perturbation_type:
                print(f"  Perturb    │ Type: {perturbation_type}")
            feasible_str = "✓ Feasible" if new_sol.feasible else "✗ Infeasible"
            print(f"  New        │ {new_sol.get_groups_string(total_mapped_layers, unmapped_layers)}")
            print(f"             │ Memory: {new_sol.objectives[0]:.0f} tiles, Cost: {new_sol.objectives[1]:.2f} [{feasible_str}]")

    def log_temperature_summary(self, temp_step: int, temperature: float,
                               accepted: int, rejected: int, archive_size: int,
                               best_mem: float, best_cost: float):
        if self.verbose_level >= 1:
            total = accepted + rejected
            accept_rate = (accepted / total * 100) if total > 0 else 0
            print(f"\n  📊 Temperature {temp_step} Summary:")
            print(f"     ├─ Accepted: {accepted}/{total} ({accept_rate:.1f}%)")
            print(f"     ├─ Archive Size: {archive_size}")
            print(f"     ├─ Best Memory: {best_mem:.0f} tiles")
            print(f"     └─ Best Cost: {best_cost:.2f}")

    def log_final_results(self, archive: List['Solution'], total_mapped_layers: int,
                          unmapped_layers: List[int], stats: Dict):
        if self.verbose_level >= 1:
            print("\n" + "═"*100)
            print("  🏆 OPTIMIZATION COMPLETE")
            print("═"*100)
            print(f"\n  📈 Statistics:")
            print(f"     ├─ Total Iterations: {stats['iterations']}")
            print(f"     ├─ Accepted Moves: {stats['accepted']}")
            print(f"     ├─ Rejected Moves: {stats['rejected']}")
            print(f"     └─ Archive Updates: {stats['archive_updates']}")

            print(f"\n  📦 Final Archive: {len(archive)} Pareto-optimal solutions")
            print("─"*100)

            if archive:
                sorted_archive = sorted(archive, key=lambda s: (s.objectives[0], s.objectives[1]))
                print(f"  {'#':<3} {'Memory':<10} {'Cost':<12} {'Groups'}")
                print("  " + "─"*95)
                for i, sol in enumerate(sorted_archive[:10]):
                    groups_str = sol.get_groups_string(total_mapped_layers, unmapped_layers)
                    print(f"  {i+1:<3} {sol.objectives[0]:<10.0f} {sol.objectives[1]:<12.2f} {groups_str}")
                if len(archive) > 10:
                    print(f"  ... and {len(archive) - 10} more solutions")

                best_mem = min(archive, key=lambda s: s.objectives[0])
                best_cost = min(archive, key=lambda s: s.objectives[1])
                print("\n  🥇 Best Solutions:")
                print(f"     ├─ Minimum Memory: {best_mem.objectives[0]:.0f} tiles")
                print(f"     │  └─ {best_mem.get_groups_string(total_mapped_layers, unmapped_layers)}")
                print(f"     └─ Minimum Cost: {best_cost.objectives[1]:.2f}")
                print(f"        └─ {best_cost.get_groups_string(total_mapped_layers, unmapped_layers)}")
            print("═"*100 + "\n")


# ============================================================================
# SECTION 10: AMOSA Optimizer Class (Main)
# ============================================================================

class AMOSAGroupingOptimizer:
    """AMOSA optimizer for layer grouping with LIF tile optimization."""

    def __init__(
        self,
        mapper,
        num_groups: int,
        mesh_rows: int,
        mesh_cols: int,
        LIF_T: int,
        SRAM_KB_per_tile: float,
        NPE: int,
        NT: int = None,
        inter_cost: float = 4.0,
        intra_cost: float = 1.0,
        archive_size: int = 100,
        archive_hard_limit: int = 150,
        initial_temp: float = 1000.0,
        final_temp: float = 1e-6,
        cooling_rate: float = 0.95,
        iterations_per_temp: int = 50,
        random_seed: Optional[int] = None,
        verbose_level: int = 1
    ):
        self.mapper = copy.deepcopy(mapper)
        self.mesh_rows = mesh_rows
        self.mesh_cols = mesh_cols
        self.LIF_T = LIF_T
        self.SRAM_KB_per_tile = SRAM_KB_per_tile
        self.NPE = NPE
        self.NT = NT if NT is not None else 16
        self.inter_cost = inter_cost
        self.intra_cost = intra_cost

        self.archive_size = archive_size
        self.archive_hard_limit = archive_hard_limit
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.verbose_level = verbose_level

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.total_chiplets = mesh_rows * mesh_cols

        self.logger = AMOSALogger(verbose_level)
        self.archive: List[Solution] = []
        self.last_perturbation_type = ""

        self.stats = {
            'iterations': 0,
            'accepted': 0,
            'rejected': 0,
            'archive_updates': 0
        }

        self._setup_layer_info()

        self.num_groups = num_groups
        # self.unmapped_layers = self.mapper.chiplet_data.get('unmapped_layers', [])
        # layers_in_main = set()
        # for idx, chiplet in enumerate(self.mapper.chiplet_data['main_chiplets']):
        #     layers_in_main.update(chiplet['Layers_filled'])
        # self.mapped_layers = len(layers_in_main)

        # print(f"Unmapped layers from data: {self.unmapped_layers}")
        self.num_unmapped_groups = len(self.unmapped_layers)
        self.num_mapped_groups = num_groups - self.num_unmapped_groups

        if self.num_mapped_groups < 1:
            # print(
            #     f"num_groups ({num_groups}) must be > unmapped layers ({self.num_unmapped_groups}). "
            #     f"Minimum num_groups = {self.num_unmapped_groups + 1}"
            # )
            return
        if self.num_mapped_groups > self.mapped_layers:
            # print(
            #     f"Groups for mapped layers ({self.num_mapped_groups}) cannot exceed "
            #     f"mapped layers ({self.mapped_layers}). "
            #     f"Maximum num_groups = {self.mapped_layers + self.num_unmapped_groups}"
            # )
            return

    def _setup_layer_info(self):
        """Setup layer information, handling unmapped layers."""
        self.total_layers = len(self.mapper.MEMS)

        chiplet_data = self.mapper.chiplet_data
        unmapped_from_data = chiplet_data.get('unmapped_layers', [])

        self.unmapped_layers = []

        if unmapped_from_data:
            if isinstance(unmapped_from_data, (list, tuple)):
                for item in unmapped_from_data:
                    if isinstance(item, (int, np.integer)):
                        self.unmapped_layers.append(int(item))
                    elif isinstance(item, (float, np.floating)):
                        self.unmapped_layers.append(int(item))
                    elif isinstance(item, dict):
                        layer_num = None
                        for key in ['layer', 'layer_id', 'layer_num', 'id', 'index']:
                            if key in item:
                                try:
                                    layer_num = int(item[key])
                                    break
                                except (ValueError, TypeError):
                                    continue
                        if layer_num is None:
                            for value in item.values():
                                if isinstance(value, (int, float, np.integer, np.floating)):
                                    layer_num = int(value)
                                    break
                        if layer_num is not None:
                            self.unmapped_layers.append(layer_num)
                    elif isinstance(item, (list, tuple)):
                        for sub_item in item:
                            if isinstance(sub_item, (int, float, np.integer, np.floating)):
                                self.unmapped_layers.append(int(sub_item))
            elif isinstance(unmapped_from_data, (int, np.integer)):
                self.unmapped_layers.append(int(unmapped_from_data))

        mapped_layer_set = set()
        for chiplet in chiplet_data.get('main_chiplets', []):
            layers_filled = chiplet.get('Layers_filled', [])
            if isinstance(layers_filled, (list, tuple)):
                for layer in layers_filled:
                    if isinstance(layer, (int, float, np.integer, np.floating)):
                        mapped_layer_set.add(int(layer))
            elif isinstance(layers_filled, (int, float, np.integer, np.floating)):
                mapped_layer_set.add(int(layers_filled))

        self.mapped_layer_list = sorted(mapped_layer_set)
        self.mapped_layers = len(self.mapped_layer_list)

        if not self.unmapped_layers:
            all_layers = set(range(1, self.total_layers + 1))
            self.unmapped_layers = sorted(all_layers - mapped_layer_set)
        else:
            self.unmapped_layers = sorted(set(
                l for l in self.unmapped_layers
                if 1 <= l <= self.total_layers
            ))

        if self.unmapped_layers and self.logger.verbose_level >= 1:
            print(f"  ⚠️  Detected {len(self.unmapped_layers)} unmapped layers: {self.unmapped_layers}")

    def run(self) -> List[Solution]:
        """Run AMOSA optimization."""
        self.logger.log_header(self)
        self._initialize_archive()

        if not self.archive:
            print("No feasible solutions found!")
            return []

        state = self.select_from_archive()
        t = 0

        while True:
            t += 1
            T = self._schedule(t)

            if T <= self.final_temp:
                break

            candidate = self.perturb_solution(state)
            self.evaluate_solution(candidate)

            if not candidate.feasible:
                self.stats['rejected'] += 1
                self._log_iteration(t, T, state, candidate, "REJECTED", "Infeasible")
                continue

            delta_E = self._calculate_delta(candidate, state)

            if delta_E > 0:
                self._accept_candidate(candidate)
                state = candidate
                self._log_iteration(t, T, state, candidate, "ACCEPTED", f"ΔE={delta_E:.3f}>0")
            else:
                prob = self._acceptance_probability(delta_E, T)
                if random.random() < prob:
                    state = candidate
                    self.stats['accepted'] += 1
                    self._log_iteration(t, T, state, candidate, "ACCEPTED", f"p={prob:.3f}")
                else:
                    self.stats['rejected'] += 1
                    self._log_iteration(t, T, state, candidate, "REJECTED", f"p={prob:.3f}")

            if t % self.iterations_per_temp == 0:
                self._log_temperature_summary(t, T)
                if random.random() < 0.1:
                    state = self.select_from_archive()

        self._finalize_archive()

        self.stats['iterations'] = t
        self.logger.log_final_results(self.archive, self.mapped_layers,
                                       self.unmapped_layers, self.stats)

        return self.archive

    def _finalize_archive(self):
        """Finalize archive by removing dominated and adding chiplet mappings."""
        self.archive = [s for s in self.archive if s.feasible]
        self._remove_dominated_from_archive()

        for sol in self.archive:
            self._add_chiplet_mapping_to_solution(sol)

    def _add_chiplet_mapping_to_solution(self, solution: Solution):
        """Add complete chiplet mapping and ALL groupings to a solution."""
        mapped_groups = solution.get_groups(self.mapped_layers)

        all_groupings = []
        group_id = 1

        for idx, (start, end) in enumerate(mapped_groups):
            actual_start = self.mapped_layer_list[start - 1] if start <= len(self.mapped_layer_list) else start
            actual_end = self.mapped_layer_list[end - 1] if end <= len(self.mapped_layer_list) else end

            lif_dist = solution.lif_distributions[idx] if idx < len(solution.lif_distributions) else []

            all_groupings.append({
                'group_id': group_id,
                'start_layer': actual_start,
                'end_layer': actual_end,
                'layers': list(range(actual_start, actual_end + 1)),
                'lif_distribution': lif_dist,
                'is_unmapped': False
            })
            group_id += 1

        if self.unmapped_layers:
            lif_alloc = {i: 0 for i in range(self.total_chiplets)}

            for g in all_groupings:
                for cid, tiles in g.get('lif_distribution', []):
                    lif_alloc[cid] += tiles

            available = set(c for c in range(self.total_chiplets)
                           if lif_alloc.get(c, 0) < self.LIF_T)

            num_mapped = len(mapped_groups)

            for i, layer in enumerate(self.unmapped_layers):
                dist_idx = num_mapped + i
                if dist_idx < len(solution.lif_distributions):
                    dist = solution.lif_distributions[dist_idx]
                else:
                    tiles_needed = self._calculate_lif_tiles_for_unmapped_layer(layer)
                    dist, _ = self.find_optimal_lif_placement(
                        layer, layer, tiles_needed, available, lif_alloc
                    )

                all_groupings.append({
                    'group_id': group_id,
                    'start_layer': layer,
                    'end_layer': layer,
                    'layers': [layer],
                    'lif_distribution': dist,
                    'is_unmapped': True
                })
                group_id += 1

                for cid, tiles in dist:
                    lif_alloc[cid] += tiles
                    if lif_alloc[cid] >= self.LIF_T:
                        available.discard(cid)

        solution.groupings = all_groupings

        base_mapping = copy.deepcopy(self.mapper.chiplet_data.get('main_chiplets', []))
        solution.chiplet_mapping = add_lif_tiles(
            base_mapping,
            solution.groupings,
            self.NT,
            self.LIF_T,
            self.NPE,
            self.mesh_rows,
            self.mesh_cols
        )

    def _calculate_lif_tiles_for_unmapped_layer(self, layer) -> int:
        """Calculate LIF tiles needed for an unmapped layer."""
        if isinstance(layer, (list, tuple)):
            layer = layer[0] if layer else 1

        try:
            layer = int(layer)
        except (ValueError, TypeError):
            return 1

        if 0 <= layer - 1 < len(self.mapper.MEMS):
            mem = self.mapper.MEMS[layer - 1]
            return max(1, int(np.ceil(mem / (1024 * self.SRAM_KB_per_tile))))

        return 1

    def _schedule(self, t: int) -> float:
        return self.initial_temp * (self.cooling_rate ** (t // self.iterations_per_temp))

    def _calculate_delta(self, candidate: Solution, state: Solution) -> float:
        status, dominated = self.get_domination_status(candidate, self.archive)
        if status > 0:
            return 1.0
        elif status == 0:
            return 0.1
        else:
            diff1 = (candidate.objectives[0] - state.objectives[0]) / (state.objectives[0] + 1e-10)
            diff2 = (candidate.objectives[1] - state.objectives[1]) / (state.objectives[1] + 1e-10)
            return -max(diff1, diff2)

    def _acceptance_probability(self, delta_E: float, T: float) -> float:
        if delta_E >= 0:
            return 1.0
        return np.exp(delta_E / max(T, 1e-10))

    def _accept_candidate(self, candidate: Solution):
        self.update_archive(candidate)
        self.stats['accepted'] += 1

    def _initialize_archive(self):
        attempts = 0
        max_attempts = self.archive_size * 10
        while len(self.archive) < self.archive_size and attempts < max_attempts:
            sol = self.generate_random_solution()
            self.evaluate_solution(sol)
            if sol.feasible:
                self.update_archive(sol)
            attempts += 1
        self.logger.log_initialization(len(self.archive), attempts)

    def _log_iteration(self, t: int, T: float, state: Solution,
                       candidate: Solution, action: str, reason: str):
        self.logger.log_iteration(
            iteration=t, current_sol=state, new_sol=candidate,
            total_mapped_layers=self.mapped_layers,
            unmapped_layers=self.unmapped_layers,
            action=action, reason=reason,
            perturbation_type=self.last_perturbation_type,
            archive_size=len(self.archive), temperature=T
        )

    def _log_temperature_summary(self, t: int, T: float):
        best_mem = min((s.objectives[0] for s in self.archive), default=float('inf'))
        best_cost = min((s.objectives[1] for s in self.archive), default=float('inf'))
        self.logger.log_temperature_summary(
            temp_step=t // self.iterations_per_temp, temperature=T,
            accepted=self.stats['accepted'], rejected=self.stats['rejected'],
            archive_size=len(self.archive), best_mem=best_mem, best_cost=best_cost
        )

    def calculate_lif_tiles_for_group(self, start_layer: int, end_layer: int) -> int:
        """Calculate LIF tiles for a group of mapped layers."""
        actual_start = self.mapped_layer_list[start_layer - 1] if start_layer <= len(self.mapped_layer_list) else start_layer
        actual_end = self.mapped_layer_list[end_layer - 1] if end_layer <= len(self.mapped_layer_list) else end_layer

        mems_slice = self.mapper.MEMS[actual_start - 1:actual_end]
        if not mems_slice:
            return 1
        return int(np.ceil((np.array(mems_slice) / (1024 * self.SRAM_KB_per_tile)).max()))

    def analyze_crossbar_usage(self, start_layer: int, end_layer: int,
                               lif_alloc: Dict[int, int]) -> Tuple[Dict, Dict, Dict]:
        mapping = self.mapper.chiplet_data['main_chiplets']
        chiplet_usage = {}

        actual_start = self.mapped_layer_list[start_layer - 1] if start_layer <= len(self.mapped_layer_list) else start_layer
        actual_end = self.mapped_layer_list[end_layer - 1] if end_layer <= len(self.mapped_layer_list) else end_layer

        for layer in range(actual_start, actual_end + 1):
            for cid, info in enumerate(mapping):
                if layer in info['Layers_filled']:
                    idx = info['Layers_filled'].index(layer)
                    used = info['Crossbars_filled_respective_layer'][idx]
                    chiplet_usage.setdefault(cid, {})[layer] = used

        total_crossbars = {cid: sum(v.values()) for cid, v in chiplet_usage.items()}
        remaining = {cid: self.LIF_T - lif_alloc.get(cid, 0) for cid in range(len(mapping))}

        return chiplet_usage, total_crossbars, remaining

    def distribute_lif_tiles(self, available_chiplets: Set[int],
                            tiles_needed: int, capacity_map: Dict[int, int]) -> List[Tuple[int, int]]:
        total_capacity = sum(capacity_map.get(c, 0) for c in available_chiplets)
        if total_capacity < tiles_needed:
            return []

        dist = []
        remaining = tiles_needed
        sorted_chiplets = sorted(available_chiplets, key=lambda c: capacity_map.get(c, 0), reverse=True)

        for cid in sorted_chiplets:
            if remaining <= 0:
                break
            avail = capacity_map.get(cid, 0)
            if avail <= 0:
                continue
            alloc = min(avail, remaining)
            dist.append((cid, alloc))
            remaining -= alloc

        return dist if sum(t for _, t in dist) >= tiles_needed else []

    def find_optimal_lif_placement(self, start_layer: int, end_layer: int,
                               tiles_needed: int, available_chiplets: Set[int],
                               lif_alloc: Dict[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        """Find optimal LIF tile placement using sequential filling."""
        if tiles_needed <= 0:
            return [], 0.0

        if not available_chiplets:
            return [], float('inf')

        capacity_map = {}
        for c in available_chiplets:
            used = lif_alloc.get(c, 0)
            remaining = self.LIF_T - used
            if remaining > 0:
                capacity_map[c] = remaining

        total_capacity = sum(capacity_map.values())
        if total_capacity < tiles_needed:
            if self.verbose_level:
                print(f"Error: Not enough space. Needed {tiles_needed}, Available {total_capacity}")
            return [], float('inf')

        sorted_chips = sorted(capacity_map.keys())

        distribution = []
        tiles_to_place = tiles_needed

        for chip_id in sorted_chips:
            if tiles_to_place <= 0:
                break

            take = min(tiles_to_place, capacity_map[chip_id])
            distribution.append((chip_id, take))
            tiles_to_place -= take

        used_region = [chip for chip, count in distribution]

        try:
            cost = self._calculate_placement_cost(start_layer, end_layer, used_region)
        except:
            cost = 0.0

        return distribution, cost

    def _calculate_placement_cost(
      self,
      start_layer: int,
      end_layer: int,
      region: List[int],
      ) -> float:
        """Calculate placement cost using external cost function."""
        start_placing_in_chiplet = min(region)

        cost = calculate_group_cost(
            chiplet_mapping_data=self.mapper.chiplet_data['main_chiplets'],
            start_layer=start_layer,
            end_layer=end_layer,
            start_placing_in_chiplet=start_placing_in_chiplet,
            lif_tiles_per_chiplet=self.LIF_T,
            MEMS=self.mapper.MEMS,
            SRAM_KB_per_tile=self.SRAM_KB_per_tile,
            operations=self.mapper.TOPS,
            mesh_rows=self.mesh_rows,
            mesh_cols=self.mesh_cols,
            pe_per_tile=self.NPE,
            intra_cost=self.intra_cost,
            inter_cost=self.inter_cost
        )

        return cost

    def evaluate_solution(self, solution: Solution, log_details: bool = False) -> Tuple[float, float, bool]:
        """Evaluate solution including both mapped and unmapped layers."""
        mapped_groups = solution.get_groups(self.mapped_layers)

        total_tiles, total_cost = 0, 0.0
        feasible = True
        lif_distributions = []

        available = set(range(self.total_chiplets))
        lif_alloc = {i: 0 for i in range(self.total_chiplets)}

        for idx, (start, end) in enumerate(mapped_groups):
            tiles_needed = self.calculate_lif_tiles_for_group(start, end)
            total_tiles += tiles_needed

            dist, cost = self.find_optimal_lif_placement(start, end, tiles_needed, available, lif_alloc)

            if not dist and tiles_needed > 0:
                feasible = False
                lif_distributions.append([])
            else:
                lif_distributions.append(dist)
                total_cost += cost
                for c, t in dist:
                    lif_alloc[c] += t
                    if lif_alloc[c] >= self.LIF_T:
                        available.discard(c)

        for layer in self.unmapped_layers:
            tiles_needed = self._calculate_lif_tiles_for_unmapped_layer(layer)
            total_tiles += tiles_needed

            dist, cost = self.find_optimal_lif_placement(layer, layer, tiles_needed, available, lif_alloc)

            if not dist and tiles_needed > 0:
                feasible = False
                lif_distributions.append([])
            else:
                lif_distributions.append(dist)
                total_cost += cost
                for c, t in dist:
                    lif_alloc[c] += t
                    if lif_alloc[c] >= self.LIF_T:
                        available.discard(c)

        solution.objectives = (total_tiles, total_cost)
        solution.lif_distributions = lif_distributions
        solution.feasible = feasible

        return total_tiles, total_cost, feasible

    def generate_random_solution(self) -> Solution:
        """Generate random solution for MAPPED layers only."""
        if self.num_mapped_groups == 1:
            return Solution(breakpoints=[self.mapped_layers])

        possible = list(range(1, self.mapped_layers))
        k = min(self.num_mapped_groups - 1, len(possible))
        breakpoints = sorted(random.sample(possible, k)) + [self.mapped_layers]

        return Solution(breakpoints=breakpoints)

    def perturb_solution(self, solution: Solution) -> Solution:
        """Perturb solution (only affects mapped layer grouping)."""
        new_bp = solution.breakpoints.copy()
        internal = new_bp[:-1]

        if not internal:
            self.last_perturbation_type = "NONE"
            return solution

        move = random.choice(['shift', 'swap', 'reset'])
        idx = random.randint(0, len(internal) - 1)

        min_val = internal[idx - 1] + 1 if idx > 0 else 1
        max_val = internal[idx + 1] - 1 if idx < len(internal) - 1 else self.mapped_layers - 1

        old_val = internal[idx]

        if move == 'shift':
            shift = random.choice([-2, -1, 1, 2])
            new_val = internal[idx] + shift
            if min_val <= new_val <= max_val:
                internal[idx] = new_val
            self.last_perturbation_type = f"SHIFT[{idx}]: {old_val}→{internal[idx]}"
        elif move == 'swap' and len(internal) >= 2 and idx < len(internal) - 1:
            max_val2 = internal[idx + 2] - 1 if idx + 2 < len(internal) else self.mapped_layers - 1
            if max_val2 - min_val >= 1:
                new_pos = sorted(random.sample(range(min_val, max_val2 + 1), 2))
                internal[idx], internal[idx + 1] = new_pos[0], new_pos[1]
            self.last_perturbation_type = f"SWAP[{idx},{idx+1}]"
        else:
            if min_val <= max_val:
                internal[idx] = random.randint(min_val, max_val)
            self.last_perturbation_type = f"RESET[{idx}]: {old_val}→{internal[idx]}"

        return Solution(breakpoints=sorted(internal) + [self.mapped_layers])

    def dominates(self, sol1: Solution, sol2: Solution) -> bool:
        better_in_one = False
        for o1, o2 in zip(sol1.objectives, sol2.objectives):
            if o1 > o2:
                return False
            if o1 < o2:
                better_in_one = True
        return better_in_one

    def get_domination_status(self, new_sol: Solution, archive: List[Solution]) -> Tuple[int, List[Solution]]:
        dominated_by_new = []
        for arch_sol in archive:
            if self.dominates(arch_sol, new_sol):
                return -1, []
            if self.dominates(new_sol, arch_sol):
                dominated_by_new.append(arch_sol)
        return (len(dominated_by_new), dominated_by_new) if dominated_by_new else (0, [])

    def update_archive(self, new_sol: Solution) -> bool:
        for arch_sol in self.archive:
            if self.dominates(arch_sol, new_sol):
                return False

        dominated = [s for s in self.archive if self.dominates(new_sol, s)]
        for sol in dominated:
            self.archive.remove(sol)

        self.archive.append(new_sol)
        self.stats['archive_updates'] += 1

        self._remove_dominated_from_archive()

        if len(self.archive) > self.archive_hard_limit:
            self._prune_archive()

        return True

    def _remove_dominated_from_archive(self):
        non_dominated = []
        for sol in self.archive:
            dominated = False
            for other in self.archive:
                if other is sol:
                    continue
                if self.dominates(other, sol):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(sol)
        self.archive = non_dominated

    def _prune_archive(self):
        while len(self.archive) > self.archive_size:
            distances = self._crowding_distance()
            min_sol = min([s for s in self.archive if distances[s] != float('inf')],
                         key=lambda s: distances[s], default=None)
            if min_sol:
                self.archive.remove(min_sol)
            else:
                self.archive.pop(random.randint(0, len(self.archive) - 1))

    def _crowding_distance(self) -> Dict[Solution, float]:
        if len(self.archive) <= 2:
            return {s: float('inf') for s in self.archive}
        distances = {s: 0.0 for s in self.archive}
        for obj_idx in range(2):
            sorted_arch = sorted(self.archive, key=lambda s: s.objectives[obj_idx])
            distances[sorted_arch[0]] = float('inf')
            distances[sorted_arch[-1]] = float('inf')
            obj_range = sorted_arch[-1].objectives[obj_idx] - sorted_arch[0].objectives[obj_idx]
            if obj_range == 0:
                continue
            for i in range(1, len(sorted_arch) - 1):
                distances[sorted_arch[i]] += (
                    sorted_arch[i + 1].objectives[obj_idx] -
                    sorted_arch[i - 1].objectives[obj_idx]
                ) / obj_range
        return distances

    def select_from_archive(self) -> Solution:
        if len(self.archive) <= 2:
            return random.choice(self.archive)
        candidates = random.sample(self.archive, min(3, len(self.archive)))
        distances = self._crowding_distance()
        return max(candidates, key=lambda s: distances.get(s, 0))

    def get_all_pareto_solutions(self) -> List[Dict]:
        """Get all TRUE Pareto-optimal solutions with complete information."""

        def is_dominated(sol, all_solutions):
            for other in all_solutions:
                if other is sol:
                    continue
                if (other.objectives[0] <= sol.objectives[0] and
                    other.objectives[1] <= sol.objectives[1] and
                    (other.objectives[0] < sol.objectives[0] or
                     other.objectives[1] < sol.objectives[1])):
                    return True
            return False

        pareto_solutions = [s for s in self.archive if not is_dominated(s, self.archive)]
        pareto_solutions.sort(key=lambda s: (s.objectives[0], s.objectives[1]))

        results = []
        for sol in pareto_solutions:
            results.append({
                'breakpoints': sol.breakpoints,
                'total_lif_tiles': sol.objectives[0],
                'total_cost': sol.objectives[1],
                'num_groups': self.num_groups,
                'num_mapped_groups': self.num_mapped_groups,
                'num_unmapped_groups': self.num_unmapped_groups,
                'groups': sol.groupings,
                'lif_distributions': sol.lif_distributions,
                'chiplet_mapping': sol.chiplet_mapping,
                'feasible': sol.feasible
            })

        return results

    def get_best_solutions(self, n: int = 5) -> List[Dict]:
        """Get top n solutions from the TRUE Pareto front."""
        pareto_solutions = self.get_all_pareto_solutions()

        if not pareto_solutions:
            return []

        obj1 = [s['total_lif_tiles'] for s in pareto_solutions]
        obj2 = [s['total_cost'] for s in pareto_solutions]

        min_obj1, max_obj1 = min(obj1), max(obj1)
        min_obj2, max_obj2 = min(obj2), max(obj2)

        r1 = max_obj1 - min_obj1 if max_obj1 != min_obj1 else 1
        r2 = max_obj2 - min_obj2 if max_obj2 != min_obj2 else 1

        scored = []
        for sol in pareto_solutions:
            norm1 = (sol['total_lif_tiles'] - min_obj1) / r1
            norm2 = (sol['total_cost'] - min_obj2) / r2
            score = 0.5 * norm1 + 0.5 * norm2
            sol['normalized_score'] = score
            scored.append((score, sol))

        scored.sort(key=lambda x: x[0])
        return [sol for _, sol in scored[:n]]

    def get_solution_by_preference(self, preference: str = 'balanced') -> Dict:
        """Get a single solution based on preference."""
        pareto_solutions = self.get_all_pareto_solutions()

        if not pareto_solutions:
            return None

        if preference == 'min_memory':
            return min(pareto_solutions, key=lambda s: s['total_lif_tiles'])
        elif preference == 'min_cost':
            return min(pareto_solutions, key=lambda s: s['total_cost'])
        else:
            best_solutions = self.get_best_solutions(1)
            return best_solutions[0] if best_solutions else None

    def format_output(self, solutions: Optional[List[Dict]] = None) -> List[List]:
        """Format output as [[layers, lif_dist], ...] for ALL groups."""
        if solutions is None:
            solutions = self.get_best_solutions(1)
        if not solutions:
            return []

        best = solutions[0]
        output = []
        for g in best['groups']:
            output.append([g['layers'], g['lif_distribution']])
        return output


def plot_pareto_front(optimizer, title="Pareto Front", figsize=(12, 8)):
    """Plot the Pareto front with feasibility coloring."""

    if not optimizer.archive:
        print("No solutions in archive.")
        return

    history_mem_valid, history_cost_valid = [], []
    history_mem_invalid, history_cost_invalid = [], []

    if hasattr(optimizer, 'logger') and optimizer.logger.iteration_log:
        for entry in optimizer.logger.iteration_log:
            m, c = entry['new_obj']
            if np.isfinite(m) and np.isfinite(c):
                if entry.get('new_feasible', True):
                    history_mem_valid.append(m)
                    history_cost_valid.append(c)
                else:
                    history_mem_invalid.append(m)
                    history_cost_invalid.append(c)

    archive_mem = [s.objectives[0] for s in optimizer.archive]
    archive_cost = [s.objectives[1] for s in optimizer.archive]
    archive_feasible = [s.feasible for s in optimizer.archive]

    archive_mem_valid = [m for m, f in zip(archive_mem, archive_feasible) if f]
    archive_cost_valid = [c for c, f in zip(archive_cost, archive_feasible) if f]
    archive_mem_invalid = [m for m, f in zip(archive_mem, archive_feasible) if not f]
    archive_cost_invalid = [c for c, f in zip(archive_cost, archive_feasible) if not f]

    pareto_indices = []
    for i, (m1, c1, f1) in enumerate(zip(archive_mem, archive_cost, archive_feasible)):
        if not f1:
            continue
        is_dominated = False
        for j, (m2, c2, f2) in enumerate(zip(archive_mem, archive_cost, archive_feasible)):
            if i == j or not f2:
                continue
            if (m2 <= m1 and c2 <= c1) and (m2 < m1 or c2 < c1):
                is_dominated = True
                break
        if not is_dominated:
            pareto_indices.append(i)

    pareto_mem = [archive_mem[i] for i in pareto_indices]
    pareto_cost = [archive_cost[i] for i in pareto_indices]

    if pareto_mem:
        sorted_pairs = sorted(zip(pareto_mem, pareto_cost))
        pareto_mem, pareto_cost = zip(*sorted_pairs)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)

    if history_mem_valid:
        ax.scatter(history_mem_valid, history_cost_valid, c='lightgreen', alpha=0.3,
                   s=15, label=f'Explored (Valid, n={len(history_mem_valid)})')
    if history_mem_invalid:
        ax.scatter(history_mem_invalid, history_cost_invalid, c='salmon', alpha=0.3,
                   s=15, marker='x', label=f'Explored (Invalid, n={len(history_mem_invalid)})')

    if archive_mem_valid:
        ax.scatter(archive_mem_valid, archive_cost_valid, c='darkgreen', s=80,
                   edgecolors='k', alpha=0.6, zorder=4,
                   label=f'Archive (Valid, n={len(archive_mem_valid)})')
    if archive_mem_invalid:
        ax.scatter(archive_mem_invalid, archive_cost_invalid, c='red', s=80,
                   edgecolors='k', alpha=0.6, marker='x', zorder=4,
                   label=f'Archive (Invalid, n={len(archive_mem_invalid)})')

    if pareto_mem:
        ax.scatter(pareto_mem, pareto_cost, c='blue', s=120, edgecolors='k',
                   linewidths=2, zorder=5, label=f'Pareto Front ({len(pareto_mem)})')
        ax.plot(pareto_mem, pareto_cost, c='blue', linestyle='--', alpha=0.6,
                linewidth=2, zorder=4)

    if pareto_mem:
        min_mem_idx = pareto_mem.index(min(pareto_mem))
        min_cost_idx = pareto_cost.index(min(pareto_cost))

        ax.scatter([pareto_mem[min_mem_idx]], [pareto_cost[min_mem_idx]],
                   c='gold', s=250, marker='*', zorder=6, edgecolors='k',
                   linewidths=2, label=f'Min Memory: {pareto_mem[min_mem_idx]:.0f}')
        ax.scatter([pareto_mem[min_cost_idx]], [pareto_cost[min_cost_idx]],
                   c='orangered', s=250, marker='*', zorder=6, edgecolors='k',
                   linewidths=2, label=f'Min Cost: {pareto_cost[min_cost_idx]:.2f}')

    ax.set_xlabel('Total LIF Tiles (Memory)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Total Communication Cost', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9)

    summary_text = (f"Total Explored: {len(history_mem_valid) + len(history_mem_invalid)}\n"
                    f"Valid: {len(history_mem_valid)} | Invalid: {len(history_mem_invalid)}\n"
                    f"Archive: {len(archive_mem)} | Pareto: {len(pareto_mem)}")
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')

    # plt.tight_layout()
    # plt.show()
    plt.close()

    return pareto_indices
#################################################
def define_networks():
    """Define all CNN and Transformer architectures"""
    # Format: (IFM_H, IFM_W, IFM_C, K_H, K_W, K_N, Pool, Stride)
    # VGG19 weights for a 224x224 input image
    weights_vgg19 = [
        # Block 1
        (224, 224, 3, 3, 3, 64, 0, 1),
        (224, 224, 64, 3, 3, 64, 1, 1),      # pool here -> 112x112
        # Block 2
        (112, 112, 64, 3, 3, 128, 0, 1),
        (112, 112, 128, 3, 3, 128, 1, 1),    # pool here -> 56x56
        # Block 3 (4 conv layers)
        (56, 56, 128, 3, 3, 256, 0, 1),
        (56, 56, 256, 3, 3, 256, 0, 1),
        (56, 56, 256, 3, 3, 256, 0, 1),      # Extra layer for VGG19
        (56, 56, 256, 3, 3, 256, 1, 1),      # pool here -> 28x28
        # Block 4 (4 conv layers)
        (28, 28, 256, 3, 3, 512, 0, 1),
        (28, 28, 512, 3, 3, 512, 0, 1),
        (28, 28, 512, 3, 3, 512, 0, 1),      # Extra layer for VGG19
        (28, 28, 512, 3, 3, 512, 1, 1),      # pool here -> 14x14
        # Block 5 (4 conv layers)
        (14, 14, 512, 3, 3, 512, 0, 1),
        (14, 14, 512, 3, 3, 512, 0, 1),
        (14, 14, 512, 3, 3, 512, 0, 1),      # Extra layer for VGG19
        (14, 14, 512, 3, 3, 512, 1, 1),      # pool here -> 7x7
        # Final classification layer
        (1, 1, 25088, 1, 1, 4096, 0, 1),
        (1, 1, 4096, 1, 1, 4096, 0, 1),
        (1, 1, 4096, 1, 1, 1000, 0, 1)
    ]

    # Format: (IFM_H, IFM_W, IFM_C, K_H, K_W, K_N, Pool, Stride)
    # ResNet-50 weights for a 224x224 input image
    weights_resnet50 = [
        # Block 1 (conv1)
        (224, 224, 3, 7, 7, 64, 1, 2),       # pool here (3x3, S=2) -> 56x56

        # Block 2 (conv2_x) - 3 blocks
        # Block 2a (IFM: 56x56x64 -> OFM: 56x56x256)
        (56, 56, 64, 1, 1, 64, 0, 1),
        (56, 56, 64, 3, 3, 64, 0, 1),
        (56, 56, 64, 1, 1, 256, 0, 1),
        # Block 2b (IFM: 56x56x256 -> OFM: 56x56x256)
        (56, 56, 256, 1, 1, 64, 0, 1),
        (56, 56, 64, 3, 3, 64, 0, 1),
        (56, 56, 64, 1, 1, 256, 0, 1),
        # Block 2c (IFM: 56x56x256 -> OFM: 56x56x256)
        (56, 56, 256, 1, 1, 64, 0, 1),
        (56, 56, 64, 3, 3, 64, 0, 1),
        (56, 56, 64, 1, 1, 256, 0, 1),

        # Block 3 (conv3_x) - 4 blocks
        # Block 3a (IFM: 56x56x256 -> OFM: 28x28x512)
        (56, 56, 256, 1, 1, 128, 0, 2),      # Stride=2 here
        (28, 28, 128, 3, 3, 128, 0, 1),
        (28, 28, 128, 1, 1, 512, 0, 1),
        # Block 3b (IFM: 28x28x512 -> OFM: 28x28x512)
        (28, 28, 512, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 128, 0, 1),
        (28, 28, 128, 1, 1, 512, 0, 1),
        # Block 3c (IFM: 28x28x512 -> OFM: 28x28x512)
        (28, 28, 512, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 128, 0, 1),
        (28, 28, 128, 1, 1, 512, 0, 1),
        # Block 3d (IFM: 28x28x512 -> OFM: 28x28x512)
        (28, 28, 512, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 128, 0, 1),
        (28, 28, 128, 1, 1, 512, 0, 1),

        # Block 4 (conv4_x) - 6 blocks
        # Block 4a (IFM: 28x28x512 -> OFM: 14x14x1024)
        (28, 28, 512, 1, 1, 256, 0, 2),      # Stride=2 here
        (14, 14, 256, 3, 3, 256, 0, 1),
        (14, 14, 256, 1, 1, 1024, 0, 1),
        # Block 4b (IFM: 14x14x1024 -> OFM: 14x14x1024)
        (14, 14, 1024, 1, 1, 256, 0, 1),
        (14, 14, 256, 3, 3, 256, 0, 1),
        (14, 14, 256, 1, 1, 1024, 0, 1),
        # Block 4c (IFM: 14x14x1024 -> OFM: 14x14x1024)
        (14, 14, 1024, 1, 1, 256, 0, 1),
        (14, 14, 256, 3, 3, 256, 0, 1),
        (14, 14, 256, 1, 1, 1024, 0, 1),
        # Block 4d (IFM: 14x14x1024 -> OFM: 14x14x1024)
        (14, 14, 1024, 1, 1, 256, 0, 1),
        (14, 14, 256, 3, 3, 256, 0, 1),
        (14, 14, 256, 1, 1, 1024, 0, 1),
        # Block 4e (IFM: 14x14x1024 -> OFM: 14x14x1024)
        (14, 14, 1024, 1, 1, 256, 0, 1),
        (14, 14, 256, 3, 3, 256, 0, 1),
        (14, 14, 256, 1, 1, 1024, 0, 1),
        # Block 4f (IFM: 14x14x1024 -> OFM: 14x14x1024)
        (14, 14, 1024, 1, 1, 256, 0, 1),
        (14, 14, 256, 3, 3, 256, 0, 1),
        (14, 14, 256, 1, 1, 1024, 0, 1),

        # Block 5 (conv5_x) - 3 blocks
        # Block 5a (IFM: 14x14x1024 -> OFM: 7x7x2048)
        (14, 14, 1024, 1, 1, 512, 0, 2),     # Stride=2 here
        (7, 7, 512, 3, 3, 512, 0, 1),
        (7, 7, 512, 1, 1, 2048, 0, 1),
        # Block 5b (IFM: 7x7x2048 -> OFM: 7x7x2048)
        (7, 7, 2048, 1, 1, 512, 0, 1),
        (7, 7, 512, 3, 3, 512, 0, 1),
        (7, 7, 512, 1, 1, 2048, 0, 1),
        # Block 5c (IFM: 7x7x2048 -> OFM: 7x7x2048)
        (7, 7, 2048, 1, 1, 512, 0, 1),
        (7, 7, 512, 3, 3, 512, 0, 1),
        (7, 7, 512, 1, 1, 2048, 0, 1),

        # Final classification layer
        (1, 1, 100352, 1, 1, 1000, 0, 1)
    ]

    # Format: (IFM_H, IFM_W, IFM_C, K_H, K_W, K_N, Pool, Stride)
    weights_densenet121 = [
        # Initial Convolution + Pooling
        (224, 224, 3, 7, 7, 64, 1, 2),      # Initial conv: 224x224x3 -> 112x112x64
        (112, 112, 64, 3, 3, 64, 1, 2),     # MaxPool: 112x112x64 -> 56x56x64

        # Dense Block 1 (6 layers)
        (56, 56, 64, 1, 1, 128, 0, 1),
        (56, 56, 128, 3, 3, 32+64, 0, 1),
        (56, 56, 32+64, 1, 1, 128, 0, 1),
        (56, 56, 128, 3, 3, 32+96, 0, 1),
        (56, 56, 32+96, 1, 1, 128, 0, 1),
        (56, 56, 128, 3, 3, 32+128, 0, 1),
        (56, 56, 32+128, 1, 1, 128, 0, 1),
        (56, 56, 128, 3, 3, 32+160, 0, 1),
        (56, 56, 32+160, 1, 1, 128, 0, 1),
        (56, 56, 128, 3, 3, 32+192, 0, 1),
        (56, 56, 32+192, 1, 1, 128, 0, 1),
        (56, 56, 128, 3, 3, 32+224, 0, 1),

        # Transition Layer 1
        (56, 56, 256, 1, 1, 128, 0, 1),     # 1x1 conv: 256->128
        (56, 56, 128, 2, 2, 128, 1, 2),     # AvgPool: 56x56x128 -> 28x28x128

        # Dense Block 2 (12 layers)
        (28, 28, 128, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 32+128, 0, 1),
        (28, 28, 32+128, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 32+160, 0, 1),
        (28, 28, 32+160, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 32+192, 0, 1),
        (28, 28, 32+192, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 32+224, 0, 1),
        (28, 28, 32+224, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 32+256, 0, 1),
        (28, 28, 32+256, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 32+288, 0, 1),
        (28, 28, 32+288, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 32+320, 0, 1),
        (28, 28, 32+320, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 32+352, 0, 1),
        (28, 28, 32+352, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 32+384, 0, 1),
        (28, 28, 32+384, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 32+416, 0, 1),
        (28, 28, 32+416, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 32+448, 0, 1),
        (28, 28, 32+448, 1, 1, 128, 0, 1),
        (28, 28, 128, 3, 3, 32+480, 0, 1),

        # Transition Layer 2
        (28, 28, 512, 1, 1, 256, 0, 1),     # 1x1 conv: 512->256
        (28, 28, 256, 2, 2, 256, 1, 2),     # AvgPool: 28x28x256 -> 14x14x256

        # Dense Block 3 (24 layers)
        (14, 14, 256, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+256, 0, 1),
        (14, 14, 32+256, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+288, 0, 1),
        (14, 14, 32+288, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+320, 0, 1),
        (14, 14, 32+320, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+352, 0, 1),
        (14, 14, 32+352, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+384, 0, 1),
        (14, 14, 32+384, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+416, 0, 1),
        (14, 14, 32+416, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+448, 0, 1),
        (14, 14, 32+448, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+480, 0, 1),
        (14, 14, 32+480, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+512, 0, 1),
        (14, 14, 32+512, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+544, 0, 1),
        (14, 14, 32+544, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+576, 0, 1),
        (14, 14, 32+576, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+608, 0, 1),
        (14, 14, 32+608, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+640, 0, 1),
        (14, 14, 32+640, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+672, 0, 1),
        (14, 14, 32+672, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+704, 0, 1),
        (14, 14, 32+704, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+736, 0, 1),
        (14, 14, 32+736, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+768, 0, 1),
        (14, 14, 32+768, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+800, 0, 1),
        (14, 14, 32+800, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+832, 0, 1),
        (14, 14, 32+832, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+864, 0, 1),
        (14, 14, 32+864, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+896, 0, 1),
        (14, 14, 32+896, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+928, 0, 1),
        (14, 14, 32+928, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+960, 0, 1),
        (14, 14, 32+960, 1, 1, 128, 0, 1),
        (14, 14, 128, 3, 3, 32+992, 0, 1),

        # Transition Layer 3
        (14, 14, 1024, 1, 1, 512, 0, 1),    # 1x1 conv: 1024->512
        (14, 14, 512, 2, 2, 512, 1, 2),     # AvgPool: 14x14x512 -> 7x7x512

        # Dense Block 4 (16 layers)
        (7, 7, 512, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+512, 0, 1),
        (7, 7, 32+512, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+544, 0, 1),
        (7, 7, 32+544, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+576, 0, 1),
        (7, 7, 32+576, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+608, 0, 1),
        (7, 7, 32+608, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+640, 0, 1),
        (7, 7, 32+640, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+672, 0, 1),
        (7, 7, 32+672, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+704, 0, 1),
        (7, 7, 32+704, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+736, 0, 1),
        (7, 7, 32+736, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+768, 0, 1),
        (7, 7, 32+768, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+800, 0, 1),
        (7, 7, 32+800, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+832, 0, 1),
        (7, 7, 32+832, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+864, 0, 1),
        (7, 7, 32+864, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+896, 0, 1),
        (7, 7, 32+896, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+928, 0, 1),
        (7, 7, 32+928, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+960, 0, 1),
        (7, 7, 32+960, 1, 1, 128, 0, 1),
        (7, 7, 128, 3, 3, 32+992, 0, 1),

        # Final Layers
        (7, 7, 1024, 7, 7, 1024, 0, 1),    # Global Average Pooling
        (1, 1, 1024, 1, 1, 1000, 0, 1),    # Classification Layer (1000 classes)
    ]

    # Transformer Decoder Architecture
    N = 197

    weights_transformer_decoder = [
        # 1. W_QKV : Combined Query, Key, Value projection
        (768, N, 768, 1, 1, 768*3, -1, 1),

        # 2. W_O : Output projection
        (N, 768, 768, 1, 1, 768, -1, 1),

        # 3. MLP1 : Feed-forward expansion
        (768, N, 768, 1, 1, 768*4, -1, 1),

        # 4. MLP2 : Feed-forward contraction
        (768*4, N, 768*4, 1, 1, 768, -1, 1)
    ]

    weights_vit = [weights_transformer_decoder * 28][0]
    weights_LVVITB = [weights_transformer_decoder * 10][0]
    weights_SPIKFORMER = [weights_transformer_decoder * 16][0]

    return {
        'vgg19': weights_vgg19,
        'resnet50': weights_resnet50,
        'densenet121': weights_densenet121,
        'LVVITB': weights_LVVITB,
        'VIT': weights_vit,
        'SPIKFORMER': weights_SPIKFORMER
    }

def run_global_lif_case(
    network,
    NPE,
    NT,
    xbar_size,
    SRAM_KB_per_tile,
    XBAR_bits_per_cell,
    Vmem_res,
    Timestep,
    NoC_buswidth,
    NoI_buswidth,
    NoC_wire_length,
    NoI_wire_length,
    NoC_Freq,
    NoI_Freq,
    TOPS,
    ENERGY_PER_MAC_pj,
    tile_area,
    chiplet_area,
    mesh_rows,
    mesh_cols,
    LIF_T_list,
    percent_keep,
    min_traffic,
    eta,
    DRAM_BW,
    run_optimized=True,
):
    """
    Run global LIF module case - sweep LIF_T and system sizes
    
    Parameters:
    -----------
    run_optimized : bool, default=True
        If True, runs both regular and optimized mappings.
        If False, runs only regular mapping.
    """
    import copy
    import numpy as np
    import pandas as pd

    # Update BookSim simulator parameters
    update_param_booksim('Chiplet_level_config', 'channel_width', NoC_buswidth)
    update_param_booksim('System_level_config', 'channel_width', NoI_buswidth)
    update_param_booksim('techfile_chiplet.txt', 'wire_length', NoC_wire_length)
    update_param_booksim('techfile_system.txt', 'wire_length', NoI_wire_length)

    # Calculate NoC cycle times
    NoC_cycle_time = 1e3 / NoC_Freq
    NoI_cycle_time = 1e3 / NoI_Freq

    # Initial mapper setup
    mapper_temp = SNNMapper(
        weights=network,
        layer_groups=[],
        NPE=NPE,
        NT=NT - max(LIF_T_list),
        X=xbar_size,
        bits_per_cell=XBAR_bits_per_cell,
        P=100,
        Vmem_res=Vmem_res,
        Timestep=Timestep,
        NoC_buswidth=NoC_buswidth,
        NoI_buswidth=NoI_buswidth,
        allow_break_columns=True,
        include_chiplets=False,
    )

    (
        mapper_temp.tunable_params,
        mapper_temp.xbars,
        mapper_temp.IFMS,
        mapper_temp.OFMS,
        mapper_temp.TOPS,
        mapper_temp.MEMS,
    ) = mapper_temp._calc_tunable_params()
  
    mapper_temp.layer_output_sizes = dict(
        zip(range(1, len(mapper_temp.OFMS) + 1), mapper_temp.OFMS)
    )
    mapper_temp.chiplet_data = mapper_temp._generate_chiplet_mapping()

    # Store results for all LIF_T values
    global_results_by_lif_regular = {}
    global_results_by_lif_optimized = {}
    LIF_T_min = int(np.ceil(np.ceil(np.array(mapper_temp.MEMS) / \
                (SRAM_KB_per_tile * 1024)).max() / (mesh_rows * mesh_cols)))
    for LIF_T_val in LIF_T_list:
        if LIF_T_val < LIF_T_min:
            print(f"Skipping LIF_T = {LIF_T_val} as it is below minimum required {LIF_T_min}.")
            continue
        print(f"Processing LIF_T = {LIF_T_val}...")

        noc_rows = int(np.sqrt(NT))
        NoC_mesh_layout = [
            [(r * noc_rows + c) for c in range(noc_rows)] for r in range(noc_rows)
        ]

        # ====== REGULAR MAPPING ======
        mapper = SNNMapper(
            weights=network,
            layer_groups=[],
            NPE=NPE,
            NT=NT - LIF_T_val,
            X=xbar_size,
            P=100,
            Vmem_res=Vmem_res,
            Timestep=Timestep,
            NoC_buswidth=NoC_buswidth,
            NoI_buswidth=NoI_buswidth,
            bits_per_cell=XBAR_bits_per_cell,
            allow_break_columns=True,
            include_chiplets=False,
            max_chiplets=mesh_cols * mesh_rows - 1,
        )

        (
            mapper.tunable_params,
            mapper.xbars,
            mapper.IFMS,
            mapper.OFMS,
            mapper.TOPS,
            mapper.MEMS,
        ) = mapper._calc_tunable_params()

        mapper.layer_output_sizes = dict(
            zip(range(1, len(mapper.OFMS) + 1), mapper.OFMS)
        )
        mapper.chiplet_data = mapper._generate_chiplet_mapping()
        if not mapper.chiplet_data['main_chiplets']:
            print("No valid chiplet mapping found for regular mapping.")
            continue
        chiplets_used = len(mapper.chiplet_data['main_chiplets'])
        
        max_chiplets_used=chiplets_used
        print(f"Chiplets used in regular mapping: {chiplets_used}/{mesh_rows * mesh_cols}")
        LIF_needed = (
            np.ceil(np.array(mapper.MEMS) / (SRAM_KB_per_tile * 1024))
            .astype(int)
            .max()
        )

        LIF_chiplet = {
            'Layers_filled': [],
            'Crossbars_filled_respective_layer': [],
            'Crossbars_remaining_respective_layer': [],
            'Layer_tile_distribution': {
                'LIF0': {i: NPE for i in range(LIF_needed)}
            },
            'Empty_crossbars': NT * NPE,
        }

        EMPTY = {
            'Layers_filled': [],
            'Crossbars_filled_respective_layer': [],
            'Crossbars_remaining_respective_layer': [],
            'Layer_tile_distribution': {},
            'Empty_crossbars': NT * NPE,
        }

        global_grp = [
            {
                'start_layer': 1,
                'end_layer': len(mapper.weights),
                'total_lif_tiles': LIF_needed,
                'lif_distribution': [(chiplets_used, LIF_needed)],
                'cost': 0.0,
            }
        ]

        cc = copy.deepcopy(mapper.chiplet_data['main_chiplets'])
        layer_ofms = copy.deepcopy(mapper_temp.layer_output_sizes)
        layer_ofms = {k: v / 1e4 for k, v in layer_ofms.items()}

        result, cost = optimize_multi_group_lif_placement(
            chiplet_data=cc,
            layer_weights=layer_ofms,
            groups=[[1, len(mapper.weights)]],
            lif_needed=[LIF_needed],
            lif_capacity=LIF_needed,
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
        )

        global_grp_optimized = [
            {
                'start_layer': 1,
                'end_layer': len(mapper.weights),
                'total_lif_tiles': LIF_needed,
                'lif_distribution': [(result[0][0][0], LIF_needed)],
                'cost': 0.0,
            }
        ]

        # Initialize accumulators
        all_remap_NoI_latencies = []
        all_remap_NoI_powers = []
        all_remap_NoC_latencies = []
        all_remap_NoC_powers = []
        all_remap_DRAM_latencies = []

        NoI_mesh_layout = [
            [(r * mesh_cols + c) for c in range(mesh_cols)]
            for r in range(mesh_rows)
        ]
        NoI_mesh_layout_remap_from_LIFDRAM = (
            [[r * mesh_cols + c for c in range(mesh_cols)] for r in range(mesh_rows)]
            + [[mesh_rows * mesh_cols]]
        )

        all_layers_noi_latencies = {}
        all_layers_noc_latencies = {}
        all_layers_dram_latencies = {}

        mapped_noi_powers_raw = []
        mapped_noi_latencies_ms = []
        mapped_noc_powers_raw = []
        mapped_noc_latencies_ms = []

        # ============================================
        # FIRST MAPPING - Process mapped layers
        # ============================================
        nn = mesh_cols * mesh_rows - len(cc) - 1
        cc += [LIF_chiplet]
        cc += [EMPTY] * nn + [LIF_chiplet]

        mapped_layers = sorted(
            {
                layer
                for chiplet in mapper.chiplet_data['main_chiplets']
                for layer in chiplet.get('Layers_filled', [])
            }
        )
        print(f"Mapped layers in first mapping: [{mapped_layers[0]}-{mapped_layers[-1]}]")
        first_map_global_NoI_latency_ms = 0
        first_map_global_NoI_power_raw = 0
        first_map_global_NoC_latency_ms = 0
        first_map_global_NoC_power_raw = 0

        for layer_id in mapped_layers:
            all_traffic = []
            traffic_dict = get_layer_traffic(
                layer_id=layer_id,
                chiplet_data=cc,
                groupings=global_grp,
                weights=mapper.weights,
                tunable_params=mapper.tunable_params,
                xbars=mapper.xbars,
                X=mapper.X,
                Vmem_res=mapper.Vmem_res,
                Timestep=Timestep,
                NoC_buswidth=1,
                lif_tiles_per_layer=np.ceil(
                    np.array(mapper.MEMS) / (SRAM_KB_per_tile * 1024)
                ).astype(int),
                SRAM_KB_per_tile=SRAM_KB_per_tile,
                acc_enabled=False,
            )
            all_traffic.extend(
                [traffic_dict["output"], traffic_dict["input"]]
            )

            tt, M_sys_df = create_system_matrix_from_edges(
                all_traffic, mesh_rows=mesh_rows, mesh_cols=mesh_cols, count_diagonal=True
            )
            M_sys_df = np.ceil(M_sys_df / NoI_buswidth).astype(int)
            M_sys_df, _ = split_top_rest(M_sys_df, percent=percent_keep)

            _, _, system_scaled_df, system_scaling_factor = (
                scale_traffic_matrices(
                    system_matrix=M_sys_df,
                    chiplet_matrices=[],
                    minimum_traffic=min_traffic,
                )
            )

            noi_latency_raw, noi_power_raw = run_booksim_NoI(
                system_scaled_df.values, NoI_mesh_layout
            )
            noI_latency_ms = noi_latency_raw * system_scaling_factor * NoI_cycle_time
            first_map_global_NoI_latency_ms += noI_latency_ms
            first_map_global_NoI_power_raw += noi_power_raw
            mapped_noi_powers_raw.append(noi_power_raw)
            mapped_noi_latencies_ms.append(noI_latency_ms)

            chiplet_dfs = []
            for c in range(mesh_cols * mesh_rows):
                M_chip_df = np.ceil(
                    create_tile_matrix_for_chiplet(
                        all_traffic, c, NT, include_chiplets=False
                    )[1]
                    / NoC_buswidth
                ).astype(int)
                chiplet_dfs.append(M_chip_df)

            M_chiplet_dfs = [
                split_top_rest(df, percent=percent_keep)[0] for df in chiplet_dfs
            ]
            (
                chiplet_scaled_dfs,
                chiplet_scaling_factors,
                _,
                _,
            ) = scale_traffic_matrices(
                system_matrix=[], chiplet_matrices=M_chiplet_dfs, minimum_traffic=min_traffic
            )

            NoC_mesh_layouts = [NoC_mesh_layout] * len(cc)
            noc_latencies_raw, noc_power_raw = run_booksim_NoC(
                chiplet_scaled_dfs, NoC_mesh_layouts
            )
            noc_latencies_ms = (
                (np.array(noc_latencies_raw) * np.array(chiplet_scaling_factors) * NoC_cycle_time)
                .sum()
            )
            noc_power_sum = np.array(noc_power_raw).sum()
            first_map_global_NoC_latency_ms += noc_latencies_ms
            first_map_global_NoC_power_raw += noc_power_sum
            mapped_noc_powers_raw.append(noc_power_sum)
            mapped_noc_latencies_ms.append(noc_latencies_ms)

            all_layers_noi_latencies[layer_id] = noI_latency_ms
            all_layers_noc_latencies[layer_id] = noc_latencies_ms
            all_layers_dram_latencies[layer_id] = 0.0

        # ============================================
        # REMAP BLOCK - Process unmapped layers
        # ============================================
        lif_to_DRAM_packets_sum = 0
        unmapped_layers = []
        for i, each_remap in enumerate(mapper.chiplet_data['unmapped_layers']):
            if max_chiplets_used<len(each_remap):
                max_chiplets_used=len(each_remap)
            print(f"Chiplets used in remapping: {len(each_remap)}/{mesh_rows * mesh_cols}")
            unmapped_layers = sorted(
                {
                    d['Layers_filled'][0]
                    for group in mapper.chiplet_data['unmapped_layers']
                    for d in group
                }
            )
            current_layer_id = sorted(
                {
                    layer
                    for chiplet in each_remap
                    for layer in chiplet.get('Layers_filled', [])
                }
            )[0]

            cc_to_lif = copy.deepcopy(each_remap)
            cc_to_lif += [EMPTY] * (mesh_cols * mesh_rows - len(cc_to_lif) - 1)
            cc_to_lif += [LIF_chiplet]

            # ============================================
            # PHASE 1: TO LIF
            # ============================================
            remap_mapped_layers = sorted(
                {
                    layer
                    for chiplet in each_remap
                    for layer in chiplet.get('Layers_filled', [])
                }
            )
            remap_NoI_latency_ms_to_LIF = 0
            remap_NoI_power_raw_to_LIF = 0
            dram_packets_to_lif = 0

            for layer_id in remap_mapped_layers:
                all_traffic = []
                traffic_dict = get_layer_traffic(
                    layer_id=layer_id,
                    chiplet_data=cc_to_lif,
                    groupings=global_grp,
                    weights=mapper.weights,
                    tunable_params=mapper.tunable_params,
                    xbars=mapper.xbars,
                    X=mapper.X,
                    Vmem_res=mapper.Vmem_res,
                    Timestep=Timestep,
                    NoC_buswidth=1,
                    lif_tiles_per_layer=np.ceil(
                        np.array(mapper.MEMS) / (SRAM_KB_per_tile * 1024)
                    ).astype(int),
                    SRAM_KB_per_tile=SRAM_KB_per_tile,
                    acc_enabled=False,
                )
                all_traffic.extend(
                    [traffic_dict["output"], traffic_dict["input"]]
                )

                _, M_sys_df_with_dram = create_system_matrix_from_edges(
                    all_traffic,
                    mesh_rows=mesh_rows + 1,
                    mesh_cols=mesh_cols,
                    count_diagonal=True,
                )
                layer_dram_packets_to = (
                    M_sys_df_with_dram.iloc[-1, :].sum()
                    + M_sys_df_with_dram.iloc[:, -1].sum()
                )
                dram_packets_to_lif += layer_dram_packets_to

                _, M_sys_df = create_system_matrix_from_edges(
                    all_traffic,
                    mesh_rows=mesh_rows,
                    mesh_cols=mesh_cols,
                    count_diagonal=True,
                )
                M_sys_df = np.ceil(M_sys_df / NoI_buswidth).astype(int)
                M_sys_df, _ = split_top_rest(M_sys_df, percent=percent_keep)

                _, _, system_scaled_df, system_scaling_factor = (
                    scale_traffic_matrices(
                        system_matrix=M_sys_df,
                        chiplet_matrices=[],
                        minimum_traffic=min_traffic,
                    )
                )

                noi_latency_raw, noi_power_raw = run_booksim_NoI(
                    system_scaled_df.values, NoI_mesh_layout
                )
                noI_latency_ms = (
                    noi_latency_raw * system_scaling_factor * NoI_cycle_time
                )
                remap_NoI_latency_ms_to_LIF += noI_latency_ms
                remap_NoI_power_raw_to_LIF += noi_power_raw

            DRAM_write_latency_ms = (
                (dram_packets_to_lif * NoI_buswidth / 8)
                / (eta * DRAM_BW * 1e9)
            ) * 1e3

            # ============================================
            # PHASE 2: FROM LIF
            # ============================================
            remap_NoI_latency_ms_from_LIF = 0
            remap_NoI_power_raw_from_LIF = 0
            DRAM_read_latency_ms = 0
            dram_packets_from_lif = 0

            if i + 1 < len(mapper.chiplet_data['unmapped_layers']):
                xx = copy.deepcopy(
                    mapper.chiplet_data['unmapped_layers'][i + 1]
                )
                dd = [LIF_chiplet] + [EMPTY] * (
                    mesh_cols * mesh_rows - len(xx)
                ) + xx

                next_remap_mapped_layers = sorted(
                    {
                        layer
                        for chiplet in xx
                        for layer in chiplet.get('Layers_filled', [])
                    }
                )

                for layer_id in next_remap_mapped_layers:
                    all_traffic = []
                    traffic_dict = get_layer_traffic(
                        layer_id=layer_id,
                        chiplet_data=dd,
                        groupings=global_grp,
                        weights=mapper.weights,
                        tunable_params=mapper.tunable_params,
                        xbars=mapper.xbars,
                        X=mapper.X,
                        Vmem_res=mapper.Vmem_res,
                        Timestep=Timestep,
                        NoC_buswidth=1,
                        lif_tiles_per_layer=np.ceil(
                            np.array(mapper.MEMS)
                            / (SRAM_KB_per_tile * 1024)
                        ).astype(int),
                        SRAM_KB_per_tile=SRAM_KB_per_tile,
                        acc_enabled=False,
                    )
                    all_traffic.extend(
                        [traffic_dict["output"], traffic_dict["input"]]
                    )

                    _, M_sys_df = create_system_matrix_from_edges(
                        all_traffic,
                        mesh_rows=mesh_rows + 1,
                        mesh_cols=mesh_cols,
                        count_diagonal=True,
                    )
                    layer_dram_packets = (
                        M_sys_df.iloc[-1, :].sum()
                        + M_sys_df.iloc[:, -1].sum()
                    )
                    lif_to_DRAM_packets_sum += layer_dram_packets
                    dram_packets_from_lif += layer_dram_packets

                    M_sys_df = M_sys_df.iloc[
                        : -(mesh_cols - 1), : -(mesh_cols - 1)
                    ]
                    M_sys_df = np.ceil(M_sys_df / NoI_buswidth).astype(int)
                    M_sys_df, _ = split_top_rest(M_sys_df, percent=percent_keep)

                    _, _, system_scaled_df, system_scaling_factor = (
                        scale_traffic_matrices(
                            system_matrix=M_sys_df,
                            chiplet_matrices=[],
                            minimum_traffic=min_traffic,
                        )
                    )

                    noi_latency_raw, noi_power_raw = run_booksim_NoI(
                        system_scaled_df.values,
                        NoI_mesh_layout_remap_from_LIFDRAM,
                    )
                    noI_latency_ms = (
                        noi_latency_raw
                        * system_scaling_factor
                        * NoI_cycle_time
                    )
                    remap_NoI_latency_ms_from_LIF += noI_latency_ms
                    remap_NoI_power_raw_from_LIF += noi_power_raw

                DRAM_read_latency_ms = (
                    (dram_packets_from_lif * NoI_buswidth / 8)
                    / (eta * DRAM_BW * 1e9)
                ) * 1e3

            # ============================================
            # NoC Latency Calculation
            # ============================================
            all_traffic_noc = []
            for layer_id in remap_mapped_layers:
                traffic_dict_noc = get_layer_traffic(
                    layer_id=layer_id,
                    chiplet_data=cc_to_lif,
                    groupings=global_grp,
                    weights=mapper.weights,
                    tunable_params=mapper.tunable_params,
                    xbars=mapper.xbars,
                    X=mapper.X,
                    Vmem_res=mapper.Vmem_res,
                    Timestep=Timestep,
                    NoC_buswidth=1,
                    lif_tiles_per_layer=np.ceil(
                        np.array(mapper.MEMS) / (SRAM_KB_per_tile * 1024)
                    ).astype(int),
                    SRAM_KB_per_tile=SRAM_KB_per_tile,
                    acc_enabled=False,
                )
                all_traffic_noc.extend(
                    [traffic_dict_noc["output"], traffic_dict_noc["input"]]
                )

            chiplet_dfs = []
            for c in range(mesh_cols * mesh_rows):
                M_chip_df = np.ceil(
                    create_tile_matrix_for_chiplet(
                        all_traffic_noc, c, NT, include_chiplets=False
                    )[1]
                    / NoC_buswidth
                ).astype(int)
                chiplet_dfs.append(M_chip_df)

            M_chiplet_dfs = [
                split_top_rest(df, percent=percent_keep)[0]
                for df in chiplet_dfs
            ]
            (
                chiplet_scaled_dfs,
                chiplet_scaling_factors,
                _,
                _,
            ) = scale_traffic_matrices(
                system_matrix=[],
                chiplet_matrices=M_chiplet_dfs,
                minimum_traffic=min_traffic,
            )

            NoC_mesh_layouts = [NoC_mesh_layout] * len(cc_to_lif)
            noc_latencies_raw, noc_power_raw = run_booksim_NoC(
                chiplet_scaled_dfs, NoC_mesh_layouts
            )
            noc_latencies_ms = (
                (
                    np.array(noc_latencies_raw)
                    * np.array(chiplet_scaling_factors)
                    * NoC_cycle_time
                )
                .sum()
            )
            noc_power_sum = np.array(noc_power_raw).sum()

            # ============================================
            # ACCUMULATE LATENCIES
            # ============================================
            total_remap_noi_latency = (
                remap_NoI_latency_ms_to_LIF
                + remap_NoI_latency_ms_from_LIF
            )
            total_remap_dram_latency = (
                DRAM_write_latency_ms + DRAM_read_latency_ms
            )
            total_remap_dram_packets = (
                dram_packets_to_lif + dram_packets_from_lif
            )

            layer_param_dram_latency_ms = (
                (mapper.tunable_params[current_layer_id - 1] / 8)
                / (eta * DRAM_BW * 1e9)
            ) * 1e3

            layer_lif_dram_latency_ms = (
                (
                    2
                    * total_remap_dram_packets
                    * NoI_buswidth
                    / 8
                )
                / (eta * DRAM_BW * 1e9)
            ) * 1e3

            total_layer_dram_latency_ms = (
                total_remap_dram_latency
                + layer_param_dram_latency_ms
                + layer_lif_dram_latency_ms
            )

            all_remap_NoI_latencies.append(total_remap_noi_latency)
            all_remap_NoI_powers.append(
                remap_NoI_power_raw_to_LIF
                + remap_NoI_power_raw_from_LIF
            )
            all_remap_NoC_latencies.append(noc_latencies_ms)
            all_remap_NoC_powers.append(noc_power_sum)
            all_remap_DRAM_latencies.append(total_remap_dram_latency)

            all_layers_noi_latencies[current_layer_id] = (
                total_remap_noi_latency
            )
            all_layers_noc_latencies[current_layer_id] = noc_latencies_ms
            all_layers_dram_latencies[current_layer_id] = (
                total_layer_dram_latency_ms
            )

        # ============================================
        # COMPUTATION LATENCY AND ENERGY PER LAYER
        # ============================================
        comp_latency_per_layer_s = (
            np.array(mapper.TOPS) * Timestep / (TOPS * 8)
        )
        comp_latency_per_layer_ms = comp_latency_per_layer_s * 1e3
        comp_energy_per_layer_J = (
            np.array(mapper.TOPS) * Timestep * ENERGY_PER_MAC_pj
        )
        total_comp_latency_ms = comp_latency_per_layer_ms.sum()
        total_comp_energy_J = comp_energy_per_layer_J.sum()

        # ============================================
        # END-TO-END LATENCY CALCULATION
        # ============================================
        mapped_layers_total_time = 0.0
        for layer_id in mapped_layers:
            layer_time = (
                all_layers_noi_latencies[layer_id]
                + all_layers_noc_latencies[layer_id]
                + comp_latency_per_layer_ms[layer_id - 1]
            )
            mapped_layers_total_time += layer_time

        end_to_end_latency = mapped_layers_total_time
        overlap_penalties = []

        if len(unmapped_layers) > 0:
            for remap_idx, unmapped_layer_id in enumerate(unmapped_layers):
                layer_processing_time = (
                    all_layers_noi_latencies[unmapped_layer_id]
                    + all_layers_noc_latencies[unmapped_layer_id]
                    + comp_latency_per_layer_ms[unmapped_layer_id - 1]
                )
                layer_total_dram = all_layers_dram_latencies[
                    unmapped_layer_id
                ]

                if remap_idx == 0:
                    available_time = mapped_layers_total_time
                else:
                    prev_layer_id = unmapped_layers[remap_idx - 1]
                    available_time = (
                        all_layers_noi_latencies[prev_layer_id]
                        + all_layers_noc_latencies[prev_layer_id]
                        + comp_latency_per_layer_ms[prev_layer_id - 1]
                    )

                overlap_penalty = max(0, layer_total_dram - available_time)
                overlap_penalties.append(overlap_penalty)
                end_to_end_latency += layer_processing_time + overlap_penalty

        # ============================================
        # CALCULATE TOTAL COMM LATENCY AND POWER/ENERGY
        # ============================================
        total_noi_latency = (
            first_map_global_NoI_latency_ms
            + sum(all_remap_NoI_latencies)
        )
        total_noc_latency = (
            first_map_global_NoC_latency_ms
            + sum(all_remap_NoC_latencies)
        )
        total_comm_latency_ms = total_noi_latency + total_noc_latency
        total_noi_power_raw = (
            first_map_global_NoI_power_raw
            + sum(all_remap_NoI_powers)
        )
        total_noc_power_raw = (
            first_map_global_NoC_power_raw
            + sum(all_remap_NoC_powers)
        )
        comm_power_W = total_noi_power_raw + total_noc_power_raw

        noi_energy_mJ_mapped = float(
            np.sum(
                np.array(mapped_noi_powers_raw)
                * np.array(mapped_noi_latencies_ms)
            )
        )
        noc_energy_mJ_mapped = float(
            np.sum(
                np.array(mapped_noc_powers_raw)
                * np.array(mapped_noc_latencies_ms)
            )
        )

        if len(all_remap_NoI_powers) > 0:
            noi_energy_mJ_remap = float(
                np.sum(
                    np.array(all_remap_NoI_powers)
                    * np.array(all_remap_NoI_latencies)
                )
            )
            noc_energy_mJ_remap = float(
                np.sum(
                    np.array(all_remap_NoC_powers)
                    * np.array(all_remap_NoC_latencies)
                )
            )
        else:
            noi_energy_mJ_remap = 0.0
            noc_energy_mJ_remap = 0.0

        noi_energy_mJ = noi_energy_mJ_mapped + noi_energy_mJ_remap
        noc_energy_mJ = noc_energy_mJ_mapped + noc_energy_mJ_remap
        comm_energy_mJ = noi_energy_mJ + noc_energy_mJ
        end_to_end_energy_mJ = comm_energy_mJ + total_comp_energy_J * 1e3

        # ============================================
        # CALCULATE TOTAL AREA AND LIF MEMORY
        # ============================================
        total_lif_tiles = LIF_needed
        total_lif_mem_kb = total_lif_tiles * SRAM_KB_per_tile
        total_compute_tiles = int(np.ceil(sum(mapper.xbars) / NPE))
        print(f"max_chiplets_used: {max_chiplets_used}")
        total_area_sq_mm = (
            max_chiplets_used * chiplet_area
            + total_lif_tiles * tile_area
        )

        # Store regular results
        global_results_regular = []
        global_results_regular.append(
            {
                'lif_t': LIF_T_val,
                'system_size': mesh_cols * mesh_rows,
                'LIF_MEM(KB)': total_lif_mem_kb,
                'LIF_MEM_Tiles': total_lif_tiles,
                'num_groups': 1,
                'Total_area_sq_mm': total_area_sq_mm,
                'NoC_latency_ms': total_noc_latency,
                'NoI_latency_ms': total_noi_latency,
                'Total_comm_latency_ms': total_comm_latency_ms,
                'Comp_Latency_ms': total_comp_latency_ms,
                'Comm_energy_mJ': comm_energy_mJ,
                'Comp_energy_mJ': total_comp_energy_J * 1e3,
                'Comm_power_W': comm_power_W,
                'End_to_end_latency_ms': end_to_end_latency,
                'End_to_end_energy_mJ': end_to_end_energy_mJ,
                'TOPS/Area': 1e3 / (end_to_end_latency * total_area_sq_mm),
                'Groupings': global_grp,
                'Chiplet_mapping': mapper.chiplet_data['main_chiplets'],
            }
        )

        global_summary_df_current_regular = pd.DataFrame(
            [
                {
                    k: v
                    for k, v in result.items()
                    if k not in ['Groupings', 'Chiplet_mapping']
                }
                for result in global_results_regular
            ]
        )

        global_results_by_lif_regular[LIF_T_val] = (
            global_summary_df_current_regular
        )

        # ============================================
        # OPTIMIZED MAPPING - Only if run_optimized is True
        # ============================================
        if run_optimized:
            cc_optimized = copy.deepcopy(cc)
            lif_index_to_remove = None
            for idx, chiplet in enumerate(cc_optimized):
                if (
                    'Layer_tile_distribution' in chiplet
                    and 'LIF0' in chiplet['Layer_tile_distribution']
                ):
                    lif_index_to_remove = idx
                    break
            if lif_index_to_remove is not None:
                cc_optimized.pop(lif_index_to_remove)

            optimized_lif_pos = result[0][0][0]
            cc_optimized.insert(optimized_lif_pos, LIF_chiplet)

            # Initialize accumulators for optimized mapping
            all_remap_NoI_latencies_opt = []
            all_remap_NoI_powers_opt = []
            all_remap_NoC_latencies_opt = []
            all_remap_NoC_powers_opt = []
            all_remap_DRAM_latencies_opt = []

            all_layers_noi_latencies_opt = {}
            all_layers_noc_latencies_opt = {}
            all_layers_dram_latencies_opt = {}

            mapped_noi_powers_raw_opt = []
            mapped_noi_latencies_ms_opt = []
            mapped_noc_powers_raw_opt = []
            mapped_noc_latencies_ms_opt = []

            # ============================================
            # OPTIMIZED FIRST MAPPING
            # ============================================
            first_map_global_NoI_latency_ms_opt = 0
            first_map_global_NoI_power_raw_opt = 0
            first_map_global_NoC_latency_ms_opt = 0
            first_map_global_NoC_power_raw_opt = 0

            for layer_id in mapped_layers:
                all_traffic_opt = []
                traffic_dict_opt = get_layer_traffic(
                    layer_id=layer_id,
                    chiplet_data=cc_optimized,
                    groupings=global_grp_optimized,
                    weights=mapper.weights,
                    tunable_params=mapper.tunable_params,
                    xbars=mapper.xbars,
                    X=mapper.X,
                    Vmem_res=mapper.Vmem_res,
                    Timestep=Timestep,
                    NoC_buswidth=1,
                    lif_tiles_per_layer=np.ceil(
                        np.array(mapper.MEMS) / (SRAM_KB_per_tile * 1024)
                    ).astype(int),
                    SRAM_KB_per_tile=SRAM_KB_per_tile,
                    acc_enabled=False,
                )
                all_traffic_opt.extend(
                    [traffic_dict_opt["output"], traffic_dict_opt["input"]]
                )

                tt_opt, M_sys_df_opt = create_system_matrix_from_edges(
                    all_traffic_opt,
                    mesh_rows=mesh_rows,
                    mesh_cols=mesh_cols,
                    count_diagonal=True,
                )
                M_sys_df_opt = np.ceil(M_sys_df_opt / NoI_buswidth).astype(int)
                M_sys_df_opt, _ = split_top_rest(
                    M_sys_df_opt, percent=percent_keep
                )

                _, _, system_scaled_df_opt, system_scaling_factor_opt = (
                    scale_traffic_matrices(
                        system_matrix=M_sys_df_opt,
                        chiplet_matrices=[],
                        minimum_traffic=min_traffic,
                    )
                )

                noi_latency_raw_opt, noi_power_raw_opt = run_booksim_NoI(
                    system_scaled_df_opt.values, NoI_mesh_layout
                )
                noI_latency_ms_opt = (
                    noi_latency_raw_opt
                    * system_scaling_factor_opt
                    * NoI_cycle_time
                )
                first_map_global_NoI_latency_ms_opt += noI_latency_ms_opt
                first_map_global_NoI_power_raw_opt += noi_power_raw_opt
                mapped_noi_powers_raw_opt.append(noi_power_raw_opt)
                mapped_noi_latencies_ms_opt.append(noI_latency_ms_opt)

                chiplet_dfs_opt = []
                for c in range(mesh_cols * mesh_rows):
                    M_chip_df_opt = np.ceil(
                        create_tile_matrix_for_chiplet(
                            all_traffic_opt, c, NT, include_chiplets=False
                        )[1]
                        / NoC_buswidth
                    ).astype(int)
                    chiplet_dfs_opt.append(M_chip_df_opt)

                M_chiplet_dfs_opt = [
                    split_top_rest(df, percent=percent_keep)[0]
                    for df in chiplet_dfs_opt
                ]
                (
                    chiplet_scaled_dfs_opt,
                    chiplet_scaling_factors_opt,
                    _,
                    _,
                ) = scale_traffic_matrices(
                    system_matrix=[],
                    chiplet_matrices=M_chiplet_dfs_opt,
                    minimum_traffic=min_traffic,
                )

                NoC_mesh_layouts_opt = [NoC_mesh_layout] * len(
                    cc_optimized
                )
                noc_latencies_raw_opt, noc_power_raw_opt = (
                    run_booksim_NoC(
                        chiplet_scaled_dfs_opt, NoC_mesh_layouts_opt
                    )
                )
                noc_latencies_ms_opt = (
                    (
                        np.array(noc_latencies_raw_opt)
                        * np.array(chiplet_scaling_factors_opt)
                        * NoC_cycle_time
                    )
                    .sum()
                )
                noc_power_sum_opt = np.array(noc_power_raw_opt).sum()
                first_map_global_NoC_latency_ms_opt += noc_latencies_ms_opt
                first_map_global_NoC_power_raw_opt += noc_power_sum_opt
                mapped_noc_powers_raw_opt.append(noc_power_sum_opt)
                mapped_noc_latencies_ms_opt.append(noc_latencies_ms_opt)

                all_layers_noi_latencies_opt[layer_id] = noI_latency_ms_opt
                all_layers_noc_latencies_opt[layer_id] = noc_latencies_ms_opt
                all_layers_dram_latencies_opt[layer_id] = 0.0

            # ============================================
            # OPTIMIZED REMAP BLOCK
            # ============================================
            lif_to_DRAM_packets_sum_opt = 0
            unmapped_layers_opt = []
            for i, each_remap in enumerate(
                mapper.chiplet_data['unmapped_layers']
            ):
                unmapped_layers_opt = sorted(
                    {
                        d['Layers_filled'][0]
                        for group in mapper.chiplet_data['unmapped_layers']
                        for d in group
                    }
                )
                current_layer_id = sorted(
                    {
                        layer
                        for chiplet in each_remap
                        for layer in chiplet.get('Layers_filled', [])
                    }
                )[0]

                cc_to_lif_opt = copy.deepcopy(each_remap)
                cc_to_lif_opt += [EMPTY] * (
                    mesh_cols * mesh_rows - len(cc_to_lif_opt) - 1
                )
                lif_placement_opt = result[0][0][0]
                cc_to_lif_opt.insert(lif_placement_opt, LIF_chiplet)

                # ============================================
                # OPTIMIZED PHASE 1: TO LIF
                # ============================================
                remap_mapped_layers_opt = sorted(
                    {
                        layer
                        for chiplet in each_remap
                        for layer in chiplet.get('Layers_filled', [])
                    }
                )
                remap_NoI_latency_ms_to_LIF_opt = 0
                remap_NoI_power_raw_to_LIF_opt = 0
                dram_packets_to_lif_opt = 0

                for layer_id in remap_mapped_layers_opt:
                    all_traffic_opt = []
                    traffic_dict_opt = get_layer_traffic(
                        layer_id=layer_id,
                        chiplet_data=cc_to_lif_opt,
                        groupings=global_grp_optimized,
                        weights=mapper.weights,
                        tunable_params=mapper.tunable_params,
                        xbars=mapper.xbars,
                        X=mapper.X,
                        Vmem_res=mapper.Vmem_res,
                        Timestep=Timestep,
                        NoC_buswidth=1,
                        lif_tiles_per_layer=np.ceil(
                            np.array(mapper.MEMS)
                            / (SRAM_KB_per_tile * 1024)
                        ).astype(int),
                        SRAM_KB_per_tile=SRAM_KB_per_tile,
                        acc_enabled=False,
                    )
                    all_traffic_opt.extend(
                        [traffic_dict_opt["output"], traffic_dict_opt["input"]]
                    )

                    _, M_sys_df_with_dram_opt = (
                        create_system_matrix_from_edges(
                            all_traffic_opt,
                            mesh_rows=mesh_rows + 1,
                            mesh_cols=mesh_cols,
                            count_diagonal=True,
                        )
                    )
                    layer_dram_packets_to_opt = (
                        M_sys_df_with_dram_opt.iloc[-1, :].sum()
                        + M_sys_df_with_dram_opt.iloc[:, -1].sum()
                    )
                    dram_packets_to_lif_opt += layer_dram_packets_to_opt

                    _, M_sys_df_opt = create_system_matrix_from_edges(
                        all_traffic_opt,
                        mesh_rows=mesh_rows,
                        mesh_cols=mesh_cols,
                        count_diagonal=True,
                    )
                    M_sys_df_opt = np.ceil(
                        M_sys_df_opt / NoI_buswidth
                    ).astype(int)
                    M_sys_df_opt, _ = split_top_rest(
                        M_sys_df_opt, percent=percent_keep
                    )

                    _, _, system_scaled_df_opt, system_scaling_factor_opt = (
                        scale_traffic_matrices(
                            system_matrix=M_sys_df_opt,
                            chiplet_matrices=[],
                            minimum_traffic=min_traffic,
                        )
                    )

                    noi_latency_raw_opt, noi_power_raw_opt = (
                        run_booksim_NoI(
                            system_scaled_df_opt.values, NoI_mesh_layout
                        )
                    )
                    noI_latency_ms_opt = (
                        noi_latency_raw_opt
                        * system_scaling_factor_opt
                        * NoI_cycle_time
                    )
                    remap_NoI_latency_ms_to_LIF_opt += noI_latency_ms_opt
                    remap_NoI_power_raw_to_LIF_opt += noi_power_raw_opt

                DRAM_write_latency_ms_opt = (
                    (dram_packets_to_lif_opt * NoI_buswidth / 8)
                    / (eta * DRAM_BW * 1e9)
                ) * 1e3

                # ============================================
                # OPTIMIZED PHASE 2: FROM LIF
                # ============================================
                remap_NoI_latency_ms_from_LIF_opt = 0
                remap_NoI_power_raw_from_LIF_opt = 0
                DRAM_read_latency_ms_opt = 0
                dram_packets_from_lif_opt = 0

                if i + 1 < len(
                    mapper.chiplet_data['unmapped_layers']
                ):
                    xx = copy.deepcopy(
                        mapper.chiplet_data['unmapped_layers'][i + 1]
                    )
                    dd_opt = [LIF_chiplet] + [EMPTY] * (
                        mesh_cols * mesh_rows - len(xx)
                    ) + xx

                    next_remap_mapped_layers_opt = sorted(
                        {
                            layer
                            for chiplet in xx
                            for layer in chiplet.get('Layers_filled', [])
                        }
                    )

                    for layer_id in next_remap_mapped_layers_opt:
                        all_traffic_opt = []
                        traffic_dict_opt = get_layer_traffic(
                            layer_id=layer_id,
                            chiplet_data=dd_opt,
                            groupings=global_grp_optimized,
                            weights=mapper.weights,
                            tunable_params=mapper.tunable_params,
                            xbars=mapper.xbars,
                            X=mapper.X,
                            Vmem_res=mapper.Vmem_res,
                            Timestep=Timestep,
                            NoC_buswidth=1,
                            lif_tiles_per_layer=np.ceil(
                                np.array(mapper.MEMS)
                                / (SRAM_KB_per_tile * 1024)
                            ).astype(int),
                            SRAM_KB_per_tile=SRAM_KB_per_tile,
                            acc_enabled=False,
                        )
                        all_traffic_opt.extend(
                            [
                                traffic_dict_opt["output"],
                                traffic_dict_opt["input"],
                            ]
                        )

                        _, M_sys_df_opt = (
                            create_system_matrix_from_edges(
                                all_traffic_opt,
                                mesh_rows=mesh_rows + 1,
                                mesh_cols=mesh_cols,
                                count_diagonal=True,
                            )
                        )
                        layer_dram_packets_opt = (
                            M_sys_df_opt.iloc[-1, :].sum()
                            + M_sys_df_opt.iloc[:, -1].sum()
                        )
                        lif_to_DRAM_packets_sum_opt += (
                            layer_dram_packets_opt
                        )
                        dram_packets_from_lif_opt += (
                            layer_dram_packets_opt
                        )

                        M_sys_df_opt = M_sys_df_opt.iloc[
                            : -(mesh_cols - 1), : -(mesh_cols - 1)
                        ]
                        M_sys_df_opt = np.ceil(
                            M_sys_df_opt / NoI_buswidth
                        ).astype(int)
                        M_sys_df_opt, _ = split_top_rest(
                            M_sys_df_opt, percent=percent_keep
                        )

                        _, _, system_scaled_df_opt, system_scaling_factor_opt = (
                            scale_traffic_matrices(
                                system_matrix=M_sys_df_opt,
                                chiplet_matrices=[],
                                minimum_traffic=min_traffic,
                            )
                        )

                        noi_latency_raw_opt, noi_power_raw_opt = (
                            run_booksim_NoI(
                                system_scaled_df_opt.values,
                                NoI_mesh_layout_remap_from_LIFDRAM,
                            )
                        )
                        noI_latency_ms_opt = (
                            noi_latency_raw_opt
                            * system_scaling_factor_opt
                            * NoI_cycle_time
                        )
                        remap_NoI_latency_ms_from_LIF_opt += (
                            noI_latency_ms_opt
                        )
                        remap_NoI_power_raw_from_LIF_opt += (
                            noi_power_raw_opt
                        )

                    DRAM_read_latency_ms_opt = (
                        (dram_packets_from_lif_opt * NoI_buswidth / 8)
                        / (eta * DRAM_BW * 1e9)
                    ) * 1e3

                # ============================================
                # OPTIMIZED NoC Latency Calculation
                # ============================================
                all_traffic_noc_opt = []
                for layer_id in remap_mapped_layers_opt:
                    traffic_dict_noc_opt = get_layer_traffic(
                        layer_id=layer_id,
                        chiplet_data=cc_to_lif_opt,
                        groupings=global_grp_optimized,
                        weights=mapper.weights,
                        tunable_params=mapper.tunable_params,
                        xbars=mapper.xbars,
                        X=mapper.X,
                        Vmem_res=mapper.Vmem_res,
                        Timestep=Timestep,
                        NoC_buswidth=1,
                        lif_tiles_per_layer=np.ceil(
                            np.array(mapper.MEMS)
                            / (SRAM_KB_per_tile * 1024)
                        ).astype(int),
                        SRAM_KB_per_tile=SRAM_KB_per_tile,
                        acc_enabled=False,
                    )
                    all_traffic_noc_opt.extend(
                        [
                            traffic_dict_noc_opt["output"],
                            traffic_dict_noc_opt["input"],
                        ]
                    )

                chiplet_dfs_opt = []
                for c in range(mesh_cols * mesh_rows):
                    M_chip_df_opt = np.ceil(
                        create_tile_matrix_for_chiplet(
                            all_traffic_noc_opt, c, NT, include_chiplets=False
                        )[1]
                        / NoC_buswidth
                    ).astype(int)
                    chiplet_dfs_opt.append(M_chip_df_opt)

                M_chiplet_dfs_opt = [
                    split_top_rest(df, percent=percent_keep)[0]
                    for df in chiplet_dfs_opt
                ]
                (
                    chiplet_scaled_dfs_opt,
                    chiplet_scaling_factors_opt,
                    _,
                    _,
                ) = scale_traffic_matrices(
                    system_matrix=[],
                    chiplet_matrices=M_chiplet_dfs_opt,
                    minimum_traffic=min_traffic,
                )

                NoC_mesh_layouts_opt = [NoC_mesh_layout] * len(
                    cc_to_lif_opt
                )
                noc_latencies_raw_opt, noc_power_raw_opt = (
                    run_booksim_NoC(
                        chiplet_scaled_dfs_opt, NoC_mesh_layouts_opt
                    )
                )
                noc_latencies_ms_opt = (
                    (
                        np.array(noc_latencies_raw_opt)
                        * np.array(chiplet_scaling_factors_opt)
                        * NoC_cycle_time
                    )
                    .sum()
                )
                noc_power_sum_opt = np.array(noc_power_raw_opt).sum()

                # ============================================
                # ACCUMULATE OPTIMIZED LATENCIES
                # ============================================
                total_remap_noi_latency_opt = (
                    remap_NoI_latency_ms_to_LIF_opt
                    + remap_NoI_latency_ms_from_LIF_opt
                )
                total_remap_dram_latency_opt = (
                    DRAM_write_latency_ms_opt + DRAM_read_latency_ms_opt
                )
                total_remap_dram_packets_opt = (
                    dram_packets_to_lif_opt
                    + dram_packets_from_lif_opt
                )

                layer_param_dram_latency_ms_opt = (
                    (mapper.tunable_params[current_layer_id - 1] / 8)
                    / (eta * DRAM_BW * 1e9)
                ) * 1e3

                layer_lif_dram_latency_ms_opt = (
                    (
                        2
                        * total_remap_dram_packets_opt
                        * NoI_buswidth
                        / 8
                    )
                    / (eta * DRAM_BW * 1e9)
                ) * 1e3

                total_layer_dram_latency_ms_opt = (
                    total_remap_dram_latency_opt
                    + layer_param_dram_latency_ms_opt
                    + layer_lif_dram_latency_ms_opt
                )

                all_remap_NoI_latencies_opt.append(
                    total_remap_noi_latency_opt
                )
                all_remap_NoI_powers_opt.append(
                    remap_NoI_power_raw_to_LIF_opt
                    + remap_NoI_power_raw_from_LIF_opt
                )
                all_remap_NoC_latencies_opt.append(noc_latencies_ms_opt)
                all_remap_NoC_powers_opt.append(noc_power_sum_opt)
                all_remap_DRAM_latencies_opt.append(
                    total_remap_dram_latency_opt
                )

                all_layers_noi_latencies_opt[current_layer_id] = (
                    total_remap_noi_latency_opt
                )
                all_layers_noc_latencies_opt[current_layer_id] = (
                    noc_latencies_ms_opt
                )
                all_layers_dram_latencies_opt[current_layer_id] = (
                    total_layer_dram_latency_ms_opt
                )

            # ============================================
            # COMPUTATION LATENCY AND ENERGY FOR OPTIMIZED
            # ============================================
            comp_latency_per_layer_s_opt = (
                np.array(mapper.TOPS) * Timestep / (TOPS * 8)
            )
            comp_latency_per_layer_ms_opt = (
                comp_latency_per_layer_s_opt * 1e3
            )
            comp_energy_per_layer_J_opt = (
                np.array(mapper.TOPS) * Timestep * ENERGY_PER_MAC_pj
            )
            total_comp_latency_ms_opt = comp_latency_per_layer_ms_opt.sum()
            total_comp_energy_J_opt = comp_energy_per_layer_J_opt.sum()

            # ============================================
            # END-TO-END LATENCY CALCULATION FOR OPTIMIZED
            # ============================================
            mapped_layers_total_time_opt = 0.0
            for layer_id in mapped_layers:
                layer_time_opt = (
                    all_layers_noi_latencies_opt[layer_id]
                    + all_layers_noc_latencies_opt[layer_id]
                    + comp_latency_per_layer_ms_opt[layer_id - 1]
                )
                mapped_layers_total_time_opt += layer_time_opt

            end_to_end_latency_opt = mapped_layers_total_time_opt
            overlap_penalties_opt = []

            if len(unmapped_layers_opt) > 0:
                for remap_idx, unmapped_layer_id in enumerate(
                    unmapped_layers_opt
                ):
                    layer_processing_time_opt = (
                        all_layers_noi_latencies_opt[unmapped_layer_id]
                        + all_layers_noc_latencies_opt[unmapped_layer_id]
                        + comp_latency_per_layer_ms_opt[
                            unmapped_layer_id - 1
                        ]
                    )
                    layer_total_dram_opt = (
                        all_layers_dram_latencies_opt[unmapped_layer_id]
                    )

                    if remap_idx == 0:
                        available_time_opt = mapped_layers_total_time_opt
                    else:
                        prev_layer_id = unmapped_layers_opt[remap_idx - 1]
                        available_time_opt = (
                            all_layers_noi_latencies_opt[prev_layer_id]
                            + all_layers_noc_latencies_opt[prev_layer_id]
                            + comp_latency_per_layer_ms_opt[
                                prev_layer_id - 1
                            ]
                        )

                    overlap_penalty_opt = max(
                        0, layer_total_dram_opt - available_time_opt
                    )
                    overlap_penalties_opt.append(overlap_penalty_opt)
                    end_to_end_latency_opt += (
                        layer_processing_time_opt + overlap_penalty_opt
                    )

            # ============================================
            # CALCULATE TOTAL COMM LATENCY AND POWER/ENERGY FOR OPTIMIZED
            # ============================================
            total_noi_latency_opt = (
                first_map_global_NoI_latency_ms_opt
                + sum(all_remap_NoI_latencies_opt)
            )
            total_noc_latency_opt = (
                first_map_global_NoC_latency_ms_opt
                + sum(all_remap_NoC_latencies_opt)
            )
            total_comm_latency_ms_opt = (
                total_noi_latency_opt + total_noc_latency_opt
            )
            total_noi_power_raw_opt = (
                first_map_global_NoI_power_raw_opt
                + sum(all_remap_NoI_powers_opt)
            )
            total_noc_power_raw_opt = (
                first_map_global_NoC_power_raw_opt
                + sum(all_remap_NoC_powers_opt)
            )
            comm_power_W_opt = (
                total_noi_power_raw_opt + total_noc_power_raw_opt
            )

            noi_energy_mJ_mapped_opt = float(
                np.sum(
                    np.array(mapped_noi_powers_raw_opt)
                    * np.array(mapped_noi_latencies_ms_opt)
                )
            )
            noc_energy_mJ_mapped_opt = float(
                np.sum(
                    np.array(mapped_noc_powers_raw_opt)
                    * np.array(mapped_noc_latencies_ms_opt)
                )
            )

            if len(all_remap_NoI_powers_opt) > 0:
                noi_energy_mJ_remap_opt = float(
                    np.sum(
                        np.array(all_remap_NoI_powers_opt)
                        * np.array(all_remap_NoI_latencies_opt)
                    )
                )
                noc_energy_mJ_remap_opt = float(
                    np.sum(
                        np.array(all_remap_NoC_powers_opt)
                        * np.array(all_remap_NoC_latencies_opt)
                    )
                )
            else:
                noi_energy_mJ_remap_opt = 0.0
                noc_energy_mJ_remap_opt = 0.0

            noi_energy_mJ_opt = (
                noi_energy_mJ_mapped_opt + noi_energy_mJ_remap_opt
            )
            noc_energy_mJ_opt = (
                noc_energy_mJ_mapped_opt + noc_energy_mJ_remap_opt
            )
            comm_energy_mJ_opt = noi_energy_mJ_opt + noc_energy_mJ_opt
            end_to_end_energy_mJ_opt = (
                comm_energy_mJ_opt + total_comp_energy_J_opt * 1e3
            )

            # ============================================
            # CALCULATE TOTAL AREA AND LIF MEMORY FOR OPTIMIZED
            # ============================================
            total_lif_tiles_opt = LIF_needed
            total_lif_mem_kb_opt = (
                total_lif_tiles_opt * SRAM_KB_per_tile
            )
            total_compute_tiles_opt = int(np.ceil(sum(mapper.xbars) / NPE))
            print('max_chiplets_used:', max_chiplets_used )
            total_area_sq_mm_opt = (
                max_chiplets_used * chiplet_area
                + total_lif_tiles_opt * tile_area
            )

            # Store optimized results
            global_results_optimized = []
            global_results_optimized.append(
                {
                    'lif_t': LIF_T_val,
                    'system_size': mesh_cols * mesh_rows,
                    'LIF_MEM(KB)': total_lif_mem_kb_opt,
                    'LIF_MEM_Tiles': total_lif_tiles_opt,
                    'num_groups': 1,
                    'Total_area_sq_mm': total_area_sq_mm_opt,
                    'NoC_latency_ms': total_noc_latency_opt,
                    'NoI_latency_ms': total_noi_latency_opt,
                    'Total_comm_latency_ms': total_comm_latency_ms_opt,
                    'Comp_Latency_ms': total_comp_latency_ms_opt,
                    'Comm_energy_mJ': comm_energy_mJ_opt,
                    'Comp_energy_mJ': total_comp_energy_J_opt * 1e3,
                    'Comm_power_W': comm_power_W_opt,
                    'End_to_end_latency_ms': end_to_end_latency_opt,
                    'End_to_end_energy_mJ': end_to_end_energy_mJ_opt,
                    'TOPS/Area': 1e3 / (end_to_end_latency_opt * total_area_sq_mm_opt),
                    'Groupings': global_grp_optimized,
                    'Chiplet_mapping': mapper.chiplet_data[
                        'main_chiplets'
                    ],
                }
            )

            global_summary_df_current_optimized = pd.DataFrame(
                [
                    {
                        k: v
                        for k, v in result.items()
                        if k not in ['Groupings', 'Chiplet_mapping']
                    }
                    for result in global_results_optimized
                ]
            )

            global_results_by_lif_optimized[LIF_T_val] = (
                global_summary_df_current_optimized
            )

    # ============================================
    # FINAL SUMMARY - All LIF_T values together
    # ============================================
    all_combined_results_regular = []
    for lif_t, df in global_results_by_lif_regular.items():
        df_with_lif = df.copy()
        all_combined_results_regular.append(df_with_lif)

    if all_combined_results_regular:
        final_global_summary_df_regular = pd.concat(
            all_combined_results_regular, ignore_index=True
        )
        global_results_dict_regular = global_results_by_lif_regular
    else:
        print("No regular results to display")
        global_results_dict_regular = {}

    all_combined_results_optimized = []
    for lif_t, df in global_results_by_lif_optimized.items():
        df_with_lif = df.copy()
        all_combined_results_optimized.append(df_with_lif)

    if all_combined_results_optimized:
        final_global_summary_df_optimized = pd.concat(
            all_combined_results_optimized, ignore_index=True
        )
        global_results_dict_optimized = global_results_by_lif_optimized
    else:
        final_global_summary_df_optimized = None
        global_results_dict_optimized = {}

    cleanup_booksim_files()
    return (
        final_global_summary_df_regular,
        final_global_summary_df_optimized,
    )

def run_amosa_optimization(
    network,
    LIF_T_list,
    NPE,
    NT,
    xbar_size,
    SRAM_KB_per_tile,
    XBAR_bits_per_cell,
    Vmem_res,
    Timestep,
    NoC_buswidth,
    NoI_buswidth,
    mesh_rows,
    mesh_cols,
    max_num_groups,
    iterations_per_temp,
    inter_cost,
    w_mem,
    N,
    debug=0,
    debug_here=False
):
    import random
    import time
    import pandas as pd
    import warnings

    # Seed RNG once
    random_seed = int(time.time())
    random.seed(random_seed)
    
    if debug:
        print(f"Random seed: {random_seed}")

    # Store results for each LIF_T value
    all_pareto_results = {}
    balanced_results = {}
    w_cost = 1.0 - w_mem

    # --- Compute individual mesh configuration ---
    try:
        mapper_temp = SNNMapper(
            weights=network, 
            layer_groups=[], 
            NPE=NPE,
            NT=np.ceil(NT/2).astype(int).item(), 
            X=xbar_size,
            bits_per_cell=XBAR_bits_per_cell,
            P=100,
            Vmem_res=Vmem_res, 
            Timestep=Timestep,
            NoC_buswidth=NoC_buswidth, 
            NoI_buswidth=NoI_buswidth,
            allow_break_columns=True,
            include_chiplets=False,
            max_chiplets=mesh_cols * mesh_rows
        )
        mapper_temp.run()
        
        compute_chiplets_needed = np.ceil(
            np.ceil(np.array(mapper_temp.xbars) / NPE) / (NT - 8)
        ).sum().astype(int)
        
        lif_chiplets_needed_total = np.ceil(
            np.ceil(np.array(mapper_temp.MEMS) / (SRAM_KB_per_tile * 1024)).sum() / 8
        ).astype(int).item()
        
        mapper_temp = None  # Free memory
        
        individual_system_size = nearest_almost_square(
            max(compute_chiplets_needed, lif_chiplets_needed_total)
        )
        ind_mesh_rows, ind_mesh_cols = find_best_grid(individual_system_size)
        
        if debug_here:
            print(f'ind_mesh_rows, ind_mesh_cols: {ind_mesh_rows}, {ind_mesh_cols}')
            
    except Exception as e:
        warnings.warn(f"Failed to compute individual mesh config: {e}. Using defaults.")
        ind_mesh_rows, ind_mesh_cols = mesh_rows, mesh_cols

    # --- Main optimization loop ---
    for LIF_T_val in LIF_T_list:
        if debug_here:
            print(f"\n{'-'*60}")
            print(f"Running AMOSA for LIF_T = {LIF_T_val}")
            print(f"{'-'*60}")

        try:
            # Build mapper once per LIF_T
            mapper = SNNMapper(
                weights=network,
                layer_groups=[],
                NPE=NPE,
                NT=NT - LIF_T_val,
                X=xbar_size,
                P=100,
                Vmem_res=Vmem_res,
                Timestep=Timestep,
                NoC_buswidth=NoC_buswidth,
                NoI_buswidth=NoI_buswidth,
                bits_per_cell=XBAR_bits_per_cell,
                allow_break_columns=True,
                include_chiplets=False,
                max_chiplets=mesh_cols * mesh_rows,
            )

            # Pre-compute mapper internals
            (
                mapper.tunable_params,
                mapper.xbars,
                mapper.IFMS,
                mapper.OFMS,
                mapper.TOPS,
                mapper.MEMS,
            ) = mapper._calc_tunable_params()

            mapper.layer_output_sizes = dict(
                zip(range(1, len(mapper.OFMS) + 1), mapper.OFMS)
            )
            mapper.chiplet_data = mapper._generate_chiplet_mapping()
            
        except Exception as e:
            warnings.warn(f"Failed to build mapper for LIF_T={LIF_T_val}: {e}")
            all_pareto_results[LIF_T_val] = pd.DataFrame()
            balanced_results[LIF_T_val] = pd.DataFrame()
            continue

        # Sweep num_groups
        all_pareto_rows = []
        balanced_rows = []
        num_memory_layers = len(mapper.MEMS)

        # --- Build num_groups list (FIXED) ---
        unmapped_layers = mapper.chiplet_data.get('unmapped_layers', [])
        num_unmapped = len(unmapped_layers) if unmapped_layers else 0
        start_groups = max(1, 1 + num_unmapped)
        
        # Create base list
        num_groups_list = list(range(start_groups, max_num_groups + 1))
        
        # Filter to valid range
        num_groups_list = [g for g in num_groups_list if g <= num_memory_layers]
        
        # Add num_memory_layers if appropriate
        if num_memory_layers >= start_groups and num_memory_layers not in num_groups_list:
            num_groups_list.append(num_memory_layers)
        
        # Sort and deduplicate
        num_groups_list = sorted(set(num_groups_list))
        
        # Handle empty list case
        if not num_groups_list:
            warnings.warn(
                f"No valid num_groups for LIF_T={LIF_T_val} "
                f"(start_groups={start_groups}, max={max_num_groups}, mem_layers={num_memory_layers})"
            )
            all_pareto_results[LIF_T_val] = pd.DataFrame()
            balanced_results[LIF_T_val] = pd.DataFrame()
            continue

        if debug_here:
            print(f"num_groups_list: {num_groups_list}")

        for num_groups in num_groups_list:
            try:
                # Determine mesh configuration
                if num_groups == num_memory_layers:
                    current_mesh_rows = ind_mesh_rows
                    current_mesh_cols = ind_mesh_cols
                    current_iterations_per_temp = 1
                else:
                    current_mesh_rows = mesh_rows
                    current_mesh_cols = mesh_cols
                    current_iterations_per_temp = iterations_per_temp

                if debug_here:
                    print(f"\n--- num_groups={num_groups}, mesh={current_mesh_rows}x{current_mesh_cols} ---")

                optimizer = AMOSAGroupingOptimizer(
                    mapper=mapper,
                    num_groups=num_groups,
                    mesh_rows=current_mesh_rows,
                    mesh_cols=current_mesh_cols,
                    LIF_T=LIF_T_val,
                    SRAM_KB_per_tile=SRAM_KB_per_tile,
                    NPE=NPE,
                    NT=NT,
                    inter_cost=inter_cost,
                    intra_cost=1,
                    archive_size=100,
                    initial_temp=500.0,
                    final_temp=1,
                    cooling_rate=0.98,
                    iterations_per_temp=current_iterations_per_temp,
                    random_seed=random_seed,
                    verbose_level=debug,
                )

                # Run optimization
                archive = optimizer.run()

                # Plot Pareto front (with error handling)
                try:
                    plot_pareto_front(
                        optimizer,
                        title=f"AMOSA (LIF_T={LIF_T_val}, groups={num_groups}, mesh={current_mesh_rows}x{current_mesh_cols})",
                        figsize=(7, 5)
                    )
                except Exception as plot_err:
                    if debug:
                        print(f"Warning: Failed to plot Pareto front: {plot_err}")

                # Get Pareto solutions
                pareto_solutions = optimizer.get_all_pareto_solutions()

                if not pareto_solutions:
                    if debug:
                        print(f"No Pareto solutions for num_groups={num_groups}")
                    continue

                df_pareto = pd.DataFrame(pareto_solutions)

                # Keep only unique (total_lif_tiles, total_cost) combos
                unique_pareto_df = (
                    df_pareto
                    .drop_duplicates(subset=["total_lif_tiles", "total_cost"])
                    .sort_values("total_lif_tiles")
                    .reset_index(drop=True)
                )

                # Tag with metadata
                unique_pareto_df["num_groups"] = num_groups
                unique_pareto_df["mesh_rows"] = current_mesh_rows
                unique_pareto_df["mesh_cols"] = current_mesh_cols
                unique_pareto_df["LIF_T"] = LIF_T_val
                unique_pareto_df["unmapped_layers"] = [
                    mapper.chiplet_data.get("unmapped_layers", [])
                ] * len(unique_pareto_df)

                all_pareto_rows.append(unique_pareto_df)

                # Get balanced solution
                try:
                    balanced_row = get_balanced_solutions(
                        unique_pareto_df,
                        n=N,
                        w_cost=w_cost,
                        w_mem=w_mem,
                    )
                    balanced_row = balanced_row.copy()
                    balanced_row["num_groups"] = num_groups
                    balanced_row["mesh_rows"] = current_mesh_rows
                    balanced_row["mesh_cols"] = current_mesh_cols
                    balanced_row["LIF_T"] = LIF_T_val
                    balanced_rows.append(balanced_row)
                except Exception as bal_err:
                    if debug_here:
                        print(f"Warning: Failed to get balanced solution: {bal_err}")

            except Exception as opt_err:
                warnings.warn(
                    f"Optimization failed for num_groups={num_groups}, LIF_T={LIF_T_val}: {opt_err}"
                )
                continue

        # Combine results for this LIF_T
        if all_pareto_rows:
            all_pareto_df = pd.concat(all_pareto_rows, ignore_index=True)
        else:
            all_pareto_df = pd.DataFrame()

        if balanced_rows:
            balanced_solutions_df = pd.concat(balanced_rows, ignore_index=True)
        else:
            balanced_solutions_df = pd.DataFrame()

        all_pareto_results[LIF_T_val] = all_pareto_df
        balanced_results[LIF_T_val] = balanced_solutions_df

    # Summary
    if debug_here:
        print(f"\n{'='*60}")
        print("Summary:")
        for lif_t in LIF_T_list:
            all_size = len(all_pareto_results.get(lif_t, pd.DataFrame()))
            balanced_size = len(balanced_results.get(lif_t, pd.DataFrame()))
            print(f"  LIF_T={lif_t}: Pareto={all_size}, Balanced={balanced_size}")

    return all_pareto_results, balanced_results
def optimize_lif_placements(
    network,
    xbar_size,
    XBAR_bits_per_cell,
    Vmem_res,
    Timestep,
    balanced_results,
    mesh_rows,
    mesh_cols,
    NoC_buswidth,
    NoI_buswidth,
    NT,
    NPE,
    LIF_T_list=None,
    visualize=True,
    debug=0
):
    import copy
    import pandas as pd
    import warnings
    
    def breakpoints_to_groups(breakpoints):
        """Convert breakpoints list to groups of [start, end] ranges."""
        groups = []
        start = 1
        for end in breakpoints:
            groups.append([start, end])
            start = end + 1
        return groups
    
    # Initialize results dictionary
    balanced_results_optimized = {}
    mapper_temp = SNNMapper(
            weights=network, 
            layer_groups=[], 
            NPE=NPE,
            NT=np.ceil(NT/2).astype(int).item(), 
            X=xbar_size,
            bits_per_cell=XBAR_bits_per_cell,
            P=100,
            Vmem_res=Vmem_res, 
            Timestep=Timestep,
            NoC_buswidth=NoC_buswidth, 
            NoI_buswidth=NoI_buswidth,
            allow_break_columns=True,
            include_chiplets=False,
            max_chiplets=mesh_cols * mesh_rows
        )
    mapper_temp.run()
    layer_output_sizes = dict(
        zip(range(1, len(mapper_temp.OFMS) + 1), mapper_temp.OFMS)
    )
    mapper_temp = None  # Free memory
    # Determine which LIF_T values to process
    if LIF_T_list is None:
        LIF_T_list = list(balanced_results.keys())
    
    # Validate LIF_T values exist in balanced_results
    valid_lif_t_list = [
        lif_t for lif_t in LIF_T_list 
        if lif_t in balanced_results and not balanced_results[lif_t].empty
    ]
    
    if not valid_lif_t_list:
        warnings.warn("No valid LIF_T values found in balanced_results")
        return balanced_results_optimized
    
    # Prepare layer weights (scaled output sizes)
    try:
        layer_ofms = copy.deepcopy(layer_output_sizes)
        layer_ofms = {k: v / 1e4 for k, v in layer_ofms.items()}
    except Exception as e:
        warnings.warn(f"Failed to prepare layer weights: {e}")
        return balanced_results_optimized
    
    # Process each LIF_T value
    for LIF_T_each in valid_lif_t_list:
        if debug >= 1:
            print(f"\n{'='*60}")
            print(f"Optimizing for LIF_T = {LIF_T_each}")
            print(f"{'='*60}")
        
        try:
            balanced_solution = balanced_results[LIF_T_each]
            
            # Validate DataFrame
            if balanced_solution.empty:
                if debug >= 1:
                    print(f"  Skipping LIF_T={LIF_T_each}: Empty DataFrame")
                balanced_results_optimized[LIF_T_each] = balanced_solution.copy()
                continue
            
            # Initialize optimized DataFrame for this LIF_T
            balanced_results_optimized[LIF_T_each] = balanced_solution.copy()
            
        except Exception as e:
            warnings.warn(f"Failed to initialize for LIF_T={LIF_T_each}: {e}")
            balanced_results_optimized[LIF_T_each] = pd.DataFrame()
            continue
        
        # Process each row in the balanced solution
        for row_id in range(len(balanced_solution)):
            if debug >= 1:
                print(f"\n  Processing row {row_id} for LIF_T={LIF_T_each}")
            
            try:
                row_data = balanced_solution.iloc[row_id]
                
                # --- Extract required data from row ---
                # Get remapped layers
                unmapped_layers = row_data.get('unmapped_layers', [])
                try:
                    remap_mapped_layers = sorted({
                        layer 
                        for entry in unmapped_layers 
                        for layer in entry[0].get('Layers_filled', [])
                    })
                except (TypeError, KeyError, IndexError):
                    remap_mapped_layers = []
                    if debug >= 2:
                        print(f"    Warning: Could not extract remap_mapped_layers")
                
                # Get breakpoints
                breakpoints = row_data.get('breakpoints', [])
                if not breakpoints:
                    if debug >= 2:
                        print(f"    Skipping row {row_id}: No breakpoints found")
                    continue
                
                # Get LIF distributions
                old_lif_distributions = row_data.get('lif_distributions', [])
                if not old_lif_distributions:
                    if debug >= 2:
                        print(f"    Skipping row {row_id}: No lif_distributions found")
                    continue
                
                # Calculate LIF needed by groups
                try:
                    lif_needed_by_groups = [
                        sum(count for _, count in group) 
                        for group in old_lif_distributions
                    ]
                except (TypeError, ValueError) as e:
                    if debug >= 2:
                        print(f"    Warning: Could not calculate lif_needed_by_groups: {e}")
                    lif_needed_by_groups = []
                
                # Get mesh configuration
                mesh_rows_each = row_data.get('mesh_rows', 1)
                mesh_cols_each = row_data.get('mesh_cols', 1)
                
                # Convert breakpoints to groups
                groups = breakpoints_to_groups(breakpoints)
                
                if debug >= 2:
                    print(f"    Groups: {groups}")
                    print(f"    LIF needed: {lif_needed_by_groups[:len(groups)]}")
                
                # --- Get and clean chiplet mapping ---
                old_chiplet_mapping = row_data.get('chiplet_mapping')
                if old_chiplet_mapping is None:
                    if debug >= 2:
                        print(f"    Skipping row {row_id}: No chiplet_mapping found")
                    continue
                
                old_chiplet_mapping = copy.deepcopy(old_chiplet_mapping)
                
                # Remove all 'LIF*' keys from Layer_tile_distribution
                for chiplet in old_chiplet_mapping:
                    if 'Layer_tile_distribution' not in chiplet:
                        continue
                    ltd = chiplet['Layer_tile_distribution']
                    keys_to_remove = [
                        k for k in ltd.keys() 
                        if isinstance(k, str) and k.startswith('LIF')
                    ]
                    for k in keys_to_remove:
                        del ltd[k]
                
                # --- Run optimizer ---
                try:
                    result, cost = optimize_multi_group_lif_placement(
                        chiplet_data=old_chiplet_mapping,
                        layer_weights=layer_ofms,
                        groups=groups,
                        lif_needed=lif_needed_by_groups[:len(groups)],
                        lif_capacity=LIF_T_each,
                        mesh_rows=mesh_rows_each,
                        mesh_cols=mesh_cols_each
                    )
                except Exception as opt_err:
                    warnings.warn(
                        f"Optimization failed for LIF_T={LIF_T_each}, row={row_id}: {opt_err}"
                    )
                    continue
                
                # --- Visualize old placement ---
                if visualize:
                    try:
                        new_groupings = copy.deepcopy(row_data.get('groups', []))
                        if debug >= 1:
                            print("    Old placement:")
                        _ = visualize_groups(
                            new_groupings, 
                            old_chiplet_mapping,
                            mesh_rows_each, 
                            mesh_cols_each,
                            LIF_T=LIF_T_each,
                            figsize=(8, 6), 
                            fig_fontsize=6, 
                            legend_fontsize=9, 
                            auto_display=True
                        )
                    except Exception as viz_err:
                        if debug >= 2:
                            print(f"    Warning: Could not visualize old placement: {viz_err}")
                
                # --- Update groupings with new LIF distributions ---
                new_groupings = copy.deepcopy(row_data.get('groups', []))
                
                for j in range(min(len(result), len(new_groupings))):
                    new_groupings[j]['lif_distribution'] = result[j]
                
                # --- Rebuild chiplet mapping with new LIF tiles ---
                try:
                    new_chiplet_mapping = add_lif_tiles(
                        old_chiplet_mapping,
                        new_groupings,
                        NT,
                        LIF_T_each,
                        NPE,
                        mesh_rows_each,
                        mesh_cols_each
                    )
                except Exception as add_err:
                    warnings.warn(
                        f"add_lif_tiles failed for LIF_T={LIF_T_each}, row={row_id}: {add_err}"
                    )
                    continue
                
                # --- Build updated row ---
                bal = copy.deepcopy(row_data.to_dict())
                bal['groups'] = new_groupings
                bal['chiplet_mapping'] = new_chiplet_mapping
                bal['lif_distributions'] = result
                
                # Safely replace entire row
                try:
                    balanced_results_optimized[LIF_T_each].iloc[row_id] = pd.Series(bal)
                except Exception as assign_err:
                    warnings.warn(
                        f"Failed to assign optimized row for LIF_T={LIF_T_each}, row={row_id}: {assign_err}"
                    )
                    continue
                
                # --- Visualize new placement ---
                if visualize:
                    try:
                        if debug >= 1:
                            print("    New placement:")
                        _ = visualize_groups(
                            balanced_results_optimized[LIF_T_each].iloc[row_id]['groups'],
                            balanced_results_optimized[LIF_T_each].iloc[row_id]['chiplet_mapping'],
                            mesh_rows_each, 
                            mesh_cols_each,
                            LIF_T=LIF_T_each,
                            figsize=(8, 6), 
                            fig_fontsize=6, 
                            legend_fontsize=9, 
                            auto_display=True
                        )
                    except Exception as viz_err:
                        if debug >= 2:
                            print(f"    Warning: Could not visualize new placement: {viz_err}")
                
                if debug >= 1:
                    print(f"    Row {row_id} optimized successfully (cost={cost})")
                    
            except Exception as row_err:
                warnings.warn(
                    f"Failed to process row {row_id} for LIF_T={LIF_T_each}: {row_err}"
                )
                continue
        
        # Summary for this LIF_T
        if debug >= 1:
            optimized_count = len(balanced_results_optimized[LIF_T_each])
            print(f"\n  LIF_T={LIF_T_each}: Processed {optimized_count} rows")
    
    # Final summary
    if debug >= 1:
        print(f"\n{'='*60}")
        print("Optimization Summary:")
        print(f"{'='*60}")
        for lif_t in valid_lif_t_list:
            if lif_t in balanced_results_optimized:
                size = len(balanced_results_optimized[lif_t])
                print(f"  LIF_T={lif_t}: {size} optimized solutions")
    
    return balanced_results_optimized

def run_distil_optimized_lif(
    balanced_results_optimized,
    network,
    LIF_T_list,
    NT,
    NPE,
    xbar_size,
    XBAR_bits_per_cell,
    Vmem_res,
    mesh_rows,
    mesh_cols,
    SRAM_KB_per_tile,
    Timestep,
    NoI_buswidth,
    NoC_buswidth,
    NoC_cycle_time,
    NoI_cycle_time,
    TOPS,
    ENERGY_PER_MAC_pj,
    chiplet_area,
    tile_area,
    percent_keep,
    min_traffic,
    eta,
    DRAM_BW,
    inter_cost,
    max_lif_tiles=None,
    visualize=True,
    debug=1
):
    """
    Run DISTIL analysis with optimized LIF placement for multiple LIF_T values.
    Skips individual layer grouping (num_groups == len(network)).
    
    Parameters
    ----------
    balanced_results_optimized : dict
        Dictionary mapping LIF_T values to optimized balanced solution DataFrames.
    network : list
        Network weights/architecture.
    LIF_T_list : list
        List of LIF_T values to process.
    NT : int
        Number of tiles per chiplet.
    NPE : int
        Number of processing elements per tile.
    xbar_size : int
        Crossbar size.
    XBAR_bits_per_cell : int
        Bits per cell in crossbar.
    Vmem_res : int
        Membrane voltage resolution.
    mesh_rows, mesh_cols : int
        System mesh dimensions.
    SRAM_KB_per_tile : float
        SRAM capacity per tile in KB.
    Timestep : int
        Number of timesteps.
    NoI_buswidth, NoC_buswidth : int
        Bus widths for NoI and NoC.
    NoC_cycle_time, NoI_cycle_time : float
        Cycle times in appropriate units.
    TOPS : float
        Peak TOPS performance.
    ENERGY_PER_MAC_pj : float
        Energy per MAC operation in pJ.
    chiplet_area : float
        Area per chiplet in mm².
    tile_area : float
        Area per tile in mm².
    percent_keep : float
        Percentage of traffic to keep after filtering.
    min_traffic : int
        Minimum traffic for scaling.
    eta : float
        DRAM efficiency factor.
    DRAM_BW : float
        DRAM bandwidth in GB/s.
    inter_cost : float
        Inter-chiplet communication cost.
    max_lif_tiles : int, optional
        Maximum LIF tiles to consider. If None, process all configurations.
    visualize : bool, optional
        Whether to visualize groupings. Default True.
    debug : int, optional
        Debug level (0=silent, 1=basic, 2=verbose). Default 1.
    
    Returns
    -------
    dict
        Dictionary mapping LIF_T values to result DataFrames.
    pd.DataFrame or None
        Combined summary DataFrame of all results.
    """
    import pandas as pd
    import copy
    import numpy as np
    import warnings
    
    all_results_by_lif_optimized_lif = {}
    
    # Get number of layers in network (for skipping individual case)
    num_layers = len(network)
    
    # Validate inputs
    if not balanced_results_optimized:
        warnings.warn("balanced_results_optimized is empty")
        return {}, None
    
    valid_lif_t_list = [
        lif_t for lif_t in LIF_T_list 
        if lif_t in balanced_results_optimized 
        and not balanced_results_optimized[lif_t].empty
    ]
    
    if not valid_lif_t_list:
        warnings.warn("No valid LIF_T values found in balanced_results_optimized")
        return {}, None
    
    # Store results for each LIF_T value
    for LIF_T_val in valid_lif_t_list:
        if debug >= 1:
            print(f"\n{'='*100}")
            print(f"PROCESSING LIF_T = {LIF_T_val}")
            print(f"{'='*100}")
        
        # ============================================
        # CREATE MAPPER FOR THIS LIF_T VALUE
        # ============================================
        try:
            mapper = None
            mapper = SNNMapper(
                weights=network,
                layer_groups=[],
                NPE=NPE,
                NT=NT - LIF_T_val,
                X=xbar_size,
                bits_per_cell=XBAR_bits_per_cell,
                P=100,
                Vmem_res=Vmem_res,
                Timestep=Timestep,
                NoC_buswidth=NoC_buswidth,
                NoI_buswidth=NoI_buswidth,
                allow_break_columns=True,
                include_chiplets=False,
                max_chiplets=mesh_cols * mesh_rows
            )
            (
                mapper.tunable_params,
                mapper.xbars,
                mapper.IFMS,
                mapper.OFMS,
                mapper.TOPS,
                mapper.MEMS,
            ) = mapper._calc_tunable_params()
        
            mapper.layer_output_sizes = dict(
                zip(range(1, len(mapper.OFMS) + 1), mapper.OFMS)
            )
            mapper.chiplet_data = mapper._generate_chiplet_mapping()
        except Exception as e:
            warnings.warn(f"Failed to create mapper for LIF_T={LIF_T_val}: {e}")
            all_results_by_lif_optimized_lif[LIF_T_val] = pd.DataFrame()
            continue
        
        # ============================================
        # CREATE OPTIMIZER FOR THIS LIF_T VALUE
        # ============================================
        try:
            optimizer = AMOSAGroupingOptimizer(
                mapper=mapper,
                num_groups=1,
                mesh_rows=mesh_rows,
                mesh_cols=mesh_cols,
                LIF_T=LIF_T_val,
                SRAM_KB_per_tile=SRAM_KB_per_tile,
                NPE=NPE,
                NT=NT,
                inter_cost=inter_cost,
                intra_cost=1,
                archive_size=100,
                initial_temp=500.0,
                final_temp=1,
                cooling_rate=0.98,
                iterations_per_temp=1,
                random_seed=12347,
                verbose_level=0,
            )
        except Exception as e:
            warnings.warn(f"Failed to create optimizer for LIF_T={LIF_T_val}: {e}")
            all_results_by_lif_optimized_lif[LIF_T_val] = pd.DataFrame()
            continue

        # Get the balanced solutions for this LIF_T value
        try:
            balanced_solutions_current = balanced_results_optimized[LIF_T_val]
            
            if balanced_solutions_current.empty:
                if debug >= 1:
                    print(f"  Skipping LIF_T={LIF_T_val}: Empty DataFrame")
                all_results_by_lif_optimized_lif[LIF_T_val] = pd.DataFrame()
                continue
            
            # Ensure index is unique for lookup
            if not balanced_solutions_current.index.is_unique:
                balanced_solutions_current = balanced_solutions_current.reset_index(drop=True)
                
        except Exception as e:
            warnings.warn(f"Failed to get balanced solutions for LIF_T={LIF_T_val}: {e}")
            all_results_by_lif_optimized_lif[LIF_T_val] = pd.DataFrame()
            continue

        # Store results for this LIF_T
        all_results = []

        # Determine last index to check
        try:
            if max_lif_tiles is not None:
                last_index = get_last_index_to_check(balanced_solutions_current, max_lif_tiles)
            else:
                last_index = len(balanced_solutions_current) - 1
            print(f"  Processing up to index {last_index} based on max_lif_tiles={max_lif_tiles}")
        except Exception as e:
            if debug >= 2:
                print(f"  Warning: get_last_index_to_check failed: {e}")
            last_index = len(balanced_solutions_current) - 1

        noc_rows = int(np.sqrt(NT))
        NoC_mesh_layout = [[(r * noc_rows + c) for c in range(noc_rows)] for r in range(noc_rows)]

        # Configuration loop
        for config_idx in range(last_index + 1):
            try:
                # ============================================
                # SKIP INDIVIDUAL LAYER GROUPING CASE
                # ============================================
                original_grouping_num = balanced_solutions_current.iloc[config_idx]['num_groups']
                
                # Skip if num_groups equals number of layers (individual case)
                if original_grouping_num >= num_layers:
                    if debug >= 1:
                        print(f"\n  Skipping config {config_idx}: num_groups={original_grouping_num} "
                              f"(individual layer grouping, num_layers={num_layers})")
                    continue
                
                mapping = balanced_solutions_current.iloc[config_idx]['chiplet_mapping']
                num_chiplets = len(mapping)
                grouping = copy.deepcopy(balanced_solutions_current.iloc[config_idx]['groups'])
                unmapped_mapping = balanced_solutions_current.iloc[config_idx]['unmapped_layers']
                total_lif_tiles_from_df = balanced_solutions_current.iloc[config_idx]['total_lif_tiles']

                # ============================================
                # INITIALIZE max_chiplets_used FROM MAIN MAPPING
                # ============================================
                max_chiplets_used = num_chiplets

                if visualize:
                    try:
                        _ = visualize_groups(grouping, mapping, mesh_rows, mesh_cols, LIF_T=LIF_T_val,
                                            figsize=(8, 6), fig_fontsize=6, legend_fontsize=9, auto_display=True)
                    except Exception as viz_err:
                        if debug >= 2:
                            print(f"  Warning: Visualization failed: {viz_err}")

                if debug >= 1:
                    print(f"\n{'='*80}")
                    print(f"CONFIG {config_idx}: {original_grouping_num} Groups (LIF_T={LIF_T_val}, System={mesh_cols}x{mesh_rows})")
                    print(f"{'='*80}")
                    print(f"Initial chiplets from mapping: {max_chiplets_used}")

                # Initialize accumulators
                first_map_noi_latency_ms = 0.0
                first_map_noc_latency_ms = 0.0
                first_map_noi_power_raw = 0.0

                remap_latencies_ms = []
                remap_NoC_latencies_ms = []
                remap_DRAM_latencies_ms = []
                remap_noi_powers_raw = []
                remap_noc_powers_raw = []
                lif_to_DRAM_packets_per_remap = []
                remap_groups = []

                # Per-layer tracking
                all_layers_noi_latencies = {}
                all_layers_noc_latencies = {}
                all_layers_dram_latencies = {}

                mapped_noi_powers_raw = []
                mapped_noi_latencies_ms = []
                mapped_noc_powers_raw = []
                mapped_noc_latencies_ms = []

                # Get mapped and unmapped layers
                mapped_layers = sorted({layer for chiplet in mapping for layer in chiplet.get('Layers_filled', [])})
                unmapped_layers = list(set(range(1, 1 + len(mapper.weights))) ^ set(mapped_layers))
                LIF_tiles = np.ceil(np.array(mapper.MEMS) / (SRAM_KB_per_tile * 1024)).astype(int)
                unmapped_layers_lif = [LIF_tiles[idx - 1].item() for idx in unmapped_layers]

                NoI_mesh_layout = [[(r * mesh_cols + c) for c in range(mesh_cols)] for r in range(mesh_rows)]
                NoI_mesh_layout_with_DRAM = [[r * mesh_cols + c for c in range(mesh_cols)] for r in range(mesh_rows)] + [[mesh_rows * mesh_cols]]

                if debug >= 1:
                    print(f"Mapped: {len(mapped_layers)} layers {mapped_layers}")
                    print(f"Unmapped: {len(unmapped_layers)} layers {unmapped_layers}\n")

                # ============================================
                # FIRST MAPPING - Process mapped layers
                # ============================================
                if debug >= 1:
                    print(f"FIRST MAPPING")
                    print(f"{'-'*80}")

                for layer_idx, layer_id in enumerate(mapped_layers, 1):
                    try:
                        # Get traffic
                        all_traffic = []
                        traffic_dict = get_layer_traffic(
                            layer_id=layer_id, chiplet_data=mapping, groupings=grouping,
                            weights=mapper.weights, tunable_params=mapper.tunable_params,
                            xbars=mapper.xbars, X=mapper.X, Vmem_res=mapper.Vmem_res,
                            Timestep=Timestep, NoC_buswidth=1,
                            lif_tiles_per_layer=np.ceil(np.array(mapper.MEMS) / (SRAM_KB_per_tile * 1024)).astype(int),
                            SRAM_KB_per_tile=SRAM_KB_per_tile, acc_enabled=False
                        )
                        all_traffic.extend([traffic_dict["output"], traffic_dict["input"]])

                        # NoI traffic
                        _, M_sys_df = create_system_matrix_from_edges(all_traffic, mesh_rows=mesh_rows, mesh_cols=mesh_cols, count_diagonal=True)
                        M_sys_df = np.ceil(M_sys_df / NoI_buswidth).astype(int)
                        M_sys_df, _ = split_top_rest(M_sys_df, percent=percent_keep)

                        # NoC traffic
                        chiplet_dfs = []
                        for c in range(mesh_cols * mesh_rows):
                            M_chip_df = np.ceil(create_tile_matrix_for_chiplet(all_traffic, c, NT, include_chiplets=False)[1] / NoC_buswidth).astype(int)
                            chiplet_dfs.append(M_chip_df)

                        M_chiplet_dfs = [split_top_rest(df, percent=percent_keep)[0] for df in chiplet_dfs]

                        # Scale and simulate
                        chiplet_scaled_dfs, chiplet_scaling_factors, system_scaled_df, system_scaling_factor = scale_traffic_matrices(
                            system_matrix=M_sys_df, chiplet_matrices=M_chiplet_dfs, minimum_traffic=min_traffic)

                        NoC_mesh_layouts = [NoC_mesh_layout] * mesh_cols * mesh_rows
                        noc_latencies_raw, noc_power_raw = run_booksim_NoC(chiplet_scaled_dfs, NoC_mesh_layouts)
                        noc_latencies_ms = (np.array(noc_latencies_raw) * np.array(chiplet_scaling_factors) * NoC_cycle_time).sum()

                        noi_latency_raw, noi_power_raw = run_booksim_NoI(system_scaled_df.values, NoI_mesh_layout)
                        noI_latency_ms = noi_latency_raw * system_scaling_factor * NoI_cycle_time

                        # Accumulate
                        mapped_noc_powers_raw.append(np.array(noc_power_raw).sum())
                        mapped_noc_latencies_ms.append(noc_latencies_ms)
                        first_map_noc_latency_ms += noc_latencies_ms

                        first_map_noi_power_raw += noi_power_raw
                        mapped_noi_powers_raw.append(noi_power_raw)
                        mapped_noi_latencies_ms.append(noI_latency_ms)
                        first_map_noi_latency_ms += noI_latency_ms

                        all_layers_noi_latencies[layer_id] = noI_latency_ms
                        all_layers_noc_latencies[layer_id] = noc_latencies_ms
                        all_layers_dram_latencies[layer_id] = 0.0
                        
                    except Exception as layer_err:
                        warnings.warn(f"Failed to process mapped layer {layer_id}: {layer_err}")
                        all_layers_noi_latencies[layer_id] = 0.0
                        all_layers_noc_latencies[layer_id] = 0.0
                        all_layers_dram_latencies[layer_id] = 0.0

                if debug >= 1:
                    print(f"NoI: {first_map_noi_latency_ms:.4f} ms | NoC: {first_map_noc_latency_ms:.4f} ms\n")

                # ============================================
                # REMAP BLOCK - Process unmapped layers
                # ============================================
                total_remap_lif_tiles = 0

                if len(unmapped_layers) > 0:
                    if debug >= 1:
                        print(f"REMAPPING {len(unmapped_layers)} LAYERS")
                        print(f"{'-'*80}")

                    for remap_idx, unmapped_layer_id in enumerate(unmapped_layers):
                        try:
                            if debug >= 1:
                                print(f"\nLayer {unmapped_layer_id} ({remap_idx + 1}/{len(unmapped_layers)})")

                            unmapped_mapping_current_layer = [sub for sub in unmapped_mapping if any(isinstance(d, dict) \
                                                                                                      and unmapped_layer_id in d.get('Layers_filled', []) for d in sub)][0]

                            # ============================================
                            # UPDATE max_chiplets_used FOR REMAP
                            # ============================================
                            remap_chiplet_count = len(unmapped_mapping_current_layer)
                            if remap_chiplet_count > max_chiplets_used:
                                max_chiplets_used = remap_chiplet_count
                                if debug >= 1:
                                    print(f"  Updated max_chiplets_used: {max_chiplets_used} "
                                          f"(from remap layer {unmapped_layer_id})")

                            unmapped = build_unmapped(
                                mapper.chiplet_data['main_chiplets'],
                                unmapped_mapping_current_layer,
                                mesh_cols * mesh_rows, NT, NPE, remap_idx
                            )

                            available = set(range(mesh_cols * mesh_rows))
                            unavailable = [idx for idx, item in enumerate(unmapped) if item['Crossbars_filled_respective_layer'] == [-1]]
                            available = available ^ set(unavailable)
                            lif_alloc = {chip_id: 0 for chip_id in available}

                            dist, opt_cost = optimizer.find_optimal_lif_placement(
                                unmapped_layer_id, unmapped_layer_id,
                                unmapped_layers_lif[remap_idx],
                                available, lif_alloc
                            )

                            total_remap_lif_tiles += unmapped_layers_lif[remap_idx]

                            this_group = [{
                                'start_layer': unmapped_layer_id,
                                'end_layer': unmapped_layer_id,
                                'total_lif_tiles': unmapped_layers_lif[remap_idx],
                                'lif_distribution': dist,
                                'cost': opt_cost
                            }]

                            remap_chiplet_mapping = add_lif_tiles(unmapped, this_group, NT, LIF_T_val, NPE, mesh_rows, mesh_cols)

                            # PHASE 1: TO LIF (DRAM write)
                            all_traffic_to_lif = []
                            traffic_dict_to_lif = get_layer_traffic(
                                layer_id=unmapped_layer_id, chiplet_data=remap_chiplet_mapping, groupings=this_group,
                                weights=mapper.weights, tunable_params=mapper.tunable_params,
                                xbars=mapper.xbars, X=mapper.X, Vmem_res=mapper.Vmem_res,
                                Timestep=Timestep, NoC_buswidth=1,
                                lif_tiles_per_layer=np.ceil(np.array(mapper.MEMS) / (SRAM_KB_per_tile * 1024)).astype(int),
                                SRAM_KB_per_tile=SRAM_KB_per_tile, acc_enabled=False
                            )
                            all_traffic_to_lif.extend([traffic_dict_to_lif["output"], traffic_dict_to_lif["input"]])

                            _, M_sys_df_to_lif = create_system_matrix_from_edges(
                                all_traffic_to_lif, mesh_rows=mesh_rows + 1, mesh_cols=mesh_cols, count_diagonal=True
                            )

                            dram_packets_to_lif = M_sys_df_to_lif.iloc[-1, :].sum() + M_sys_df_to_lif.iloc[:, -1].sum()

                            M_sys_df_to_lif_no_dram = M_sys_df_to_lif.iloc[:-(mesh_cols - 1), :-(mesh_cols - 1)]
                            M_sys_df_to_lif_no_dram = np.ceil(M_sys_df_to_lif_no_dram / NoI_buswidth).astype(int)
                            M_sys_df_to_lif_no_dram, _ = split_top_rest(M_sys_df_to_lif_no_dram, percent=percent_keep)

                            _, _, system_scaled_df_to_lif, system_scaling_factor_to_lif = scale_traffic_matrices(
                                system_matrix=M_sys_df_to_lif_no_dram, chiplet_matrices=[], minimum_traffic=min_traffic
                            )

                            noi_latency_raw_to_lif, noi_power_raw_to_lif = run_booksim_NoI(
                                system_scaled_df_to_lif.values, NoI_mesh_layout_with_DRAM
                            )
                            noI_latency_ms_to_lif = noi_latency_raw_to_lif * system_scaling_factor_to_lif * NoI_cycle_time
                            DRAM_write_latency_ms = ((dram_packets_to_lif * NoI_buswidth / 8) / (eta * DRAM_BW * 1e9)) * 1e3

                            # PHASE 2: FROM LIF (DRAM read)
                            noI_latency_ms_from_lif = 0.0
                            DRAM_read_latency_ms = 0.0
                            dram_packets_from_lif = 0
                            noi_power_raw_from_lif = 0.0

                            if remap_idx + 1 < len(unmapped_layers):
                                next_layer_id = unmapped_layers[remap_idx + 1]

                                unmapped_mapping_current_layer = [sub for sub in unmapped_mapping if any(isinstance(d, dict) \
                                                                                                          and next_layer_id in d.get('Layers_filled', []) for d in sub)][0]

                                next_unmapped = build_unmapped(
                                    mapper.chiplet_data['main_chiplets'],
                                    unmapped_mapping_current_layer,
                                    mesh_cols * mesh_rows, NT, NPE, remap_idx + 1
                                )

                                next_available = set(range(mesh_cols * mesh_rows))
                                next_unavailable = [idx for idx, item in enumerate(next_unmapped) if item['Crossbars_filled_respective_layer'] == [-1]]
                                next_available = next_available ^ set(next_unavailable)
                                next_lif_alloc = {chip_id: 0 for chip_id in next_available}

                                next_dist, next_opt_cost = optimizer.find_optimal_lif_placement(
                                    next_layer_id, next_layer_id,
                                    unmapped_layers_lif[remap_idx + 1],
                                    next_available, next_lif_alloc
                                )

                                next_group = [{
                                    'start_layer': next_layer_id,
                                    'end_layer': next_layer_id,
                                    'total_lif_tiles': unmapped_layers_lif[remap_idx + 1],
                                    'lif_distribution': next_dist,
                                    'cost': next_opt_cost
                                }]

                                next_remap_chiplet_mapping = add_lif_tiles(
                                    next_unmapped, next_group, NT, LIF_T_val, NPE, mesh_rows, mesh_cols
                                )

                                all_traffic_from_lif = []
                                traffic_dict_from_lif = get_layer_traffic(
                                    layer_id=next_layer_id, chiplet_data=next_remap_chiplet_mapping, groupings=next_group,
                                    weights=mapper.weights, tunable_params=mapper.tunable_params,
                                    xbars=mapper.xbars, X=mapper.X, Vmem_res=mapper.Vmem_res,
                                    Timestep=Timestep, NoC_buswidth=1,
                                    lif_tiles_per_layer=np.ceil(np.array(mapper.MEMS) / (SRAM_KB_per_tile * 1024)).astype(int),
                                    SRAM_KB_per_tile=SRAM_KB_per_tile, acc_enabled=False
                                )
                                all_traffic_from_lif.extend([traffic_dict_from_lif["output"], traffic_dict_from_lif["input"]])

                                _, M_sys_df_from_lif = create_system_matrix_from_edges(
                                    all_traffic_from_lif, mesh_rows=mesh_rows + 1, mesh_cols=mesh_cols, count_diagonal=True
                                )

                                dram_packets_from_lif = M_sys_df_from_lif.iloc[-1, :].sum() + M_sys_df_from_lif.iloc[:, -1].sum()

                                M_sys_df_from_lif_no_dram = M_sys_df_from_lif.iloc[:-(mesh_cols - 1), :-(mesh_cols - 1)]
                                M_sys_df_from_lif_no_dram = np.ceil(M_sys_df_from_lif_no_dram / NoI_buswidth).astype(int)
                                M_sys_df_from_lif_no_dram, _ = split_top_rest(M_sys_df_from_lif_no_dram, percent=percent_keep)

                                _, _, system_scaled_df_from_lif, system_scaling_factor_from_lif = scale_traffic_matrices(
                                    system_matrix=M_sys_df_from_lif_no_dram, chiplet_matrices=[], minimum_traffic=min_traffic
                                )

                                noi_latency_raw_from_lif, noi_power_raw_from_lif = run_booksim_NoI(
                                    system_scaled_df_from_lif.values, NoI_mesh_layout_with_DRAM
                                )
                                noI_latency_ms_from_lif = noi_latency_raw_from_lif * system_scaling_factor_from_lif * NoI_cycle_time
                                DRAM_read_latency_ms = ((dram_packets_from_lif * NoI_buswidth / 8) / (eta * DRAM_BW * 1e9)) * 1e3

                            # NoC Latency
                            _, _, M_sys_df_noc, _chiplet_dfs = calculate_traffic(
                                this_group, remap_chiplet_mapping,
                                mesh_rows=mesh_rows, mesh_cols=mesh_cols,
                                NoI_buswidth=NoI_buswidth, NoC_buswidth=NoC_buswidth,
                                Timestep=Timestep
                            )

                            M_chiplet_dfs = [split_top_rest(df, percent=percent_keep)[0] for df in _chiplet_dfs]

                            chiplet_scaled_dfs, chiplet_scaling_factors, _, _ = scale_traffic_matrices(
                                system_matrix=[], chiplet_matrices=M_chiplet_dfs, minimum_traffic=min_traffic
                            )

                            NoC_mesh_layouts = [NoC_mesh_layout] * len(remap_chiplet_mapping)
                            noc_latencies_raw, noc_power_raw = run_booksim_NoC(chiplet_scaled_dfs, NoC_mesh_layouts)
                            noc_latencies_ms = (np.array(noc_latencies_raw) * np.array(chiplet_scaling_factors) * NoC_cycle_time).sum()

                            # Accumulate
                            total_remap_noi_latency = noI_latency_ms_to_lif + noI_latency_ms_from_lif
                            total_remap_dram_latency = DRAM_write_latency_ms + DRAM_read_latency_ms
                            total_remap_dram_packets = dram_packets_to_lif + dram_packets_from_lif

                            remap_noi_powers_raw.append(noi_power_raw_to_lif + noi_power_raw_from_lif)
                            remap_noc_powers_raw.append(np.array(noc_power_raw).sum())

                            layer_param_dram_latency_ms = ((mapper.tunable_params[unmapped_layer_id - 1] / 8) / (eta * DRAM_BW * 1e9)) * 1e3
                            layer_lif_dram_latency_ms = ((2 * total_remap_dram_packets * NoI_buswidth / 8) / (eta * DRAM_BW * 1e9)) * 1e3
                            total_layer_dram_latency_ms = total_remap_dram_latency + layer_param_dram_latency_ms + layer_lif_dram_latency_ms

                            all_layers_noi_latencies[unmapped_layer_id] = total_remap_noi_latency
                            all_layers_noc_latencies[unmapped_layer_id] = noc_latencies_ms
                            all_layers_dram_latencies[unmapped_layer_id] = total_layer_dram_latency_ms

                            remap_latencies_ms.append(total_remap_noi_latency)
                            remap_NoC_latencies_ms.append(noc_latencies_ms)
                            remap_DRAM_latencies_ms.append(total_remap_dram_latency)
                            lif_to_DRAM_packets_per_remap.append(total_remap_dram_packets)

                            remap_groups.append(this_group[0])

                            if debug >= 1:
                                print(f"  NoI: {total_remap_noi_latency:.4f} | NoC: {noc_latencies_ms:.4f} | DRAM: {total_layer_dram_latency_ms:.4f} ms")
                                
                        except Exception as remap_err:
                            warnings.warn(f"Failed to process remap layer {unmapped_layer_id}: {remap_err}")
                            all_layers_noi_latencies[unmapped_layer_id] = 0.0
                            all_layers_noc_latencies[unmapped_layer_id] = 0.0
                            all_layers_dram_latencies[unmapped_layer_id] = 0.0

                # ============================================
                # LIF TILES & AREA
                # ============================================
                total_lif_tiles = total_lif_tiles_from_df + total_remap_lif_tiles
                total_lif_mem_kb = total_lif_tiles * SRAM_KB_per_tile

                total_compute_tiles = int(np.ceil(sum(mapper.xbars) / NPE))
                total_tiles = total_compute_tiles + total_lif_tiles
                total_area_sq_mm = max_chiplets_used * chiplet_area

                if debug >= 1:
                    print(f"\n{'='*80}")
                    print(f"RESOURCES")
                    print(f"{'-'*80}")
                    print(f"Max Chiplets Used: {max_chiplets_used}")
                    print(f"LIF Tiles: {total_lif_tiles} ({total_lif_tiles_from_df} mapped + {total_remap_lif_tiles} remap)")
                    print(f"LIF Memory: {total_lif_mem_kb:.2f} KB")
                    print(f"Compute Tiles: {total_compute_tiles}")
                    print(f"Total Tiles: {total_tiles}")
                    print(f"Total Area: {total_area_sq_mm:.2f} mm²")

                # ============================================
                # COMPUTATION
                # ============================================
                comp_latency_per_layer_s = np.array(mapper.TOPS) * Timestep / (TOPS * 8)
                comp_latency_per_layer_ms = comp_latency_per_layer_s * 1e3
                comp_energy_per_layer_J = np.array(mapper.TOPS) * Timestep * ENERGY_PER_MAC_pj

                total_comp_latency_ms = comp_latency_per_layer_ms.sum()
                total_comp_energy_J = comp_energy_per_layer_J.sum()

                # ============================================
                # END-TO-END LATENCY
                # ============================================
                mapped_layers_total_time = sum(
                    all_layers_noi_latencies[layer_id] + all_layers_noc_latencies[layer_id] + comp_latency_per_layer_ms[layer_id - 1]
                    for layer_id in mapped_layers
                )

                end_to_end_latency = mapped_layers_total_time
                overlap_penalties = []

                if len(unmapped_layers) > 0:
                    for remap_idx, unmapped_layer_id in enumerate(unmapped_layers):
                        layer_processing_time = (all_layers_noi_latencies[unmapped_layer_id] +
                                                  all_layers_noc_latencies[unmapped_layer_id] +
                                                  comp_latency_per_layer_ms[unmapped_layer_id - 1])

                        layer_total_dram = all_layers_dram_latencies[unmapped_layer_id]

                        if remap_idx == 0:
                            available_time = mapped_layers_total_time
                        else:
                            prev_layer_id = unmapped_layers[remap_idx - 1]
                            available_time = (all_layers_noi_latencies[prev_layer_id] +
                                              all_layers_noc_latencies[prev_layer_id] +
                                              comp_latency_per_layer_ms[prev_layer_id - 1])

                        overlap_penalty = max(0, layer_total_dram - available_time)
                        overlap_penalties.append(overlap_penalty)
                        end_to_end_latency += layer_processing_time + overlap_penalty

                # ============================================
                # COMMUNICATION POWER/ENERGY
                # ============================================
                total_noi_latency = first_map_noi_latency_ms + sum(remap_latencies_ms)
                total_noc_latency = first_map_noc_latency_ms + sum(remap_NoC_latencies_ms)
                total_comm_latency_ms = total_noi_latency + total_noc_latency

                total_noi_power_raw = first_map_noi_power_raw + sum(remap_noi_powers_raw)
                total_noc_power_raw = sum(mapped_noc_powers_raw) + sum(remap_noc_powers_raw)
                comm_power_W = total_noi_power_raw + total_noc_power_raw

                noi_energy_mJ = float(np.sum(np.array(mapped_noi_powers_raw) * np.array(mapped_noi_latencies_ms))) if mapped_noi_powers_raw else 0.0
                noc_energy_mJ = float(np.sum(np.array(mapped_noc_powers_raw) * np.array(mapped_noc_latencies_ms))) if mapped_noc_powers_raw else 0.0

                if len(remap_noi_powers_raw) > 0:
                    noi_energy_mJ += float(np.sum(np.array(remap_noi_powers_raw) * np.array(remap_latencies_ms)))
                    noc_energy_mJ += float(np.sum(np.array(remap_noc_powers_raw) * np.array(remap_NoC_latencies_ms)))

                comm_energy_mJ = noi_energy_mJ + noc_energy_mJ
                end_to_end_energy_mJ = comm_energy_mJ + total_comp_energy_J * 1e3

                # ============================================
                # FINAL SUMMARY
                # ============================================
                tops_per_area = 1e3 / (end_to_end_latency * total_area_sq_mm) if end_to_end_latency > 0 and total_area_sq_mm > 0 else 0

                if debug >= 1:
                    print(f"\n{'='*80}")
                    print(f"FINAL METRICS")
                    print(f"{'-'*80}")
                    print(f"Latency:")
                    print(f"  NoI: {total_noi_latency:.4f} ms")
                    print(f"  NoC: {total_noc_latency:.4f} ms")
                    print(f"  Comm: {total_comm_latency_ms:.4f} ms")
                    print(f"  Comp: {total_comp_latency_ms:.4f} ms")
                    print(f"  DRAM overlap: {sum(overlap_penalties):.4f} ms")
                    print(f"  End-to-end: {end_to_end_latency:.4f} ms")
                    print(f"\nEnergy:")
                    print(f"  Comm: {comm_energy_mJ:.4f} mJ")
                    print(f"  Comp: {total_comp_energy_J * 1e3:.4f} mJ")
                    print(f"  Total: {end_to_end_energy_mJ:.4f} mJ")
                    print(f"\nMetrics:")
                    print(f"  TOPS/Area: {tops_per_area:.6f}")
                    print(f"{'='*80}\n")

                # Store results
                all_results.append({
                    'lif_t': LIF_T_val,
                    'system_size': mesh_rows * mesh_cols,
                    'max_chiplets_used': max_chiplets_used,
                    'LIF_MEM(KB)': total_lif_mem_kb,
                    'LIF_MEM_Tiles': total_lif_tiles,
                    'num_groups': original_grouping_num + len(unmapped_layers),
                    'Total_area_sq_mm': total_area_sq_mm,
                    'NoC_latency_ms': total_noc_latency,
                    'NoI_latency_ms': total_noi_latency,
                    'Total_comm_latency_ms': total_comm_latency_ms,
                    'Comp_Latency_ms': total_comp_latency_ms,
                    'Comm_energy_mJ': comm_energy_mJ,
                    'Comp_energy_mJ': total_comp_energy_J * 1e3,
                    'Comm_power_W': comm_power_W,
                    'End_to_end_latency_ms': end_to_end_latency,
                    'End_to_end_energy_mJ': end_to_end_energy_mJ,
                    'TOPS/Area': tops_per_area,
                    'Groupings': list(grouping) + list(remap_groups),
                    'Chiplet_mapping': mapping
                })
                
            except Exception as config_err:
                warnings.warn(f"Failed to process config {config_idx} for LIF_T={LIF_T_val}: {config_err}")
                continue

        # ============================================
        # SUMMARY for current LIF_T
        # ============================================
        if all_results:
            # if debug >= 1:
            #     print(f"\n{'='*80}")
            #     print(f"SUMMARY OF ALL CONFIGURATIONS FOR LIF_T = {LIF_T_val}")
            #     print(f"{'='*80}\n")

            distil_summary_df_current = pd.DataFrame([
                {k: v for k, v in result.items() if k not in ['Groupings', 'Chiplet_mapping']}
                for result in all_results
            ])

            # if debug >= 1:
            #     try:
            #         display(distil_summary_df_current)
            #     except NameError:
            #         print(distil_summary_df_current.to_string())

            # Store results for this LIF_T value
            all_results_by_lif_optimized_lif[LIF_T_val] = distil_summary_df_current
        else:
            all_results_by_lif_optimized_lif[LIF_T_val] = pd.DataFrame()

    # ============================================
    # FINAL SUMMARY - All LIF_T values together
    # ============================================
    # if debug >= 1:
    #     print(f"\n{'='*100}")
    #     print("FINAL SUMMARY - ALL LIF_T VALUES COMBINED")
    #     print(f"{'='*100}\n")

    # Combine all results into a single DataFrame
    all_combined_results = []
    for lif_t, df in all_results_by_lif_optimized_lif.items():
        if not df.empty:
            df_with_lif = df.copy()
            all_combined_results.append(df_with_lif)

    if all_combined_results:
        final_summary_df = pd.concat(all_combined_results, ignore_index=True)
        
        # if debug >= 1:
        #     try:
        #         display(final_summary_df)
        #     except NameError:
        #         print(final_summary_df.to_string())

        #     print(f"\nResults stored in dictionary (keys = LIF_T values)")
        #     print(f"Available LIF_T values: {list(all_results_by_lif_optimized_lif.keys())}")
    else:
        if debug >= 1:
            print("No results to display")
        final_summary_df = None

    return all_results_by_lif_optimized_lif, final_summary_df
