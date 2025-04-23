import gurobipy as gp
from utils.utilize import config, set_sumo
from peritsc.perimeterdata import PeriSignals
from typing import Dict, List


def constr_name_attach(*args, join_char: str = '_'):
    return join_char.join(args)


class WebsterController:
    """

    """
    def __init__(self, peri_data: PeriSignals):
        self.peri_data = peri_data

    def signal_optimize(self):

        for signal_id, signal in self.peri_data.peri_signals.items():
            real_arrival_rate = {}
            for lane_group_id, lane_group in signal.lane_groups.items():
                real_arrival_rate[lane_group_id] = lane_group.total_queue / signal.cycle + lane_group.arrival_rate
            # assume that the vehicles on each arm discharge together
            vc_ratio = {}
            arm_lanegroup_dict = {'south': (3, 8),
                                  'north': (4, 7),
                                  'west': (2, 5),
                                  'east': (1, 6)}
            for arm, lane_groups in arm_lanegroup_dict.items():
                volume = sum(real_arrival_rate['_'.join((signal_id, str(lg_id)))] for lg_id in lane_groups)
                capacity = sum(signal.lane_groups['_'.join((signal_id, str(lg_id)))].saturation_flow_rate for lg_id in lane_groups)
                vc_ratio[arm] = volume / capacity
            # green time proportional to the vc_ratio
            total_green_time = signal.cycle - config['yellow_duration'] * 4 - config['min_green'] * 4
            if sum(vc_ratio.values()) == 0:
                green_time = {arm: int(total_green_time / 4) + config['min_green'] for arm in arm_lanegroup_dict}
                total_green_diff = sum(green_time.values()) - total_green_time - config['min_green'] * 4
                green_time['east'] -= total_green_diff
            else:
                green_ratio = {arm: vc_ratio[arm] / sum(vc_ratio.values()) for arm in arm_lanegroup_dict}
                green_time = {arm: int(green_ratio[arm] * total_green_time) + config['min_green'] for arm in arm_lanegroup_dict}
                # check the maximum green time
                excess_time, valid_directions = 0, []
                for arm, green_duration in green_time.items():
                    if green_duration > config['max_green']:
                        excess_time += green_duration - config['max_green']
                        green_time[arm] = config['max_green']
                    else:
                        valid_directions.append(arm)
                total_valid_green_ratio = sum(green_ratio[arm] for arm in valid_directions)
                if total_valid_green_ratio == 0:
                    for arm in valid_directions:
                        green_time[arm] += int(excess_time / len(valid_directions))
                else:
                    for arm in valid_directions:
                        green_time[arm] += int(excess_time * green_ratio[arm] / total_valid_green_ratio)
            # final check
            total_green_diff = sum(green_time.values()) - total_green_time - config['min_green'] * 4
            green_time['east'] -= total_green_diff
            # phase sequence: discharge the arm with the highest vc_ratio first
            phase_sequence = sorted(vc_ratio, key=vc_ratio.get, reverse=True)
            # set the green time to peri_data
            green_start = 0
            for arm in phase_sequence:
                green_duration = green_time[arm]
                for lane_group_idx in arm_lanegroup_dict[arm]:
                    lane_group_id = '_'.join((signal_id, str(lane_group_idx)))
                    lane_group = signal.lane_groups[lane_group_id]
                    lane_group.green_start = green_start
                    lane_group.green_duration = green_duration
                    for movement_id, movement in lane_group.movements.items():
                        movement.green_start = [green_start]
                        movement.green_duration = [green_duration]
                    for lane_id, lane in lane_group.lanes.items():
                        lane.green_start = [green_start]
                        lane.green_duration = [green_duration]
                green_start += green_duration + config['yellow_duration']
            for connection in signal.connections.values():
                connection.update_timing()

        return 0


