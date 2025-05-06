import sumolib
import traci
import xml.etree.cElementTree as ET
import numpy as np
import copy
from typing import Dict, List, Union, Tuple

M = 1e5

class DownstreamLane:

    def __init__(self, tls_id: str, lane_id: str):
        self.id = lane_id
        self.signal = tls_id
        self.length = 0
        self.last_halt_position = 1e5     # position of the last halting vehicle along the lane
        self.remain_capacity = 0

    def update_remain_capacity(self, veh_length):
        self.remain_capacity = int(self.last_halt_position / veh_length)


class Movement:

    def __init__(self, in_edge: str, direction: str, movement_type: str, tls_id: str):
        self.id = in_edge + '_' + direction
        self.in_edge = in_edge
        self.signal: str = tls_id
        self.dir = direction
        self.type = movement_type   # inflow / outflow / normal flow
        self.lanes: Dict[str, Lane] = {}
        self.connections: Dict[str, Connection] = {}
        self.stages: List[str] = []

        self.flow_rate = 0
        self.green_start: list = [0]
        self.green_duration: list = [0]
        self.fixed_gated_green = 0
        self.min_green = 0
        self.max_green = 0
        self.yellow_start = [0]  # 用于slot-based
        self.yellow_duration = 0

    def add_stage(self, stage_id: str):
        if stage_id not in self.stages:
            self.stages.append(stage_id)


class Stage:

    def __init__(self, stage_id: str, tls_id: str, movements: Tuple[str]):
        self.id = stage_id
        self.signal = tls_id
        self.movements: Tuple[str] = movements


class Slot:

    def __init__(self, slot_id: str, tls_id: str, ):
        pass


class Connection:

    def __init__(self, tls_id: str, connect_id: str, in_lane: str, to_lane: str, direction: str):
        self.id = connect_id
        self.signal = tls_id
        self.in_lane = in_lane
        self.to_lane = to_lane
        self.dir = direction
        self.movement = None

        self.green_start: list = [0]
        self.green_duration: list = [0]
        self.yellow_start: list = [0]          # 用于slot-based
        self.yellow_duration = 0

    def update_timing(self):
        self.green_duration = copy.deepcopy(self.movement.green_duration)
        self.green_start = copy.deepcopy(self.movement.green_start)
        self.yellow_start = copy.deepcopy(self.movement.yellow_start)


class Lane:

    def __init__(self, lane_id: str, edge: str, tls_id: str):
        # [topology elements]
        self.id = lane_id
        self.edge = edge
        self.signal = tls_id
        self.direction = []
        self.movements: Dict[str, Movement] = {}
        self.downstream_lanes: Dict[str, DownstreamLane] = {}
        self.downstream_edges: List = []
        self.type = ''
        # [traffic parameters]
        self.length = 0
        self.capacity = 0
        self.saturation_flow_rate = 0
        self.vehicle_length = 0  # average vehicle length
        self.saturation_limit = 0
        # [model parameters]
        self.arrival_rate = 0
        self.entered_vehicles = set()  # newly entered vehicles **in the recorded area** in the last interval
        self.queue = 0      # number of vehicle at the end of the interval (used for optimization model)
        self.queueing_vehicles = set()  # vehicles that have already joined the queue **on the lane**
        self.last_interval_max_halting_number = 0   # the maximum number of halting vehicles (derived by traci) in the last interval (used for critical point update)
        self.this_interval_max_halting_number = 0   # the maximum number of halting vehicles in the current interval
        self.last_halt_position = 0     # distance to the stop lane of the halting vehicle at the end of the queue (used for downstream capacity)
        self.virtual_queue_vehicles = set()    # vehicles waiting for joining the queue on adjacent lanes
        self.outflow_vehicle_num = 0    # number of vehicles passing the stop line (derived by induction loops)
        self.estimate_throughput = 0  # the estimated throughput of the last interval (dervied from model)
        # [timing variables and parameters]
        self.green_start: list = [0]
        self.green_duration: list = [0]
        self.min_green = 0
        self.max_green = 0
        self.yellow_duration = 0

    def set_length(self, length: float):
        self.length = length

    def set_capacity(self, capacity: int):
        self.capacity = capacity

    def update_traffic_state(self, step):
        self.arrival_rate = len(self.entered_vehicles) / step
        self.queue = len(self.queueing_vehicles) + len(self.virtual_queue_vehicles)

    def get_downstream_capacity(self):
        capacity = np.min([lane.remain_capacity for lane in self.downstream_lanes.values()])
        # capacity = max(capacity, self.min_green * self.saturation_flow_rate)
        return capacity


class LaneGroup:
    """
    相同进口道的同时放行的车道集合
    """

    def __init__(self, group_id: str, tls_id: str):
        # [topology elements]
        self.id = group_id
        self.signal = tls_id
        self.type = None    # inflow / outflow / normal flow / combined
        self.movements: Dict[str, Movement] = {}
        self.lanes: Dict[str, Lane] = {}
        # [traffic parameters]
        self.length = 0     # 等于车道长度
        self.total_capacity = 0             # 所有车道可容纳车辆数, 与
        self.saturation_flow_rate = 0       # 所有车道的饱和流率之和, veh/h
        self.throughput_upperbound = 0
        self.arrival_rate = 0       # sum of all lanes, veh/s
        self.total_queue = 0        # veh
        self.vehicle_length = 0     # average vehicle length
        self.saturation_limit = 0
        # [model parameters]
        self.optimal_inflow = 0         # 上层模型优化结果, veh/s
        self.this_interval_max_halting_number = 0
        self.last_interval_max_halting_number = 0
        self.critical_queue_ratio = 0
        self.queue_pressure_coef = 0
        self.critical_queue_vehicle = 0
        self.target_queue_vehicle = 0
        self.target_inflow = 0        # veh/s
        self.estimate_throughput = 0
        # [timing variables and parameters]
        self.green_start: list = [0]
        self.green_duration: list = [0]
        self.min_green = 0
        self.max_green = 0
        self.min_inflow = 0
        self.max_inflow = 0
        self.yellow_duration = 0

    def get_main_movement_id(self):
        if len(self.movements) == 1:
            return next(iter(self.movements))
        else:
            assert len(self.movements) == 2
            return next(k for k in self.movements if k[-1] == 's')

    def get_edge_id(self):
        return next(iter(self.lanes)).split('_')[0]

    def get_downstream_capacity(self):
        capacity = min([lane.get_downstream_capacity() for lane in self.lanes.values()]) * len(self.lanes)
        # capacity = sum([lane.get_downstream_capacity() for lane in self.lanes.values()])
        return capacity

    def update_traffic_state(self):
        self.arrival_rate = sum([lane.arrival_rate for lane in self.lanes.values()])
        self.total_queue = sum([lane.queue for lane in self.lanes.values()])
        self.this_interval_max_halting_number = np.mean([lane.this_interval_max_halting_number for lane in self.lanes.values()])
        for lane in self.lanes.values():
            lane.arrival_rate = self.arrival_rate / len(self.lanes)

    def update_target_state(self, cycle, spill_ratio):
        """
        Update the target state before updating the queue coefficient
        """
        # 2025.3.30更新: 在排队消散期间动态调节critical state
        if self.this_interval_max_halting_number >= self.last_interval_max_halting_number:
            # 排队蔓延: 将目标状态设为防溢流状态即可
            self.critical_queue_ratio = spill_ratio
        else:
            # 排队消散: 当优化时刻的排队与之前的critical state接近或更短, 则减小critical queue ratio
            if self.total_queue <= self.critical_queue_ratio * self.total_capacity * 1.1:
                # decrease_ratio = self.this_interval_max_halting_number / self.last_interval_max_halting_number
                decrease_ratio = 0.5
                self.critical_queue_ratio = self.critical_queue_ratio * decrease_ratio
                # self.critical_queue_ratio = 0 if self.critical_queue_ratio < 0.05 else self.critical_queue_ratio

        # 更新目标排队车辆和inflow
        self.critical_queue_vehicle = self.total_capacity * self.critical_queue_ratio
        self.target_inflow = min(self.max_inflow,
                                 self.get_downstream_capacity() / cycle,
                                 self.total_queue / cycle + self.arrival_rate,
                                 max(self.min_inflow,
                                     self.total_queue / cycle + self.arrival_rate - self.critical_queue_vehicle / cycle
                                     )
                                 )
        assert self.target_inflow >= 0
        self.target_queue_vehicle = self.total_queue + self.arrival_rate * cycle - self.target_inflow * cycle
        assert self.target_queue_vehicle >= 0
        if self.target_queue_vehicle == 0:
            self.target_queue_vehicle = 1

    def update_queue_coef(self, control_mode, target_gap, cycle):
        if control_mode == 'PI-Balance':
            if target_gap <= 0.001:
                self.queue_pressure_coef = self.target_queue_vehicle / M
            else:
                # 2025.3.30更新: 动态调节critical point (target_queue_vehicle)
                self.queue_pressure_coef = target_gap * self.target_queue_vehicle / cycle
        elif control_mode in ['PI-Cordon', 'PI']:
            self.queue_pressure_coef = self.target_queue_vehicle / M

    def check_target_state_reached(self, cycle):
        '''
        called after the upper problem solved, with the total inflow larger than metering rate but not reaching the bound
        '''
        estimate_queue_vehicle = self.total_queue + cycle * self.arrival_rate - cycle * self.optimal_inflow
        if abs(self.target_queue_vehicle - estimate_queue_vehicle) <= 3:
            print(f'Lanegroup {self.id} will reach the target state of {self.target_queue_vehicle} vehs. ')
        else:
            print(f'Lanegroup {self.id} targets at {self.target_queue_vehicle} vehs, while actually reach {estimate_queue_vehicle} vehs. ')

    def check_optimal_inflow_implemented(self, cycle):
        '''
        called after the lower problem solved
        Note: might not be satisfied when the queues on each lane in the lane-group are imbalanced
        '''
        lower_throughput = sum([lane.estimate_throughput for lane in self.lanes.values()])
        if abs(self.optimal_inflow * cycle - lower_throughput) <= 5:
            print(f'The signal time plan satisfies the inflow requirements of {lower_throughput} vehs on lanegroup {self.id}. ')
        else:
            print(f'Lanegroup {self.id} requires {self.optimal_inflow * cycle} vehs, while the signal plan discharges {lower_throughput} vehs. ')

    def check_queue_balance(self, cycle):
        '''
        called after the upper problem solver, with the total inflow equal to metering rate or the bound
        '''
        nextstep_queue = self.total_queue + cycle * self.arrival_rate - cycle * self.optimal_inflow
        relative_queue = nextstep_queue / self.target_queue_vehicle
        discharge_vehicle = cycle * self.optimal_inflow
        print(f'Lanegroup {self.id}: relative queue length {relative_queue}, {discharge_vehicle} vehs to be discharged. ')


class Intersection:

    def __init__(self, tls_id: str):
        self.id = tls_id
        self.in_edges: List[str] = []
        self.in_lanes: Dict[str, Lane] = {}
        self.out_lanes: Dict[str, DownstreamLane] = {}
        self.movements: Dict[str, Movement] = {}
        self.connections: Dict[str, Connection] = {}
        self.lane_groups: Dict[str, LaneGroup] = {}
        self.conflict_matrix = {}
        self.cycle = 0

        # Slot-based parameters
        self.stages: Dict[str, Stage] = {}
        self.generalized_cycle_num = 0

    def add_connection(self, new_con: Connection):
        if new_con.id not in self.connections:
            self.connections[new_con.id] = new_con

    def add_movement(self, new_mov: Movement):
        if new_mov.id not in self.movements:
            self.movements[new_mov.id] = new_mov

    def add_stage(self, new_stage: Stage):
        if new_stage.id not in self.stages:
            self.stages[new_stage.id] = new_stage

    def add_edge(self, new_edge: str):
        if new_edge not in self.in_edges:
            self.in_edges.append(new_edge)

    def add_inlane(self, new_lane: Lane, direction: str):
        if new_lane.id not in self.in_lanes:
            self.in_lanes[new_lane.id] = new_lane
        self.in_lanes[new_lane.id].direction.append(direction)

    def add_outlane(self, new_lane: DownstreamLane):
        if new_lane.id not in self.out_lanes:
            self.out_lanes[new_lane.id] = new_lane

    def add_lane_group(self, new_lane_group: LaneGroup):
        if new_lane_group.id not in self.lane_groups:
            self.lane_groups[new_lane_group.id] = new_lane_group

    def set_conflict(self, mov1_id, mov2_id):
        self.conflict_matrix[mov1_id][mov2_id] = 1
        self.conflict_matrix[mov2_id][mov1_id] = 1

    def combine_elements(self, new_lane: Lane, new_movement: Movement,
                         new_connection: Connection, new_downlane: DownstreamLane):
        new_lane_id, new_movement_id = new_lane.id, new_movement.id
        new_connection_id, new_downlane_id = new_connection.id, new_downlane.id
        self.in_lanes[new_lane_id].movements[new_movement_id] = self.movements[new_movement_id]
        self.in_lanes[new_lane_id].downstream_lanes[new_downlane_id] = self.out_lanes[new_downlane_id]
        self.movements[new_movement_id].lanes[new_lane_id] = self.in_lanes[new_lane_id]
        self.connections[new_connection_id].movement = self.movements[new_movement_id]

    def reset(self):
        # [lane measurements]
        for lane in self.in_lanes.values():
            lane.entered_vehicles.clear()
            lane.outflow_vehicle_num = 0
            lane.estimate_throughput = 0
            lane.last_interval_max_halting_number = lane.this_interval_max_halting_number
            lane.this_interval_max_halting_number = 0
        for lane in self.out_lanes.values():
            lane.last_halt_position = lane.length
        for lane_group in self.lane_groups.values():
            lane_group.last_interval_max_halting_number = lane_group.this_interval_max_halting_number
            lane_group.this_interval_max_halting_number = 0
            lane_group.estimate_throughput = 0

        # [signal settings]
        for lane in self.in_lanes.values():
            lane.green_start = [0]
            lane.green_duration = [0]
        for movement in self.movements.values():
            movement.green_start = [0]
            movement.green_duration = [0]
            movement.yellow_start = [0]
        for connection in self.connections.values():
            connection.green_start = [0]
            connection.green_duration = [0]
            connection.yellow_start = [0]


class PeriSignals:

    def __init__(self, net_fp: str, sumo_cmd, config):
        self.peri_info = config['Peri_info']
        self.peri_signals: Dict[str, Intersection] = {tls: Intersection(tls) for tls in self.peri_info}
        self.peri_nodes: List[str] = [tls['node'] for tls in self.peri_info.values()]
        self.peri_inflows: Dict[str, Movement] = {}     # 仅受控方向
        self.peri_lane_groups: Dict[str, LaneGroup] = {}     # 仅受控方向
        self.peri_inflow_lanes_by_laneID: Dict[str, Lane] = {}      # 仅受控方向
        self.peri_inflow_lanes: List[tuple] = []    # 仅受控方向
        self.netfile = net_fp
        self.sumo_cmd = sumo_cmd
        self.phase_mode = config['peri_signal_phase_mode']

        print("SUCCESSFULLY GENERATED PERI INTERSECTION DATA")

    def get_basic_inform(self, config):
        tree = ET.ElementTree(file=self.netfile)
        root = tree.getroot()
        connections = root.findall('connection')
        for connection in connections:
            if 'tl' in connection.attrib:
                tls_id = connection.attrib['tl']
                if tls_id in self.peri_info:
                    peri_signal = self.peri_signals[tls_id]
                    # Get information
                    from_edge = connection.attrib['from']
                    to_edge = connection.attrib['to']
                    from_lane_num = connection.attrib['fromLane']
                    to_lane_num = connection.attrib['toLane']
                    connection_id = connection.attrib['via']
                    connection_dir = connection.attrib['dir']
                    if connection_dir == 't':
                        connection_dir = 'l'  # treat turn-about as left-turn
                    # Add edges
                    peri_signal.add_edge(from_edge)
                    # Add outcoming lanes
                    to_lane_id = '_'.join((to_edge, to_lane_num))
                    new_downstream_lane = DownstreamLane(tls_id=tls_id, lane_id=to_lane_id)
                    peri_signal.add_outlane(new_downstream_lane)
                    # Add incoming lanes
                    from_lane_id = from_edge + '_' + from_lane_num
                    new_income_lane = Lane(from_lane_id, from_edge, tls_id)
                    peri_signal.add_inlane(new_income_lane, connection_dir)
                    # Add movements
                    if config['network'] == 'FullGrid':
                        if int(from_edge) > 100 and int(to_edge) < 100:
                            movement_type = 'inflow'
                        elif int(from_edge) < 100 and int(to_edge) > 100:
                            movement_type = 'outflow'
                        else:
                            movement_type = 'normal flow'
                    else:
                        if int(from_edge) in self.peri_info[tls_id]['external_in_edges'] and int(to_edge) in \
                                self.peri_info[tls_id]['internal_out_edges']:
                            movement_type = 'inflow'
                        elif int(from_edge) in self.peri_info[tls_id]['internal_in_edges'] and int(to_edge) in \
                                self.peri_info[tls_id]['external_out_edges']:
                            movement_type = 'outflow'
                        else:
                            movement_type = 'normal flow'
                    new_movement = Movement(from_edge, connection_dir, movement_type, tls_id)
                    peri_signal.add_movement(new_movement)
                    # Add connections
                    to_lane_id = to_edge + '_' + to_lane_num
                    new_connection = Connection(tls_id=tls_id, connect_id=connection_id, in_lane=from_lane_id,
                                                to_lane=to_lane_id, direction=connection_dir)
                    peri_signal.add_connection(new_connection)
                    # Add the lane/movement to the movement/lane (must add the new lane/movement/connection first)
                    peri_signal.combine_elements(new_lane=new_income_lane,
                                                 new_movement=new_movement,
                                                 new_connection=new_connection,
                                                 new_downlane=new_downstream_lane)

        # Add necessary attributes
        for signal_id, signal in self.peri_signals.items():
            for lane_id, lane in signal.in_lanes.items():
                lane.type = '-'.join(set(mov.type for mov in lane.movements.values()))
                lane.downstream_edges = [down_lane.split('_')[0] for down_lane in lane.downstream_lanes]

        # Add stages for slot-based control
        if self.phase_mode == 'Slot':
            for peri_signal_id, peri_signal in self.peri_signals.items():
                for stage_id, stage_movements in self.peri_info[peri_signal_id]['slot_stage_plan'].items():
                    new_stage = Stage(stage_id, peri_signal_id, stage_movements)
                    peri_signal.add_stage(new_stage)
                    for movement_id in stage_movements:
                        movement = peri_signal.movements[movement_id]
                        movement.add_stage(stage_id)

        # Add basic information of inflow movement
        for peri_signal_id, peri_signal in self.peri_signals.items():
            for movement_id, movement in peri_signal.movements.items():
                if movement.type == 'inflow':
                    self.peri_inflows[movement_id] = movement
                    for lane_id in movement.lanes:
                        if (peri_signal_id, lane_id) not in self.peri_inflow_lanes:
                            self.peri_inflow_lanes.append((peri_signal_id, lane_id))
                            self.peri_inflow_lanes_by_laneID[lane_id] = peri_signal.in_lanes[lane_id]

        # Add traffic parameters
        edges = root.findall('edge')
        for edge in edges:
            if edge.attrib['id'][0] != ':':
                lanes = edge.findall('lane')
                for lane in lanes:
                    lane_id = lane.attrib['id']
                    lane_length = float(lane.attrib['length'])
                    lane_capacity = int(lane_length / config['vehicle_length'])
                    for peri_signal in self.peri_signals.values():
                        if lane_id in peri_signal.in_lanes:
                            peri_signal.in_lanes[lane_id].set_length(lane_length)
                            peri_signal.in_lanes[lane_id].set_capacity(lane_capacity)
                            peri_signal.in_lanes[lane_id].vehicle_length = config['vehicle_length']
                        if lane_id in peri_signal.out_lanes:
                            peri_signal.out_lanes[lane_id].length = lane_length

        # Add timing parameters
        for peri_signal in self.peri_signals.values():
            peri_signal.cycle = config['cycle_time']
            if self.phase_mode == 'Slot':
                peri_signal.generalized_cycle_num = config['slot_merge_cycle_num']
            for movement in peri_signal.movements.values():
                movement.min_green = config['min_green']
                movement.max_green = config['max_green']
                movement.yellow_duration = config['yellow_duration']
                flow_rate = 0
                for lane in movement.lanes.values():
                    flow_rate += lane.saturation_flow_rate / len(lane.movements)
                movement.flow_rate = flow_rate
            for lane in peri_signal.in_lanes.values():
                lane.min_green = config['min_green']
                lane.max_green = config['max_green']
                lane.yellow_duration = config['yellow_duration']
                # TODO: 根据车道类型修改饱和流率
                lane.saturation_flow_rate = config['saturation_flow_rate']
                lane.saturation_limit = config['saturation_limit']
            for connection in peri_signal.connections.values():
                connection.yellow_duration = config['yellow_duration']

        # Add lane-group definitions for inflow edges after the basic elements determined (new)
        # For each peri signal, 8 lane groups under NEMA structure are defined
        # Index naming rules: refer to SPDL (Ma et al., 2022) Fig. 3
        # N-S through: odd edge id with lane 0 and 1; N-S left: odd edge id with lane 2
        # For P11, P21, P31, P41, P51, index of northbound edge > southbound edge
        # E-W through: even edge id with lane 0 and 1; E-W left: even edge id with lane 2
        # For P11, P12, P13, P14, P15, index of eastbound edge > westbound edge
        for tls_id, tls_info in config['Peri_info'].items():
            node_id = tls_info['node']
            # in_edges = self.netdata.node_data[node_id]['incoming']
            in_edges = self.peri_signals[node_id].in_edges
            edge_ns = [int(e) for e in in_edges if int(e) % 2 == 1]
            edge_ew = [int(e) for e in in_edges if int(e) % 2 == 0]
            if node_id in ['P11', 'P12', 'P13', 'P14']:
                edge_info = {'south': max(edge_ns), 'north': min(edge_ns), 'west': max(edge_ew), 'east': min(edge_ew)}
            elif node_id in ['P25', 'P35', 'P45', 'P55']:
                edge_info = {'south': min(edge_ns), 'north': max(edge_ns), 'west': min(edge_ew), 'east': max(edge_ew)}
            elif node_id == 'P15':
                edge_info = {'south': min(edge_ns), 'north': max(edge_ns), 'west': max(edge_ew), 'east': min(edge_ew)}
            else:
                edge_info = {'south': max(edge_ns), 'north': min(edge_ns), 'west': min(edge_ew), 'east': max(edge_ew)}
            # Add each lane group
            for idx, (in_edge_loc, move_dir) in config['lane_group_info'].items():
                if move_dir == 'l':
                    lane_idx_list = [2]
                    if config['network_version'] == 'GridBufferFull1':
                        if tls_id in ['P15', 'P51'] and in_edge_loc in ['west', 'east']:
                            lane_idx_list = []
                        elif tls_id in ['P11', 'P55'] and in_edge_loc in ['north', 'south']:
                            lane_idx_list = []
                elif move_dir == 's':
                    lane_idx_list = [0, 1]
                    if config['network_version'] == 'GridBufferFull1':
                        if tls_id in ['P15', 'P51'] and in_edge_loc in ['west', 'east']:
                            lane_idx_list = [0, 1, 2]
                        elif tls_id in ['P11', 'P55'] and in_edge_loc in ['north', 'south']:
                            lane_idx_list = [0, 1, 2]
                else:
                    raise ValueError('Invalid movement direction')

                group_id = '_'.join((tls_id, str(idx)))
                if len(lane_idx_list) == 0:
                    new_lane_group = LaneGroup(group_id=group_id, tls_id=tls_id)
                    new_lane_group.type = 'virtual'
                    continue

                new_lane_group = LaneGroup(group_id=group_id, tls_id=tls_id)
                for lane_idx in lane_idx_list:
                    lane_id = '_'.join((str(edge_info[in_edge_loc]), str(lane_idx)))
                    lane = self.peri_signals[tls_id].in_lanes[lane_id]
                    new_lane_group.lanes[lane_id] = lane
                    for movement_id, movement in lane.movements.items():
                        if movement_id not in new_lane_group.movements:
                            new_lane_group.movements[movement_id] = movement
                            if new_lane_group.type is None:
                                new_lane_group.type = movement.type
                            else:   # 直行右转车道组
                                if movement.dir == 's':
                                    new_lane_group.type = movement.type

                # Add parameters for the lane group
                new_lane_group.length = np.mean([lane.length for lane in new_lane_group.lanes.values()])
                new_lane_group.total_capacity = sum([lane.capacity for lane in new_lane_group.lanes.values()])
                new_lane_group.saturation_flow_rate = sum(
                    [lane.saturation_flow_rate for lane in new_lane_group.lanes.values()])
                new_lane_group.saturation_limit = config['saturation_limit']
                new_lane_group.yellow_duration = config['yellow_duration']
                new_lane_group.min_green = config['min_green']
                new_lane_group.max_green = config['max_green']
                new_lane_group.min_inflow = config['min_green'] / config['cycle_time'] * sum(
                    [lane.saturation_flow_rate for lane in new_lane_group.lanes.values()])
                new_lane_group.max_inflow = config['max_green'] / config['cycle_time'] * sum(
                    [lane.saturation_flow_rate for lane in new_lane_group.lanes.values()])
                new_lane_group.vehicle_length = config['vehicle_length']
                new_lane_group.throughput_upperbound = new_lane_group.max_green * new_lane_group.saturation_flow_rate
                new_lane_group.critical_queue_ratio = config['spillover_critical_ratio']
                # new_lane_group.critical_queue_vehicle = new_lane_group.total_capacity * config['spillover_critical_ratio']
                self.peri_signals[tls_id].add_lane_group(new_lane_group)
                if 'inflow' in new_lane_group.type:
                    self.peri_lane_groups[group_id] = new_lane_group

        # Update perimeter info for config file
        config['perimeter_lane_number'] = sum([len(lane_group.lanes) for lane_group in self.peri_lane_groups.values()])
        config['network_maximal_inflow'] = config['saturation_flow_rate'] * config['max_green'] / config['cycle_time'] * \
                                           config['perimeter_lane_number']
        config['network_minimal_inflow'] = config['saturation_flow_rate'] * config['min_green'] / config['cycle_time'] * \
                                           config['perimeter_lane_number']

    def get_all_lanes(self):
        peri_lanes = []
        for peri_signal_id, peri_signal in self.peri_signals.items():
            for lane_id in peri_signal.in_lanes:
                peri_lanes.append((peri_signal_id, lane_id))
        return peri_lanes

    def get_inflow_green_duration(self):
        green_time_dict = {}
        for peri_signal_id, peri_signal in self.peri_signals.items():
            for movement_id, movement in peri_signal.movements.items():
                if movement.type == 'inflow' and movement_id[-1] != 'r':    # 排除右转
                    signal_movement = '_'.join((peri_signal_id, movement_id))
                    green = np.around(sum(movement.green_duration))
                    green_time_dict[signal_movement] = green
        return green_time_dict

    def get_conflict_matrix(self):

        traci.start(self.sumo_cmd)

        for peri_signal in self.peri_signals.values():
            # Initialize the matrix
            for mov1 in peri_signal.movements:
                peri_signal.conflict_matrix[mov1] = {}
                for mov2 in peri_signal.movements:
                    peri_signal.conflict_matrix[mov1][mov2] = 0
            # Add conflict relationship
            for connection_id, connection in peri_signal.connections.items():
                movement_id = connection.movement.id
                # Find the conflict connection
                conflict_cons_id = traci.lane.getInternalFoes(laneID=connection_id)
                for conflict_con_id in conflict_cons_id:
                    if conflict_con_id in peri_signal.connections:
                        conflict_connection = peri_signal.connections[conflict_con_id]
                        # Check if connection 2 is conflict with connection 1 in traditional concept
                        true_conflict = 1
                        if conflict_connection.movement.in_edge == connection.movement.in_edge:
                            true_conflict = 0  # Movements on the same edge
                        if conflict_connection.movement.dir == 'l' and connection.movement.dir == 'r':
                            true_conflict = 0  # Right-turn and left-turn movements to the same edge
                        if conflict_connection.movement.dir == 'r' and connection.movement.dir == 'l':
                            true_conflict = 0  # Right-turn and left-turn movements to the same edge
                        if true_conflict:
                            conflict_movement_id = conflict_connection.movement.id
                            peri_signal.set_conflict(movement_id, conflict_movement_id)

        traci.close()

    def set_normalized_coef(self):
        for lane_group in self.peri_lane_groups.values():
            lane_group.queue_pressure_coef = lane_group.target_queue_vehicle / M

    def check_conflict_matrix(self):

        error = 0
        for peri_signal in self.peri_signals.values():
            matrix = peri_signal.conflict_matrix
            movements = peri_signal.movements
            for mov1 in movements:
                for mov2 in movements:
                    if matrix[mov1][mov2] != matrix[mov2][mov1]:
                        error += 1
                        # print(f'Error with {mov1} and {mov2}')

        if error > 0:
            print(f'error count: {error}')

        return error

    def plan2program(self):

        peri_program_dict = {}

        for signal_id, signal in self.peri_signals.items():
            program_dict = {}
            current_state, current_state_duration = '', 1

            if self.phase_mode == 'Slot':
                cycle_length = signal.cycle * signal.generalized_cycle_num
            else:
                cycle_length = signal.cycle
            for i in range(cycle_length):
                green_idx, yellow_idx = [], []
                for connection_id, connection in signal.connections.items():
                    inform = connection_id.split('_')
                    idx = int(inform[-1]) + int(inform[-2])
                    yellow = connection.yellow_duration
                    for (start, duration) in zip(connection.green_start, connection.green_duration):
                        if start <= i < start + duration:
                            green_idx.append(idx)
                        elif start + duration <= i < start + duration + yellow:
                            yellow_idx.append(idx)
                new_state = ''
                for j in range(len(signal.connections)):
                    if j in green_idx:
                        new_state += 'G'
                    elif j in yellow_idx:
                        new_state += 'y'
                    else:
                        new_state += 'r'
                if new_state == current_state:
                    current_state_duration += 1
                else:
                    if i > 0:  # Do not update at the beginning
                        if current_state in program_dict:
                            program_dict[current_state] += current_state_duration
                        else:
                            program_dict[current_state] = current_state_duration
                    current_state, current_state_duration = new_state, 1
            if current_state in program_dict:  # Final state
                program_dict[current_state] += current_state_duration
            else:
                program_dict[current_state] = current_state_duration
            peri_program_dict[signal_id] = program_dict

        return peri_program_dict

    def print_program(self):
        for signal_id, signal in self.peri_signals.items():
            if self.phase_mode == 'Slot':
                print(f'Signal plan of signal {signal_id}:')
                for movement_id, movement in signal.movements.items():
                    green_start, green_dur = movement.green_start, movement.green_duration
                    green_end = green_start + green_dur
                    print(f'Movement {movement_id} green starts at {green_start} and ends at {green_end}.')
            else:
                print('To be implemented')

    def print_green_time(self, signal_list: list = None):
        if signal_list is None:
            signal_list = list(self.peri_signals)
        for signal_id in signal_list:
            signal = self.peri_signals[signal_id]
            print(f'Signal {signal_id}: ')
            for lanegroup_id, lanegroup in signal.lane_groups.items():
                # print(f'Queue length of lane group {lanegroup_id}: {lanegroup.total_queue} veh. ')
                # print(f'Inflow of lane group {lanegroup_id}: {lanegroup.arrival_rate} veh/s. ')
                if isinstance(lanegroup.green_duration, list):
                    print(f'Green duration of lane group {lanegroup_id}: {lanegroup.green_duration[0]}s. ')
                else:
                    print(f'Green duration of lane group {lanegroup_id}: {lanegroup.green_duration}s. ')

    def reset_green_time(self):
        '''
        Used for MP algorithm: reset the green duration at the start of the new interval
        '''
        for signal_id, signal in self.peri_signals.items():
            for lanegroup_id, lanegroup in signal.lane_groups.items():
                lanegroup.green_duration = [0]

    def update_green_from_phase_index(self, signal_id, phase_num: int, green_time: int, config):
        '''
        Used for MP algorithm: update the green time according to the given SIGNAL dict
        '''
        signal = self.peri_signals[signal_id]
        if phase_num == 0:
            if config['network_version'] == 'GridBufferFull1' and signal_id in ['P11', 'P55']:
                discharge_lanegroup = [8]
            else:
                discharge_lanegroup = [3, 8]
        elif phase_num == 1:
            if config['network_version'] == 'GridBufferFull1' and signal_id in ['P15', 'P51']:
                discharge_lanegroup = [2]
            else:
                discharge_lanegroup = [2, 5]
        elif phase_num == 2:
            if config['network_version'] == 'GridBufferFull1' and signal_id in ['P15', 'P51']:
                discharge_lanegroup = [6]
            else:
                discharge_lanegroup = [1, 6]
        elif phase_num == 3:
            if config['network_version'] == 'GridBufferFull1' and signal_id in ['P11', 'P55']:
                discharge_lanegroup = [4]
            else:
                discharge_lanegroup = [4, 7]
        elif phase_num == 4:
            discharge_lanegroup = [4, 8]
        elif phase_num == 5:
            discharge_lanegroup = [2, 6]
        elif phase_num == 6:
            if config['network_version'] == 'GridBufferFull1' and signal_id in ['P11', 'P55']:
                discharge_lanegroup = [1, 5]
            else:
                discharge_lanegroup = [3, 7]
        elif phase_num == 7:
            discharge_lanegroup = [1, 5]
        else:
            raise NotImplementedError
        for lane_group_idx in discharge_lanegroup:
            lane_group = signal.lane_groups[signal_id + '_' + str(lane_group_idx)]
            lane_group.green_duration[0] += green_time
            for lane_id, lane in lane_group.lanes.items():
                lane.green_duration[0] += green_time

