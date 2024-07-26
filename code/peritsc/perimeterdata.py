import sumolib
import traci
from utils.utilize import config
import xml.etree.cElementTree as ET
import numpy as np
import copy
from typing import Dict, List, Union, Tuple


class DownstreamLink:

    def __init__(self, tls_id: str, link_id: str):
        self.id = link_id
        self.signal = tls_id
        self.total_capacity = 0
        self.queue = 0
        self.remain_capacity = 0

    def set_total_capacity(self, capacity: int):
        self.total_capacity = capacity

    def add_capacity(self, capacity: int):
        self.total_capacity += capacity

    def update_state(self, queue: int):
        self.queue = queue
        self.remain_capacity = self.total_capacity - self.queue

    def get_remain_capacity(self):
        return self.remain_capacity


class Movement:

    def __init__(self, in_edge: str, direction: str, movement_type: str, tls_id: str):
        self.id = in_edge + '_' + direction
        self.in_edge = in_edge
        self.signal: str = tls_id
        self.dir = direction
        self.type = movement_type   # inflow / outflow / normal flow
        self.lanes: Dict[str, Lane] = {}
        self.connections: Dict[str, Connection] = {}
        self.down_link: DownstreamLink = None
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
        self.id = lane_id
        self.edge = edge
        self.signal = tls_id
        self.direction = []
        self.movements: Dict[str, Movement] = {}

        self.length = 0
        self.capacity = 0
        self.saturation_flow_rate = 0
        self.vehicle_length = 0  # average vehicle length
        self.saturation_limit = 0

        self.arrival_vehicle = 0    # number of vehicle entered within the last interval
        self.arrival_rate = 0
        self.queue = 0      # number of vehicle at the end of the interval
        self.laststep_vehicles = []  # all the vehicles on the lane at the last step
        self.queueing_vehicles = []     # vehicles that have already joined the queue
        self.entered_vehicles = []      # newly entered vehicles in the last interval

        self.green_start: list = [0]
        self.green_duration: list = [0]
        self.min_green = 0
        self.max_green = 0
        self.yellow_duration = 0

    def add_direction(self, direction: str):
        if direction not in self.direction:
            self.direction.append(str)

    def set_length(self, length: float):
        self.length = length

    def set_capacity(self, capacity: int):
        self.capacity = capacity

    def update_traffic_state(self):
        self.arrival_rate = len(self.entered_vehicles) / config['infostep']
        self.entered_vehicles = []
        self.queue = len(self.queueing_vehicles)

    def set_vehicle_length(self, vehicle_length: float):
        self.vehicle_length = vehicle_length

    def get_downstream_capacity(self):
        min_cap_movement, capacity = '', 1e7
        for movement_id, movement in self.movements.items():
            down_link = movement.down_link
            down_link_capacity = down_link.get_remain_capacity()
            if down_link_capacity < capacity:
                min_cap_movement = movement_id
                capacity = down_link_capacity
        return capacity


class LaneGroup:
    """
    相同进口道的同时放行的车道集合
    """

    def __init__(self, group_id: str, tls_id: str):
        self.id = group_id      # 暂定与edge_id相同
        self.signal = tls_id
        self.movements: Dict[str, Movement] = {}
        self.lanes: Dict[str, Lane] = {}

        self.length = 0     # 所在edge的长度
        self.total_capacity = 0             # 所有车道可容纳车辆数, 与
        self.saturation_flow_rate = 0       # 所有车道的饱和流率之和
        self.throughput_upperbound = 0
        self.arrival_rate = 0       # 所有车道的到达率之和
        self.total_queue = 0        # 所有车道的排队车辆数之和
        self.vehicle_length = 0     # average vehicle length
        self.saturation_limit = 0
        self.queue_pressure_coef = 0

        self.green_start: list = [0]
        self.green_duration: list = [0]
        self.min_green = 0
        self.max_green = 0
        self.yellow_duration = 0

    def get_downstream_capacity(self):
        min_cap_movement, capacity = '', 1e7
        for movement_id, movement in self.movements.items():
            down_link = movement.down_link
            down_link_capacity = down_link.get_remain_capacity()
            if down_link_capacity < capacity:
                capacity = down_link_capacity
        return capacity


class Intersection:

    def __init__(self, tls_id: str):
        self.id = tls_id
        self.in_edges: List[str] = []
        self.in_lanes: Dict[str, Lane] = {}
        self.movements: Dict[str, Movement] = {}
        self.connections: Dict[str, Connection] = {}
        self.conflict_matrix = {}
        self.downLinks: Dict[str, DownstreamLink] = {}
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

    def add_lane(self, new_lane: Lane):
        if new_lane.id not in self.in_lanes:
            self.in_lanes[new_lane.id] = new_lane

    def add_downstream_link(self, new_downstream_link: DownstreamLink):
        if new_downstream_link.id not in self.downLinks:
            self.downLinks[new_downstream_link.id] = new_downstream_link

    def set_conflict(self, mov1_id, mov2_id):
        self.conflict_matrix[mov1_id][mov2_id] = 1
        self.conflict_matrix[mov2_id][mov1_id] = 1

    def combine_elements(self, new_lane: Lane, new_movement: Movement,
                         new_connection: Connection, new_downlink: DownstreamLink):
        new_lane_id, new_movement_id = new_lane.id, new_movement.id
        new_connection_id, new_downlink_id = new_connection.id, new_downlink.id
        self.in_lanes[new_lane_id].movements[new_movement_id] = self.movements[new_movement_id]
        self.movements[new_movement_id].lanes[new_lane_id] = self.in_lanes[new_lane_id]
        self.connections[new_connection_id].movement = self.movements[new_movement_id]
        self.movements[new_movement_id].down_link = self.downLinks[new_downlink_id]

    def reset_signal_settings(self):
        for lane in self.in_lanes.values():
            lane.green_start = []
            lane.green_duration = []
        for movement in self.movements.values():
            movement.green_start = []
            movement.green_duration = []
            movement.yellow_start = []
        for connection in self.connections.values():
            connection.green_start = []
            connection.green_duration = []
            connection.yellow_start = []


class PeriSignals:

    def __init__(self, net_fp, sumo_cmd):
        self.peri_info = config['Peri_info']
        self.peri_signals: Dict[str, Intersection] = {tls: Intersection(tls) for tls in self.peri_info}
        self.peri_nodes: List[str] = [tls['node'] for tls in self.peri_info.values()]
        self.peri_inflows: Dict[str, Movement] = {}     # 仅受控方向
        self.peri_edges: Dict[str, LaneGroup] = {}      # 仅受控方向, lanegroup -> edge
        self.peri_inflow_lanes_by_laneID: Dict[str, Lane] = {}      # 仅受控方向
        self.peri_inflow_lanes: List[tuple] = []    # 仅受控方向
        self.peri_downstream_links: Dict[str, DownstreamLink] = {}
        self.netfile = net_fp
        self.sumo_cmd = sumo_cmd
        self.phase_mode = config['peri_signal_phase_mode']

        print("SUCCESSFULLY GENERATED PERI INTERSECTION DATA")

    def get_basic_inform(self):
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
                    # Add downstream links
                    new_downstream_link = DownstreamLink(tls_id=tls_id, link_id=to_edge)
                    peri_signal.add_downstream_link(new_downstream_link)
                    # Add lanes
                    from_lane_id = from_edge + '_' + from_lane_num
                    new_lane = Lane(from_lane_id, from_edge, tls_id)
                    new_lane.add_direction(connection_dir)
                    peri_signal.add_lane(new_lane)
                    # Add movements
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
                    peri_signal.combine_elements(new_lane=new_lane,
                                                 new_movement=new_movement,
                                                 new_connection=new_connection,
                                                 new_downlink=new_downstream_link)

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
            for downstream_link_id, downstream_link in peri_signal.downLinks.items():
                self.peri_downstream_links[downstream_link_id] = downstream_link
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
                edge_id = edge.attrib['id']
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
                    if edge_id in self.peri_downstream_links:
                        self.peri_downstream_links[edge.attrib['id']].add_capacity(lane_capacity)

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
                lane.saturation_flow_rate = config['saturation_flow_rate']
                lane.saturation_limit = config['saturation_limit']
            for connection in peri_signal.connections.values():
                connection.yellow_duration = config['yellow_duration']

        # Add lane-group definitions for inflow edges after the basic elements determined
        for tls_id, tls_info in config['Peri_info'].items():
            inflow_edge_id = str(tls_info['edge'])
            new_lane_group = LaneGroup(group_id=inflow_edge_id, tls_id=tls_id)
            # Add lanes and movements
            for lane_id, lane in self.peri_signals[tls_id].in_lanes.items():
                if lane.edge == inflow_edge_id:
                    new_lane_group.lanes[lane_id] = lane
                    for movement_id, movement in lane.movements.items():
                        if movement_id not in new_lane_group.movements:
                            new_lane_group.movements[movement_id] = movement
            # Add parameters
            new_lane_group.length = np.mean([lane.length for lane in new_lane_group.lanes.values()])
            new_lane_group.total_capacity = sum([lane.capacity for lane in new_lane_group.lanes.values()])
            new_lane_group.saturation_flow_rate = sum([lane.saturation_flow_rate for lane in new_lane_group.lanes.values()])
            new_lane_group.saturation_limit = config['saturation_limit']
            new_lane_group.yellow_duration = config['yellow_duration']
            new_lane_group.min_green = config['min_green']
            new_lane_group.max_green = config['max_green']
            new_lane_group.vehicle_length = config['vehicle_length']
            new_lane_group.throughput_upperbound = new_lane_group.max_green * new_lane_group.saturation_flow_rate
            new_lane_group.queue_pressure_coef = 1 / (2 * new_lane_group.total_capacity * config['spillover_critical_ratio'])
            self.peri_edges[inflow_edge_id] = new_lane_group

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