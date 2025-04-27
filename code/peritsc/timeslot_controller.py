from peritsc.signal_controller import PeriSignalController
from peritsc.perimeterdata import PeriSignals
import gurobipy as gp
import numpy as np
from typing import Dict, List

def constr_name_attach(*args, join_char: str = '_'):
    return join_char.join(args)


class TimeSlotPeriSignalController(PeriSignalController):

    def __init__(self, peri_data: PeriSignals, action: float, action_bound: float,
                 config: dict, slot_num: int):
        super(TimeSlotPeriSignalController, self).__init__(peri_data, action, action_bound, config)
        self.time_slot_num = slot_num

    # 重写分布式各交叉口slot-based配时确定方法
    def set_local_green(self):
        '''
        输入：无（交叉口状态）
        固定变量: inflow方向绿灯时长
        决策变量：所有方向绿灯时长+绿灯启亮（+相序）
        目标：最大化吞吐量
        '''
        cycle, yellow = self.config['cycle_time'], self.config['yellow_duration']
        M = 1e5
        perimeter_total_throughput = 0  # 返回值

        for signal_id, signal in self.peri_data.peri_signals.items():

            stage_num = len(self.peri_data.peri_info[signal_id]['slot_stage_plan'])

            # 对每个交叉口分别优化信号配时
            m = gp.Model('lower-signal-time-slot')
            m.setParam('OutputFlag', 0)
            m.setParam(gp.ParamConstClass.DualReductions, 0)

            # Variables: queue length at the next step of each lane;
            # green start of each lane/movement; green duration of each lane/movement
            slot_green_start: List[gp.Var] = []  # green start time of each slot
            slot_green_duration: List[gp.Var] = []  # green duration of each slot
            slot_stage_selection: Dict[str, List[gp.Var]] = {}
            slot_movement_selection: Dict[str, List[gp.Var]] = {}
            succ_slot_movement_selection: Dict[str, List[gp.Var]] = {}
            slot_movement_no_merge: Dict[str, List[gp.Var]] = {}
            slot_movement_green_duration: Dict[str, List[gp.Var]] = {}
            slot_lane_selection: Dict[str, List[gp.Var]] = {}
            slot_lane_queue: Dict[str, List[gp.Var]] = {}  # The lane queue at the end of each time slot
            slot_lane_throughput: Dict[str, List[gp.Var]] = {}  # The real throughput within each time slot
            slot_lane_green_throughput: Dict[str, List[gp.Var]] = {}
            slot_lane_green_demand: Dict[str, List[gp.Var]] = {}
            slot_lane_green_sfr: Dict[str, List[gp.Var]] = {}
            slot_lane_green_duration: Dict[str, List[gp.Var]] = {}
            lane_throughput: Dict[str, gp.Var] = {}

            # Add variables to each container
            for i in range(self.time_slot_num):
                var_slot_green_start = m.addVar(lb=0, name=constr_name_attach('slot' + str(i), 'green_start'))
                var_slot_green_dur = m.addVar(lb=0, name=constr_name_attach('slot' + str(i), 'green_duration'))
                slot_green_start.append(var_slot_green_start)
                slot_green_duration.append(var_slot_green_dur)
                for stage_id in signal.stages.keys():
                    var_slot_stage = m.addVar(vtype=gp.GRB.BINARY,
                                              name=constr_name_attach('slot' + str(i), 'stage'))
                    slot_stage_selection.setdefault(stage_id, []).append(var_slot_stage)
                for movement_id in signal.movements.keys():
                    var_slot_movement = m.addVar(vtype=gp.GRB.BINARY,
                                                 name=constr_name_attach('slot' + str(i), movement_id, 'movement'))
                    var_succ_slot_movement = m.addVar(vtype=gp.GRB.BINARY,
                                                      name=constr_name_attach('slot' + str(i), movement_id, 'succ'))
                    var_slot_movement_no_merge = m.addVar(vtype=gp.GRB.BINARY,
                                                          name=constr_name_attach('slot' + str(i), movement_id,
                                                                                  'merge'))
                    var_slot_movement_green_dur = m.addVar(lb=0,
                                                           name=constr_name_attach('slot' + str(i), movement_id,
                                                                                   'green'))
                    slot_movement_selection.setdefault(movement_id, []).append(var_slot_movement)
                    succ_slot_movement_selection.setdefault(movement_id, []).append(var_succ_slot_movement)
                    slot_movement_no_merge.setdefault(movement_id, []).append(var_slot_movement_no_merge)
                    slot_movement_green_duration.setdefault(movement_id, []).append(var_slot_movement_green_dur)
                for lane_id in signal.in_lanes.keys():
                    var_slot_lane_selection = m.addVar(vtype=gp.GRB.BINARY,
                                                       name=constr_name_attach('slot' + str(i), lane_id, 'lane'))
                    var_slot_queue = m.addVar(lb=0, name=constr_name_attach('slot' + str(i), lane_id, 'queue'))
                    var_slot_throughput = m.addVar(lb=0, name=constr_name_attach('slot' + str(i), lane_id,
                                                                                 'throughput'))  # 真实output
                    var_slot_green_throughput = m.addVar(
                        lb=0, name=constr_name_attach('slot' + str(i), lane_id, 'green_throughput'))  # 如果是绿灯的output
                    var_slot_green_demand = m.addVar(lb=0, name=constr_name_attach('slot' + str(i), lane_id,
                                                                                   'green_demand'))
                    var_slot_green_sfr = m.addVar(lb=0,
                                                  name=constr_name_attach('slot' + str(i), lane_id, 'green_sfr'))
                    var_slot_lane_green_dur = m.addVar(lb=0,
                                                       name=constr_name_attach('slot' + str(i), lane_id, 'green'))
                    slot_lane_selection.setdefault(lane_id, []).append(var_slot_lane_selection)
                    slot_lane_queue.setdefault(lane_id, []).append(var_slot_queue)
                    slot_lane_throughput.setdefault(lane_id, []).append(var_slot_throughput)
                    slot_lane_green_throughput.setdefault(lane_id, []).append(var_slot_green_throughput)
                    slot_lane_green_demand.setdefault(lane_id, []).append(var_slot_green_demand)
                    slot_lane_green_sfr.setdefault(lane_id, []).append(var_slot_green_sfr)
                    slot_lane_green_duration.setdefault(lane_id, []).append(var_slot_lane_green_dur)
            for lane_id in signal.in_lanes.keys():
                var_throughput = m.addVar(lb=0, name=constr_name_attach(lane_id, 'throughput'))
                lane_throughput[lane_id] = var_throughput
            # Other variables
            total_throughput = m.addVar(lb=0, name='total_thrp')

            # obj: Maximize the (weighted) normalized throughput
            obj = 0
            m.addConstr(total_throughput == gp.quicksum(lane_throughput[lane_id] for lane_id in signal.in_lanes),
                        name='cal_thrp')
            obj -= total_throughput
            m.setObjective(obj)

            # Add constraints
            # Constraint 1: Queue length at the end of each time slot
            for i in range(self.time_slot_num):
                for lane_id, lane in signal.in_lanes.items():
                    if i == 0:  # The first time slot
                        m.addConstr(
                            slot_lane_queue[lane_id][i] == lane.queue + lane.arrival_rate * slot_green_duration[i] -
                            slot_lane_throughput[lane_id][i],
                            name=constr_name_attach('slot' + str(i), lane_id, 'queue'))
                    else:
                        m.addConstr(
                            slot_lane_queue[lane_id][i] == slot_lane_queue[lane_id][i - 1] + lane.arrival_rate *
                            slot_green_duration[i] - slot_lane_throughput[lane_id][i - 1],
                            name=constr_name_attach('slot' + str(i), lane_id, 'queue'))
            # Constraint 2: Throughput of each time slot
            for i in range(self.time_slot_num):
                for lane_id, lane in signal.in_lanes.items():
                    m.addConstr(
                        slot_lane_green_sfr[lane_id][i] == slot_green_duration[i] * lane.saturation_flow_rate,
                        name=constr_name_attach('slot' + str(i), lane_id, 'green_sfr'))
                    if i == 0:  # The first time slot
                        m.addConstr(slot_lane_green_demand[lane_id][i] == lane.queue +
                                    lane.arrival_rate * slot_green_duration[i],
                                    name=constr_name_attach('slot' + str(i), lane_id, 'green_demand'))
                    else:
                        m.addConstr(slot_lane_green_demand[lane_id][i] == slot_lane_queue[lane_id][i] +
                                    lane.arrival_rate * slot_green_duration[i],
                                    name=constr_name_attach('slot' + str(i), lane_id, 'green_demand'))
                    m.addConstr(slot_lane_green_throughput[lane_id][i] == gp.min_(
                        slot_lane_green_demand[lane_id][i], slot_lane_green_sfr[lane_id][i]),
                                name=constr_name_attach('slot' + str(i), lane_id, 'green_throughput'))
                    m.addConstr(slot_lane_throughput[lane_id][i] <= M * slot_lane_selection[lane_id][i],
                                name=constr_name_attach('slot' + str(i), lane_id, 'red_output1'))
                    m.addConstr(slot_lane_throughput[lane_id][i] >= -M * slot_lane_selection[lane_id][i],
                                name=constr_name_attach('slot' + str(i), lane_id, 'red_output2'))
                    m.addConstr(slot_lane_throughput[lane_id][i] - slot_lane_green_throughput[lane_id][i] <= M * (
                            1 - slot_lane_selection[lane_id][i]),
                                name=constr_name_attach('slot' + str(i), lane_id, 'green_output1'))
                    m.addConstr(slot_lane_throughput[lane_id][i] - slot_lane_green_throughput[lane_id][i] >= -M * (
                            1 - slot_lane_selection[lane_id][i]),
                                name=constr_name_attach('slot' + str(i), lane_id, 'green_output2'))
            # Constraint 3: Total throughput
            for lane_id, lane in signal.in_lanes.items():
                m.addConstr(lane_throughput[lane_id] == gp.quicksum(slot_lane_throughput[lane_id]),
                            name=constr_name_attach(lane_id, 'total_throughput'))
            # Constraint 4: Downstream capacity constraint
            for lane_id, lane in signal.in_lanes.items():
                m.addConstr(lane_throughput[lane_id] <= lane.get_downstream_capacity(),
                            name=constr_name_attach(signal_id, lane_id, 'down_capacity'))
            # Constraint 5: Each movement is selected at least once
            for movement_id in signal.movements.keys():
                m.addConstr(gp.quicksum(slot_movement_selection[movement_id]) >= 1,
                            name=constr_name_attach(movement_id, 'movement_appearance'))
            # Constraint 6: At least one stage is selected for each slot
            for i in range(self.time_slot_num):
                stage_list = []
                for stage_id in signal.stages.keys():
                    stage_list.append(slot_stage_selection[stage_id][i])
                m.addConstr(gp.quicksum(stage_list) == 1,
                            name=constr_name_attach('slot' + str(i), 'stage_selection'))
            # Constraint 7: The same stage is not selected consequently
            for i in range(self.time_slot_num - 1):
                for stage_id in signal.stages.keys():
                    m.addConstr(
                        slot_stage_selection[stage_id][i] <= M * (1 - slot_stage_selection[stage_id][i + 1]),
                        name=constr_name_attach('slot' + str(i), 'stage_change1'))
                    m.addConstr(
                        slot_stage_selection[stage_id][i] >= -M * (1 - slot_stage_selection[stage_id][i + 1]),
                        name=constr_name_attach('slot' + str(i), 'stage_change2'))
            # Constraint 8: Stage-movement-lane relationship
            for i in range(self.time_slot_num):
                for movement_id, movement in signal.movements.items():
                    stage_list = []
                    for stage_id in movement.stages:
                        stage_list.append(slot_stage_selection[stage_id][i])
                    m.addConstr(slot_movement_selection[movement_id][i] == gp.max_(stage_list),
                                name=constr_name_attach('slot' + str(i), movement_id, 'selection'))
                for lane_id, lane in signal.in_lanes.items():
                    # 由于没有直左车道，因此可以直接根据直行/左转movement确定lane的selection
                    for movement_id in lane.movements.keys():
                        if movement_id[-1] != 'r':
                            m.addConstr(slot_lane_selection[lane_id][i] == slot_movement_selection[movement_id][i],
                                        name=constr_name_attach('slot' + str(i), lane_id, 'selection'))
            # Constraint 9: Signal light merge indicator
            for i in range(self.time_slot_num - 1):
                for movement_id, movement in signal.movements.items():
                    m.addConstr(
                        succ_slot_movement_selection[movement_id][i] == slot_movement_selection[movement_id][i] -
                        slot_movement_selection[movement_id][i + 1],
                        name=constr_name_attach('slot' + str(i), movement_id, 'succ'))
                    m.addConstr(slot_movement_no_merge[movement_id][i] == gp.abs_(
                        succ_slot_movement_selection[movement_id][i]),
                                name=constr_name_attach('slot' + str(i), movement_id, 'merge'))
            # Constraint 10: The green time sequence
            for i in range(self.time_slot_num - 1):
                m.addConstr(slot_green_start[i + 1] == slot_green_start[i] + slot_green_duration[i],
                            name=constr_name_attach('slot' + str(i), 'slot_sequence'))
            # Constraint 11: Boundary
            m.addConstr(slot_green_start[0] == 0, name='first_slot_green_start')
            m.addConstr(slot_green_start[self.time_slot_num - 1] + slot_green_duration[
                self.time_slot_num - 1] == signal.cycle,
                        name='last_slot_green_end')
            # Constraint 12: Minimum green
            for i in range(self.time_slot_num):
                m.addConstr(slot_green_duration[i] >= self.config['min_green'] + yellow,
                            name=constr_name_attach('slot' + str(i), 'min_slot_duration'))
            # Constraint 13: The green time of movement and lane
            for i in range(self.time_slot_num):
                for movement_id in signal.movements.keys():
                    # No right-of-way movement
                    m.addConstr(
                        slot_movement_green_duration[movement_id][i] <= M * slot_movement_selection[movement_id][i],
                        name=constr_name_attach('slot' + str(i), movement_id, 'no_green1'))
                    m.addConstr(
                        slot_movement_green_duration[movement_id][i] >= -M * slot_movement_selection[movement_id][
                            i],
                        name=constr_name_attach('slot' + str(i), movement_id, 'no_green2'))
                    # Right-of-way movement
                    if i < self.time_slot_num - 1:  # Merged slot have no yellow time
                        m.addConstr(slot_movement_green_duration[movement_id][i] - slot_green_duration[i] <= M * (
                                1 - slot_movement_no_merge[movement_id][i] - slot_movement_selection[movement_id][
                            i]),
                                    name=constr_name_attach('slot' + str(i), movement_id, 'merge_green1'))
                        m.addConstr(slot_movement_green_duration[movement_id][i] - slot_green_duration[i] >= -M * (
                                1 - slot_movement_no_merge[movement_id][i] - slot_movement_selection[movement_id][
                            i]),
                                    name=constr_name_attach('slot' + str(i), movement_id, 'merge_green2'))
                        m.addConstr(
                            slot_movement_green_duration[movement_id][i] - slot_green_duration[i] + yellow <= M * (
                                    2 - slot_movement_no_merge[movement_id][i] -
                                    slot_movement_selection[movement_id][i]),
                            name=constr_name_attach('slot' + str(i), movement_id, 'unmerge_green1'))
                        m.addConstr(
                            slot_movement_green_duration[movement_id][i] - slot_green_duration[i] + yellow >= -M * (
                                    2 - slot_movement_no_merge[movement_id][i] -
                                    slot_movement_selection[movement_id][i]),
                            name=constr_name_attach('slot' + str(i), movement_id, 'unmerge_green2'))
                    else:  # The last slot always have yellow time (can be improved)
                        m.addConstr(
                            slot_movement_green_duration[movement_id][i] - slot_green_duration[i] + yellow <= M * (
                                    1 - slot_movement_selection[movement_id][i]),
                            name=constr_name_attach('slot' + str(i), movement_id, 'unmerge_green1'))
                        m.addConstr(
                            slot_movement_green_duration[movement_id][i] - slot_green_duration[i] + yellow >= -M * (
                                    1 - slot_movement_selection[movement_id][i]),
                            name=constr_name_attach('slot' + str(i), movement_id, 'unmerge_green2'))
                for lane_id, lane in signal.in_lanes.items():
                    # 由于没有直左车道，因此可以直接根据直行/左转movement确定lane的green
                    for movement_id in lane.movements.keys():
                        if movement_id[-1] != 'r':
                            m.addConstr(
                                slot_lane_green_duration[lane_id][i] == slot_movement_green_duration[movement_id][
                                    i],
                                name=constr_name_attach('slot' + str(i), lane_id, 'green'))
            # Constraint 14: Fixed the duration of inflow movement
            for movement_id, movement in signal.movements.items():
                if movement.type == 'inflow':
                    m.addConstr(
                        gp.quicksum(slot_movement_green_duration[movement_id]) == movement.fixed_gated_green,
                        name=constr_name_attach(movement_id, 'inflow_fixed_green_duration'))

            m.optimize()

            if m.status == gp.GRB.OPTIMAL:
                # get the objective value
                throughput_result = total_throughput.x
                perimeter_total_throughput += throughput_result
                # reset signal settings
                signal.reset()
                # extract the optimal solution
                for i in range(self.time_slot_num):
                    green_start = slot_green_start[i].x
                    for movement_id, movement in signal.movements.items():
                        # get the green time for the right-of-way movements
                        if slot_movement_selection[movement_id][i].x == 1:
                            movement.green_start.append(green_start)
                            green_duration = slot_movement_green_duration[movement_id][i].x
                            movement.green_duration.append(green_duration)
                            # get the yellow start time for the end of the right-of-way movements
                            if slot_movement_no_merge[movement_id][i].x == 1:
                                movement.yellow_start.append(green_start + green_duration)
                    for lane_id, lane in signal.in_lanes.items():
                        # get the green time for the right-of-way lanes
                        if slot_lane_selection[lane_id][i].x == 1:
                            lane.green_start.append(green_start)
                            green_duration = slot_lane_green_duration[lane_id][i].x
                            lane.green_duration.append(green_duration)
                    for connection in signal.connections.values():
                        connection.update_timing()
            elif m.status == gp.GRB.INFEASIBLE:
                # m.computeIIS()
                # for c in m.getConstrs():
                #     if c.IISConstr:
                #         print(c)
                # m.write('infeasible.ilp')
                print("The lower level problem is infeasible. ")
            elif m.status == gp.GRB.UNBOUNDED:
                print("The lower level problem is unbounded. ")
            else:
                print("No solution of the lower level problem found in this step!")

        return perimeter_total_throughput