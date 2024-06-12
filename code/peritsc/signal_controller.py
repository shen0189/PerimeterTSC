import gurobipy as gp
from utils.utilize import config, set_sumo
from peritsc.perimeterdata import PeriSignals
from typing import Dict, List


def constr_name_attach(*args, join_char: str = '_'):
    return join_char.join(args)


class PeriSignalController:
    """
    Optimize the signal plan given the inflow vehicles (action)
    """

    def __init__(self, peri_data: PeriSignals, action: float):
        # control mode
        self.distribution_mode = config['peri_distribution_mode']
        self.signal_phase_mode = config['peri_signal_phase_mode']
        self.optimization_mode = config['peri_optimization_mode']
        # input
        self.upper_inflow = action
        self.peri_data = peri_data

    def signal_optimize(self):

        if self.optimization_mode == 'centralize':
            estimate_inflow = self.set_inflow_local_green()
        else:
            estimate_inflow = self.inflow_local_iteration()
        return estimate_inflow

    # 单层直接求解
    def set_inflow_local_green(self):
        """
        Centralized method: optimize the inflow distribution and signal plan simultaneously
        """
        cycle, sfr, yellow = config['cycle_time'], config['through_sfr'], config['yellow_duration']
        lane_maximal_throughput = cycle * sfr
        maximal_action = config['saturation_flow_rate'] * 3600 * len(self.peri_data.peri_inflow_lanes)
        M = 1e5

        # Multi-objective optimization
        m = gp.Model('qp')

        m.setParam('OutputFlag', 0)
        m.setParam(gp.ParamConstClass.DualReductions, 0)

        # Variables: queue length at the next step of each lane;
        # green start of each lane/movement; green duration of each lane/movement
        lane_final_queue_estimate: Dict[str, gp.Var] = {tls: {} for tls in self.peri_data.peri_signals.keys()}
        lane_final_queue: Dict[str, gp.Var] = {tls: {} for tls in self.peri_data.peri_signals.keys()}
        lane_relative_queue: Dict[str, gp.Var] = {tls: {} for tls in self.peri_data.peri_signals.keys()}
        movement_green_start: Dict[str, gp.Var] = {tls: {} for tls in self.peri_data.peri_signals.keys()}
        movement_green_dur: Dict[str, gp.Var] = {tls: {} for tls in self.peri_data.peri_signals.keys()}
        # movement_green_start: Dict[str, gp.Var] = {tls: {'inflow': {}, 'outflow': {}, 'normal flow': {}}
        #                                            for tls in peri_data.peri_signals.keys()}
        # movement_green_dur: Dict[str, gp.Var] = {tls: {'inflow': {}, 'outflow': {}, 'normal flow': {}}
        #                                            for tls in peri_data.peri_signals.keys()}
        # movement_green_overlap: Dict[tuple, gp.Var] = {}
        movement_sequence: Dict[str, gp.Var] = {tls: {} for tls in self.peri_data.peri_signals.keys()}
        lane_green_start: Dict[str, gp.Var] = {tls: {} for tls in self.peri_data.peri_signals.keys()}
        lane_green_dur: Dict[str, gp.Var] = {tls: {} for tls in self.peri_data.peri_signals.keys()}
        lane_throughput: Dict[str, gp.Var] = {tls: {} for tls in self.peri_data.peri_signals.keys()}
        lane_green_discharge: Dict[str, gp.Var] = {tls: {} for tls in self.peri_data.peri_signals.keys()}
        lane_green_demand: Dict[str, gp.Var] = {tls: {} for tls in self.peri_data.peri_signals.keys()}
        # Add variables to each container
        for signal_id, signal in self.peri_data.peri_signals.items():
            for lane_id in signal.in_lanes.keys():
                var_queue_estimate = m.addVar(lb=-gp.GRB.INFINITY, name=constr_name_attach(lane_id, 'queue_estimate'))
                var_queue = m.addVar(lb=0, name=constr_name_attach(lane_id, 'queue'))
                var_relative_queue = m.addVar(lb=0, name=constr_name_attach(lane_id, 'relative_queue'))
                var_green_start = m.addVar(lb=0, name=constr_name_attach(lane_id, 'green_start'))
                var_green_dur = m.addVar(lb=0, name=constr_name_attach(lane_id, 'green_duration'))
                var_throughput = m.addVar(lb=0, name=constr_name_attach(lane_id, 'throughput'))
                var_green_discharge = m.addVar(lb=0, name=constr_name_attach(lane_id, 'green_discharge'))
                var_green_demand = m.addVar(lb=0, name=constr_name_attach(lane_id, 'green_demand'))
                lane_final_queue_estimate[signal_id][lane_id] = var_queue_estimate
                lane_final_queue[signal_id][lane_id] = var_queue
                lane_relative_queue[signal_id][lane_id] = var_relative_queue
                lane_green_start[signal_id][lane_id] = var_green_start
                lane_green_dur[signal_id][lane_id] = var_green_dur
                lane_throughput[signal_id][lane_id] = var_throughput
                lane_green_discharge[signal_id][lane_id] = var_green_discharge
                lane_green_demand[signal_id][lane_id] = var_green_demand
            for movement_id, movement in signal.movements.items():
                var_green_start = m.addVar(lb=0, name=constr_name_attach(movement_id, 'green_start'))
                var_green_dur = m.addVar(lb=0, name=constr_name_attach(movement_id, 'green_duration'))
                # movement_type = movement.type
                movement_green_start[signal_id][movement_id] = var_green_start
                movement_green_dur[signal_id][movement_id] = var_green_dur
                for movement2_id, movement2 in signal.movements.items():
                    var_movement_sequence = m.addVar(vtype=gp.GRB.BINARY, name=constr_name_attach(movement_id, movement2_id, 'sequence'))
                    movement_sequence[signal_id][(movement_id, movement2_id)] = var_movement_sequence
        # Other variables
        inflow_diff = m.addVar(lb=-gp.GRB.INFINITY, name='inflow_diff')
        abs_inflow_diff = m.addVar(lb=0, name='abs_inflow_diff')
        total_squared_queue = m.addVar(lb=0, name='squared_queue')
        total_squared_throughput = m.addVar(lb=0, name='squared_throughput')
        total_throughput = m.addVar(lb=0, name='total_thrp')
        # for movement_id, movement in peri_data.peri_inflows.items():
        #     for movement2_id, movement2 in peri_data.peri_inflows.items():
        #         if movement2_id != movement_id:
        #             var_green_overlap = m.addVar(name=movement_id + '_' + movement2_id + '_glap')
        #             movement_green_overlap[signal_id][tuple([movement_id, movement2_id])] = var_green_overlap

        # Multi-objectives
        obj = 0
        # obj1: Minimize the difference between total inflow and required total inflow (normalized)
        inflow_lanes = gp.tuplelist(self.peri_data.peri_inflow_lanes)
        m.addConstr(inflow_diff == (self.upper_inflow - gp.quicksum(
            lane_throughput[signal_id][lane_id] for signal_id, lane_id in inflow_lanes)) / maximal_action,
                    name='cal_diff')
        m.addGenConstrAbs(abs_inflow_diff, inflow_diff, name='abs')
        obj += config['obj_weight']['gating'] * abs_inflow_diff
        # obj2 (queue-balance): Minimize the variance of relative queue length,
        # which is equivalent to minimize the summation of squared relative queue length when obj1 >> obj2
        if self.distribution_mode == 'balance_queue':
            for signal_id, signal in self.peri_data.peri_signals.items():
                for lane_id, lane in signal.in_lanes.items():
                    m.addConstr(lane_relative_queue[signal_id][lane_id] == lane_final_queue[signal_id][
                        lane_id] * lane.vehicle_length / lane.length,
                                name=constr_name_attach(signal_id, lane_id, 'cal_relative_queue'))
            m.addConstr(
                total_squared_queue == gp.quicksum(lane_relative_queue[signal][lane] * lane_relative_queue[signal][lane]
                                                   for signal, lane in inflow_lanes), name='cal_squared_queue')
            obj += config['obj_weight']['balance'] * total_squared_queue
        # obj2 (equally distributed): Minimize the variance of normalized throughput,
        # which is equivalent to minimize the summation of squared throughput when obj1 >> obj2
        elif self.distribution_mode == 'equal':
            m.addConstr(total_squared_throughput == gp.quicksum(
                lane_throughput[signal][lane] * lane_throughput[signal][lane] / pow(lane_maximal_throughput, 2)
                for signal, lane in inflow_lanes), name='cal_squared_throughput')
            obj += config['obj_weight']['balance'] * total_squared_throughput
        # obj3: Maximize the (weighted) normalized throughput
        all_lanes = gp.tuplelist(self.peri_data.get_all_lanes())
        m.addConstr(total_throughput == gp.quicksum(
            lane_throughput[signal][lane] / lane_maximal_throughput for signal, lane in all_lanes), name='cal_thrp')
        obj -= config['obj_weight']['local'] * total_throughput
        m.setObjective(obj)

        # Add constraints
        for signal_id, signal in self.peri_data.peri_signals.items():
            # Constraint 1: Final queue estimation
            for lane_id, lane in signal.in_lanes.items():
                m.addConstr(lane_final_queue_estimate[signal_id][lane_id] == lane.queue + signal.cycle * lane.arrival_rate -
                            lane_throughput[signal_id][lane_id], name=constr_name_attach(signal_id, lane_id, 'queue_estimate'))
                m.addConstr(lane_final_queue[signal_id][lane_id] == gp.max_(
                    lane_final_queue_estimate[signal_id][lane_id], 0), name=constr_name_attach(signal_id, lane_id, 'queue'))
            # Constraint 2: Lane throughput calculation
            for lane_id, lane in signal.in_lanes.items():
                m.addConstr(lane_green_discharge[signal_id][lane_id] == lane_green_dur[signal_id][
                    lane_id] * lane.saturation_flow_rate, name=constr_name_attach(signal_id, lane_id, 'green_discharge'))
                m.addConstr(lane_green_demand[signal_id][lane_id] == lane.queue + lane.arrival_rate * (
                        lane_green_start[signal_id][lane_id] + lane_green_dur[signal_id][lane_id]),
                            name=constr_name_attach(signal_id, lane_id, 'green_demand'))
                if config['peri_green_start_model']:
                    m.addConstr(lane_throughput[signal_id][lane_id] == gp.min_(lane_green_discharge[signal_id][lane_id],
                                                                               lane_green_demand[signal_id][lane_id]),
                                name=constr_name_attach(signal_id, lane_id, 'throughput'))
                else:
                    if lane_id in self.peri_data.peri_inflow_lanes_by_laneID:
                        m.addConstr(
                            lane_throughput[signal_id][lane_id] == lane_green_discharge[signal_id][lane_id],
                            name=constr_name_attach(signal_id, lane_id, 'throughput'))
                    else:       # 非inflow方向正常处理
                        m.addConstr(
                            lane_throughput[signal_id][lane_id] == gp.min_(lane_green_discharge[signal_id][lane_id],
                                                                           lane_green_demand[signal_id][lane_id]),
                            name=constr_name_attach(signal_id, lane_id, 'throughput'))
            # Constraint 3: Downstream capacity constraint
            for lane_id, lane in signal.in_lanes.items():
                m.addConstr(lane_throughput[signal_id][lane_id] <= lane.get_downstream_capacity(),
                            name=constr_name_attach(signal_id, lane_id, 'down_capacity'))
            # Constraint 4: Minimum / Maximum green duration
            for movement_id, movement in signal.movements.items():
                m.addConstr(movement_green_dur[signal_id][movement_id] <= config['max_green'],
                            name=constr_name_attach(signal_id, movement_id, 'max_green'))
                m.addConstr(movement_green_dur[signal_id][movement_id] >= config['min_green'],
                            name=constr_name_attach(signal_id, movement_id, 'max_green'))
            # Constraint 5: Phase order (Signal timing of movement)
            # Constraint 5.1: Phase order for NEMA
            if self.signal_phase_mode == 'NEMA':
                phase_order = self.peri_data.peri_info[signal_id]['nema_plan']
                barrier_movement = []  # 每个ring的第二个barrier的首个movement
                for ring_name, ring in phase_order.items():
                    for idx in range(0, len(ring) - 1):
                        pred_movement, succ_movement = ring[idx], ring[idx + 1]
                        # 5.1.1 start and end of the cycle
                        if idx == 0:
                            m.addConstr(movement_green_start[signal_id][pred_movement] == 0,
                                        name=constr_name_attach(ring_name, 'start'))
                        elif idx == len(ring) - 2:
                            m.addConstr(movement_green_start[signal_id][succ_movement] + movement_green_dur[signal_id][
                                succ_movement] + yellow == signal.cycle,
                                        name=constr_name_attach(ring_name, 'end'))
                        # 5.1.2 normal phase order
                        if pred_movement == '':
                            barrier_movement.append(succ_movement)
                            continue
                        if succ_movement == '':
                            succ_movement = ring[idx + 2]
                        m.addConstr(movement_green_start[signal_id][pred_movement] + movement_green_dur[signal_id][
                            pred_movement] + yellow == movement_green_start[signal_id][succ_movement],
                                    name=constr_name_attach(ring_name, str(idx)))
                # 5.1.3 barrier time
                ring1_second_barrier_movement, ring2_second_barrier_movement = barrier_movement
                m.addConstr(
                    movement_green_start[signal_id][ring1_second_barrier_movement] == movement_green_start[signal_id][
                        ring2_second_barrier_movement], name='barrier')
            # Constraint 5.2: Phase order for unfixed structure
            elif self.signal_phase_mode == 'Unfixed':
                conflict_matrix = signal.conflict_matrix
                # 5.2.1 First, constraint the end of green duration
                for movement_id, movement in signal.movements.items():
                    m.addConstr(
                        movement_green_start[signal_id][movement_id] + movement_green_dur[signal_id][movement_id] +
                        yellow <= signal.cycle,
                        name=constr_name_attach(signal_id, movement_id, 'green_end'))
                    # 5.2.2 Then, avoid the conflict
                    for movement2_id, movement2 in signal.movements.items():
                        if movement_id != movement2_id:
                            m.addConstr(movement_sequence[signal_id][(movement_id, movement2_id)] +
                                        movement_sequence[signal_id][(movement2_id, movement_id)] == 1,
                                        name=constr_name_attach(signal_id, movement_id, movement2_id, 'sequence'))
                            m.addConstr(movement_green_start[signal_id][movement_id] +
                                        movement_green_dur[signal_id][movement_id] + yellow -
                                        movement_green_start[signal_id][movement2_id] - signal.cycle *
                                        movement_sequence[signal_id][(movement_id, movement2_id)] <= M * (
                                                    1 - conflict_matrix[movement_id][movement2_id]),
                                        name=constr_name_attach(signal_id, movement_id, movement2_id, 'conflict'))
            # Constraint 6: Signal timing of right-turn movement, same as that of through movement
            for movement_id, movement in signal.movements.items():
                if movement.dir == 'r':
                    through_movement = movement_id[:-1] + 's'
                    m.addConstr(
                        movement_green_dur[signal_id][movement_id] == movement_green_dur[signal_id][through_movement],
                        name=constr_name_attach(signal_id, movement_id, 'inflow_right_greendur'))
                    m.addConstr(
                        movement_green_start[signal_id][movement_id] == movement_green_start[signal_id][
                            through_movement],
                        name=constr_name_attach(signal_id, movement_id, 'inflow_right_greenstart'))
                    # if movement.type == 'inflow':
                    #     through_movement = movement_id[:-1] + 's'
                    #     m.addConstr(movement_green_dur[signal_id][movement_id] == movement_green_dur[signal_id][through_movement],
                    #                 name=constr_name_attach((signal_id, movement_id, 'inflow_right_greendur')))
                    #     m.addConstr(movement_green_start[signal_id][movement_id] == movement_green_start[signal_id][through_movement],
                    #                 name=constr_name_attach((signal_id, movement_id, 'inflow_right_greenstart')))
                    # else:
                    #     m.addConstr(movement_green_dur[signal_id][movement_id] == signal.cycle,
                    #                 name=constr_name_attach((signal_id, movement_id, 'noninflow_right_greendur')))
                    #     m.addConstr(movement_green_start[signal_id][movement_id] == 0,
                    #                 name=constr_name_attach((signal_id, movement_id, 'noninflow_right_greenstart')))
            # Constraint 7: Match the signal timing of movement and lane
            for lane_id, lane in signal.in_lanes.items():
                for movement_id, movement in lane.movements.items():
                    # print(f'Combine green time of lane {lane_id} and movement {movement_id}')
                    m.addConstr(movement_green_dur[signal_id][movement_id] == lane_green_dur[signal_id][lane_id],
                                name=constr_name_attach(signal_id, lane_id, 'lane_green_duration'))
                    m.addConstr(movement_green_start[signal_id][movement_id] == lane_green_start[signal_id][lane_id],
                                name=constr_name_attach(signal_id, lane_id, 'lane_green_start'))

        m.optimize()

        if m.status == gp.GRB.OPTIMAL:
            # check the value of three objectives
            # print(f"Objective value: {config['obj_weight']['gating'] * abs_inflow_diff.x + config['obj_weight']['balance'] * total_squared_throughput.x - config['obj_weight']['local'] * total_throughput.x}")
            # print(f'Objective one value: {abs_inflow_diff.x}')
            # print(f'Objective two value: {total_squared_throughput.x}')
            # print(f'Objective three value: {total_throughput.x}')
            # extract the optimal solution
            for signal_id, signal in self.peri_data.peri_signals.items():
                for movement_id, movement in signal.movements.items():
                    movement.green_start = int(movement_green_start[signal_id][movement_id].x)
                    movement.green_duration = int(movement_green_dur[signal_id][movement_id].x)
                    # if movement.type == 'inflow' and movement_id[-1] != 'r':
                    #     print(f'The green duration of movement {movement_id} is {movement.green_duration}')
                for lane_id, lane in signal.in_lanes.items():
                    lane.green_start = int(lane_green_start[signal_id][lane_id].x)
                    lane.green_duration = int(lane_green_dur[signal_id][lane_id].x)
                for connection in signal.connections.values():
                    connection.update_timing()
            # check the real inflow
            inflow_diff_value = abs_inflow_diff.x
            inflow = 0
            for signal_id, lane_id in self.peri_data.peri_inflow_lanes:
                inflow += lane_throughput[signal_id][lane_id].x
            print(f'The estimated inflow by model is {inflow}. ')
            return inflow
        elif m.status == gp.GRB.INFEASIBLE:
            m.computeIIS()
            for c in m.getConstrs():
                if c.IISConstr:
                    print(c)
            m.write('infeasible.ilp')
            return 0
        elif m.status == gp.GRB.UNBOUNDED:
            return 0
        else:
            print("No solution found in this step!")
            return 0


    # 双层分布式迭代求解框架
    def inflow_local_iteration(self):
        '''
        迭代进行分配绿灯时长-确定边界交叉口具体配时的过程，以边界流入量、边界排队方差、交叉口吞吐量之和的变化作为收敛标准
        '''
        print('----------Signal optimization begins------------')
        epsilon, iteration = 1, 0
        total_inflow_evolution, queue_variance_evolution, total_throughput_evolution = [], [], []
        while True:
            total_inflow, queue_variance = self.set_inflow_green()
            total_throughput = self.set_local_green()
            total_inflow_evolution.append(total_inflow)
            queue_variance_evolution.append(queue_variance)
            total_throughput_evolution.append(total_throughput)
            # print(f'Total_inflow_evolution: {total_inflow_evolution}')
            # print(f'Queue_variance_evolution: {queue_variance_evolution}')
            # print(f'Total_throughput_evolution: {total_throughput_evolution}')
            if len(total_inflow_evolution) > 1:
                # calculate the difference between two results
                total_inflow_diff = abs(total_inflow_evolution[-1] - total_inflow_evolution[-2])
                queue_variance_diff = abs(queue_variance_evolution[-1] - queue_variance_evolution[-2])
                total_throughput_diff = abs(total_throughput_evolution[-1] - total_throughput_evolution[-2])
                objective_diff = total_inflow_diff + queue_variance_diff + total_throughput_diff
                # print(f'Difference in this step: {objective_diff}')
                iteration += 1
                if objective_diff < epsilon or iteration > config['max_iteration_step']:
                    break
        print('----------Signal optimization ends------------')
        return total_inflow

    def set_inflow_green(self):
        '''
        输入：路网流入量action
        固定变量: inflow方向绿灯启亮时刻 (第一次迭代时启亮时间为上一次优化的求解结果)
        决策变量：inflow方向绿灯时长
        目标：流入量接近action + 排队均衡（或其他）
        '''
        cycle, sfr, yellow = config['cycle_time'], config['through_sfr'], config['yellow_duration']
        lane_maximal_throughput = cycle * sfr
        maximal_action = config['saturation_flow_rate'] * config['max_green'] * len(self.peri_data.peri_inflow_lanes)
        M = 1e5

        # Multi-objective optimization
        m = gp.Model('qp')

        m.setParam('OutputFlag', 0)
        m.setParam(gp.ParamConstClass.DualReductions, 0)

        # Variables
        lane_final_queue: Dict[str, gp.Var] = {}
        lane_final_queue_estimate: Dict[str, gp.Var] = {}       # 可能为负
        lane_spillover: Dict[str, gp.Var] = {}      # 车道排队是否溢出
        lane_relative_queue: Dict[str, gp.Var] = {}
        movement_green_dur: Dict[str, gp.Var] = {}
        lane_green_dur: Dict[str, gp.Var] = {}
        lane_throughput: Dict[str, gp.Var] = {}
        lane_green_discharge: Dict[str, gp.Var] = {}
        lane_green_demand: Dict[str, gp.Var] = {}
        # Add variables to each container
        for _, lane_id in self.peri_data.peri_inflow_lanes:
            var_queue = m.addVar(lb=0, name=constr_name_attach(lane_id, 'queue'))
            var_queue_estimate = m.addVar(lb=-gp.GRB.INFINITY, name=constr_name_attach(lane_id, 'queue_estimate'))
            var_spillover = m.addVar(vtype=gp.GRB.BINARY, name=constr_name_attach(lane_id, 'spillover'))
            var_relative_queue = m.addVar(lb=0, name=constr_name_attach(lane_id, 'relative_queue'))
            var_green_dur = m.addVar(lb=0, name=constr_name_attach(lane_id, 'green_duration'))
            var_throughput = m.addVar(lb=0, name=constr_name_attach(lane_id, 'throughput'))
            var_green_discharge = m.addVar(lb=0, name=constr_name_attach(lane_id, 'green_discharge'))
            var_green_demand = m.addVar(lb=0, name=constr_name_attach(lane_id, 'green_demand'))
            lane_final_queue[lane_id] = var_queue
            lane_final_queue_estimate[lane_id] = var_queue_estimate
            lane_spillover[lane_id] = var_spillover
            lane_relative_queue[lane_id] = var_relative_queue
            lane_green_dur[lane_id] = var_green_dur
            lane_throughput[lane_id] = var_throughput
            lane_green_discharge[lane_id] = var_green_discharge
            lane_green_demand[lane_id] = var_green_demand
            lane = self.peri_data.peri_inflow_lanes_by_laneID[lane_id]
            for movement_id, movement in lane.movements.items():
                var_green_dur = m.addVar(lb=0, name=constr_name_attach(movement_id, 'green_duration'))
                # movement_type = movement.type
                movement_green_dur[movement_id] = var_green_dur
        # Other variables
        inflow_diff: gp.Var = m.addVar(lb=-gp.GRB.INFINITY, name='inflow_diff')
        abs_inflow_diff: gp.Var = m.addVar(lb=0, name='abs_inflow_diff')
        total_squared_queue: gp.Var = m.addVar(lb=0, name='squared_queue')
        total_squared_throughput: gp.Var = m.addVar(lb=0, name='squared_throughput')
        total_spillover: gp.Var = m.addVar(lb=0, name='total_spillover')

        # Multi-objectives
        obj = 0
        # obj1: Minimize the difference between total inflow and required total inflow (normalized)
        inflow_lanes = gp.tuplelist(self.peri_data.peri_inflow_lanes)
        m.addConstr(inflow_diff == (self.upper_inflow - gp.quicksum(
            lane_throughput[lane_id] for _, lane_id in inflow_lanes)) / maximal_action, name='cal_diff')
        m.addGenConstrAbs(abs_inflow_diff, inflow_diff, name='abs')
        obj += config['upper_obj_weight']['gating'] * abs_inflow_diff
        # obj2 (queue-balance): Minimize the variance of relative queue length,
        # which is equivalent to minimize the summation of squared relative queue length when obj1 >> obj2
        if self.distribution_mode == 'balance_queue':
            for signal_id, lane_id in self.peri_data.peri_inflow_lanes:
                lane = self.peri_data.peri_inflow_lanes_by_laneID[lane_id]
                m.addConstr(lane_relative_queue[lane_id] == lane_final_queue[lane_id] * lane.vehicle_length / lane.length,
                            name=constr_name_attach(signal_id, lane_id, 'cal_relative_queue'))
            m.addConstr(
                total_squared_queue == gp.quicksum(lane_relative_queue[lane_id] * lane_relative_queue[lane_id]
                                                   for _, lane_id in inflow_lanes), name='cal_squared_queue')
            obj += config['upper_obj_weight']['balance'] * total_squared_queue
        # obj2 (equally distributed): Minimize the variance of normalized throughput,
        # which is equivalent to minimize the summation of squared throughput when obj1 >> obj2
        elif self.distribution_mode == 'equal':
            m.addConstr(total_squared_throughput == gp.quicksum(
                lane_throughput[lane_id] * lane_throughput[lane_id] / pow(lane_maximal_throughput, 2)
                for _, lane_id in inflow_lanes), name='cal_squared_throughput')
            obj += config['upper_obj_weight']['balance'] * total_squared_throughput
        m.setObjective(obj)
        # obj3 (prevent spillover): Minimize the number of lanes with spillover
        if config['peri_spillover_penalty'] is True:
            m.addConstr(total_spillover == gp.quicksum(lane_spillover[lane_id] for _, lane_id in inflow_lanes) / len(inflow_lanes),
                        name='cal_total_spillover')
            obj += config['upper_obj_weight']['spillover'] * total_spillover

        # Add constraints (for inflow movements)
        for signal_id, lane_id in self.peri_data.peri_inflow_lanes:
            signal = self.peri_data.peri_signals[signal_id]
            lane = self.peri_data.peri_inflow_lanes_by_laneID[lane_id]
            # Constraint 1: Final queue estimation
            m.addConstr(lane_final_queue_estimate[lane_id] == lane.queue + signal.cycle * lane.arrival_rate -
                        lane_throughput[lane_id], name=constr_name_attach(signal_id, lane_id, 'queue_estimate'))
            m.addConstr(lane_final_queue[lane_id] == gp.max_(lane_final_queue_estimate[lane_id], 0),
                        name=constr_name_attach(signal_id, lane_id, 'queue'))
            # Constraint 2: Lane throughput calculation
            m.addConstr(lane_green_discharge[lane_id] == lane_green_dur[lane_id] * lane.saturation_flow_rate,
                        name=constr_name_attach(signal_id, lane_id, 'green_discharge'))
            # print(f'The green start of lane {lane_id} is {lane.green_start}')
            m.addConstr(lane_green_demand[lane_id] == lane.queue + lane.arrival_rate * (
                    lane.green_start + lane_green_dur[lane_id]),
                        name=constr_name_attach(signal_id, lane_id, 'green_demand'))
            if config['peri_green_start_model']:
                m.addConstr(lane_throughput[lane_id] == gp.min_(lane_green_discharge[lane_id], lane_green_demand[lane_id]),
                            name=constr_name_attach(signal_id, lane_id, 'throughput'))
            else:
                m.addConstr(lane_throughput[lane_id] == lane_green_discharge[lane_id],
                            name=constr_name_attach(signal_id, lane_id, 'throughput'))
            # Constraint 3: Downstream capacity constraint
            m.addConstr(lane_throughput[lane_id] <= lane.get_downstream_capacity(),
                        name=constr_name_attach(signal_id, lane_id, 'down_capacity'))
            # Constraint 4: Minimum / Maximum green duration
            for movement_id, movement in lane.movements.items():
                if self.signal_phase_mode == 'NEMA':
                    m.addConstr(movement_green_dur[movement_id] <= config['max_green'] - config['min_green'] - yellow,
                                name=constr_name_attach(signal_id, movement_id, 'max_green'))
                else:       # TODO: time-slot方法
                    m.addConstr(movement_green_dur[movement_id] <= config['max_green'],
                                name=constr_name_attach(signal_id, movement_id, 'max_green'))
                m.addConstr(movement_green_dur[movement_id] >= config['min_green'],
                            name=constr_name_attach(signal_id, movement_id, 'max_green'))
            # Constraint 5: Match the signal timing of movement and lane
            for movement_id, movement in lane.movements.items():
                # print(f'Combine green time of lane {lane_id} and movement {movement_id}')
                m.addConstr(movement_green_dur[movement_id] == lane_green_dur[lane_id],
                            name=constr_name_attach(signal_id, lane_id, 'lane_green_duration'))
            # Constraint 6: Judge whether the lane is spillover
            if config['peri_spillover_penalty']:
                alpha = config['spillover_threshold']
                m.addConstr(alpha * lane.length - lane_final_queue[lane_id] * lane.vehicle_length <= M * (1 - lane_spillover[lane_id]),
                            name=constr_name_attach(signal_id, lane_id, 'lane_spillover_1'))
                m.addConstr(alpha * lane.length - lane_final_queue[lane_id] * lane.vehicle_length >= M * lane_spillover[lane_id] + 1 / M,
                            name=constr_name_attach(signal_id, lane_id, 'lane_spillover_2'))

        m.optimize()

        if m.status == gp.GRB.OPTIMAL:
            # calculate the objective value
            inflow, queue_var = 0, 0
            for _, lane_id in self.peri_data.peri_inflow_lanes:
                inflow += lane_throughput[lane_id].x
            if self.distribution_mode == 'balance_queue':
                queue_var = total_squared_queue.x
            elif self.distribution_mode == 'equal':
                queue_var = total_squared_throughput.x
            # extract the optimal solution
            for _, lane_id in self.peri_data.peri_inflow_lanes:
                lane = self.peri_data.peri_inflow_lanes_by_laneID[lane_id]
                lane.green_duration = int(lane_green_dur[lane_id].x)
                for movement_id, movement in lane.movements.items():
                    movement.green_duration = int(movement_green_dur[movement_id].x)
                    for connection_id, connection in movement.connections.items():
                        connection.update_timing()
            # console output
            print(f'The estimated inflow by model is {inflow}. ')
            return inflow, queue_var
        elif m.status == gp.GRB.INFEASIBLE:
            # m.computeIIS()
            # for c in m.getConstrs():
            #     if c.IISConstr:
            #         print(c)
            # m.write('infeasible.ilp')
            print("The upper problem is infeasible. ")
            return 0, 0
        elif m.status == gp.GRB.UNBOUNDED:
            print("The upper problem is unbounded. ")
            return 0, 0
        else:
            print("No solution of the upper problem found in this step!")
            return 0, 0

    def set_local_green(self):
        '''
        输入：无（交叉口状态）
        固定变量: inflow方向绿灯时长
        决策变量：所有方向绿灯时长+绿灯启亮（+相序）
        目标：最大化吞吐量
        '''
        cycle, yellow = config['cycle_time'], config['yellow_duration']
        M = 1e5
        perimeter_total_throughput = 0     # 返回值

        for signal_id, signal in self.peri_data.peri_signals.items():
            # 对每个交叉口分别优化信号配时
            m = gp.Model('lower-signal')
            m.setParam('OutputFlag', 0)
            m.setParam(gp.ParamConstClass.DualReductions, 0)

            # Variables: queue length at the next step of each lane;
            # green start of each lane/movement; green duration of each lane/movement
            movement_green_start: Dict[str, gp.Var] = {}
            movement_green_dur: Dict[str, gp.Var] = {}
            movement_sequence: Dict[tuple, gp.Var] = {}
            lane_green_start: Dict[str, gp.Var] = {}
            lane_green_dur: Dict[str, gp.Var] = {}
            lane_throughput: Dict[str, gp.Var] = {}
            lane_green_discharge: Dict[str, gp.Var] = {}
            lane_green_demand: Dict[str, gp.Var] = {}
            # Add variables to each container
            for lane_id in signal.in_lanes.keys():
                var_green_start = m.addVar(lb=0, name=constr_name_attach(lane_id, 'green_start'))
                var_green_dur = m.addVar(lb=0, name=constr_name_attach(lane_id, 'green_duration'))
                var_throughput = m.addVar(lb=0, name=constr_name_attach(lane_id, 'throughput'))
                var_green_discharge = m.addVar(lb=0, name=constr_name_attach(lane_id, 'green_discharge'))
                var_green_demand = m.addVar(lb=0, name=constr_name_attach(lane_id, 'green_demand'))
                lane_green_start[lane_id] = var_green_start
                lane_green_dur[lane_id] = var_green_dur
                lane_throughput[lane_id] = var_throughput
                lane_green_discharge[lane_id] = var_green_discharge
                lane_green_demand[lane_id] = var_green_demand
            for movement_id, movement in signal.movements.items():
                var_green_start = m.addVar(lb=0, name=constr_name_attach(movement_id, 'green_start'))
                var_green_dur = m.addVar(lb=0, name=constr_name_attach(movement_id, 'green_duration'))
                # movement_type = movement.type
                movement_green_start[movement_id] = var_green_start
                movement_green_dur[movement_id] = var_green_dur
                for movement2_id, movement2 in signal.movements.items():
                    var_movement_sequence = m.addVar(vtype=gp.GRB.BINARY,
                                                     name=constr_name_attach(movement_id, movement2_id, 'sequence'))
                    movement_sequence[(movement_id, movement2_id)] = var_movement_sequence
            # Other variables
            total_throughput = m.addVar(lb=0, name='total_thrp')

            # obj: Maximize the (weighted) normalized throughput
            obj = 0
            m.addConstr(total_throughput == gp.quicksum(lane_throughput[lane_id] for lane_id in signal.in_lanes),
                        name='cal_thrp')
            obj -= total_throughput
            m.setObjective(obj)

            # Add constraints
            # Constraint 1: Lane throughput calculation
            for lane_id, lane in signal.in_lanes.items():
                m.addConstr(lane_green_discharge[lane_id] == lane_green_dur[lane_id] * lane.saturation_flow_rate,
                            name=constr_name_attach(signal_id, lane_id, 'green_discharge'))
                m.addConstr(lane_green_demand[lane_id] == lane.queue + lane.arrival_rate * (
                        lane_green_start[lane_id] + lane_green_dur[lane_id]),
                            name=constr_name_attach(signal_id, lane_id, 'green_demand'))
                m.addConstr(lane_throughput[lane_id] == gp.min_(lane_green_discharge[lane_id], lane_green_demand[lane_id]),
                            name=constr_name_attach(signal_id, lane_id, 'throughput'))
            # Constraint 2: Downstream capacity constraint
            for lane_id, lane in signal.in_lanes.items():
                m.addConstr(lane_throughput[lane_id] <= lane.get_downstream_capacity(),
                            name=constr_name_attach(signal_id, lane_id, 'down_capacity'))
            # Constraint 3: Minimum / Maximum green duration
            for movement_id, movement in signal.movements.items():
                m.addConstr(movement_green_dur[movement_id] <= config['max_green'],
                            name=constr_name_attach(signal_id, movement_id, 'max_green'))
                m.addConstr(movement_green_dur[movement_id] >= config['min_green'],
                            name=constr_name_attach(signal_id, movement_id, 'max_green'))
            # Constraint 4: Phase order (Signal timing of movement)
            # Constraint 4.1: Phase order for NEMA
            if self.signal_phase_mode == 'NEMA':
                phase_order = self.peri_data.peri_info[signal_id]['nema_plan']
                barrier_movement = []  # 每个ring的第二个barrier的首个movement
                for ring_name, ring in phase_order.items():
                    for idx in range(0, len(ring) - 1):
                        pred_movement, succ_movement = ring[idx], ring[idx + 1]
                        # 4.1.1 start and end of the cycle
                        if idx == 0:
                            m.addConstr(movement_green_start[pred_movement] == 0,
                                        name=constr_name_attach(ring_name, 'start'))
                        elif idx == len(ring) - 2:
                            m.addConstr(movement_green_start[succ_movement] + movement_green_dur[
                                succ_movement] + yellow == signal.cycle,
                                        name=constr_name_attach(ring_name, 'end'))
                        # 4.1.2 normal phase order
                        if pred_movement == '':
                            barrier_movement.append(succ_movement)
                            continue
                        if succ_movement == '':
                            succ_movement = ring[idx + 2]
                        m.addConstr(movement_green_start[pred_movement] + movement_green_dur[
                            pred_movement] + yellow == movement_green_start[succ_movement],
                                    name=constr_name_attach(ring_name, str(idx)))
                # 4.1.3 barrier time
                ring1_second_barrier_movement, ring2_second_barrier_movement = barrier_movement
                m.addConstr(movement_green_start[ring1_second_barrier_movement] == movement_green_start[
                            ring2_second_barrier_movement], name='barrier')
            # Constraint 4.2: Phase order for unfixed structure
            elif self.signal_phase_mode == 'Unfixed':
                conflict_matrix = signal.conflict_matrix
                # 4.2.1 First, constraint the end of green duration
                for movement_id, movement in signal.movements.items():
                    m.addConstr(movement_green_start[movement_id] + movement_green_dur[movement_id] + yellow <= signal.cycle,
                                name=constr_name_attach(signal_id, movement_id, 'green_end'))
                    # 4.2.2 Then, avoid the conflict
                    for movement2_id, movement2 in signal.movements.items():
                        if movement_id != movement2_id:
                            m.addConstr(movement_sequence[(movement_id, movement2_id)] +
                                        movement_sequence[(movement2_id, movement_id)] == 1,
                                        name=constr_name_attach(signal_id, movement_id, movement2_id, 'sequence'))
                            m.addConstr(
                                movement_green_start[movement_id] + movement_green_dur[movement_id] + yellow -
                                movement_green_start[movement2_id] - signal.cycle * movement_sequence[(
                                    movement_id, movement2_id)] <= M * (1 - conflict_matrix[movement_id][movement2_id]),
                                name=constr_name_attach(signal_id, movement_id, movement2_id, 'conflict'))
            # Constraint 5: Signal timing of right-turn movement, same as that of through movement
            for movement_id, movement in signal.movements.items():
                if movement.dir == 'r':
                    through_movement = movement_id[:-1] + 's'
                    m.addConstr(movement_green_dur[movement_id] == movement_green_dur[through_movement],
                                name=constr_name_attach(signal_id, movement_id, 'inflow_right_greendur'))
                    m.addConstr(movement_green_start[movement_id] == movement_green_start[through_movement],
                        name=constr_name_attach(signal_id, movement_id, 'inflow_right_greenstart'))
            # Constraint 6: Match the signal timing of movement and lane
            for lane_id, lane in signal.in_lanes.items():
                for movement_id, movement in lane.movements.items():
                    # print(f'Combine green time of lane {lane_id} and movement {movement_id}')
                    m.addConstr(movement_green_dur[movement_id] == lane_green_dur[lane_id],
                                name=constr_name_attach(signal_id, lane_id, 'lane_green_duration'))
                    m.addConstr(movement_green_start[movement_id] == lane_green_start[lane_id],
                                name=constr_name_attach(signal_id, lane_id, 'lane_green_start'))
            # Constraint 7: Fixed the green duration of the inflow movements
            for movement_id, movement in signal.movements.items():
                if movement.type == 'inflow':
                    print(f'The green duration of movement {movement_id} is {movement.green_duration}')
                    m.addConstr(movement_green_dur[movement_id] == movement.green_duration,
                                name=constr_name_attach(signal_id, movement_id, 'inflow_fixed_green_duration'))

            m.optimize()

            if m.status == gp.GRB.OPTIMAL:
                # get the objective value
                throughput_result = total_throughput.x
                perimeter_total_throughput += throughput_result
                # extract the optimal solution
                for movement_id, movement in signal.movements.items():
                    movement.green_start = int(movement_green_start[movement_id].x)
                    movement.green_duration = int(movement_green_dur[movement_id].x)
                for lane_id, lane in signal.in_lanes.items():
                    lane.green_start = int(lane_green_start[lane_id].x)
                    lane.green_duration = int(lane_green_dur[lane_id].x)
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


class TimeSlotPeriSignalController(PeriSignalController):

    def __init__(self, peri_data: PeriSignals, action: float, slot_num: int, cycle_num: int):
        super(TimeSlotPeriSignalController, self).__init__(peri_data, action)
        self.time_slot_num = slot_num
        self.cycle_num = cycle_num          # 大周期（Slot-based的实际周期）所包含的小周期（基础cycle）的个数

    # 重写分布式各交叉口slot-based配时确定方法
    def set_local_green(self):
        '''
        输入：无（交叉口状态）
        固定变量: inflow方向绿灯时长
        决策变量：所有方向绿灯时长+绿灯启亮（+相序）
        目标：最大化吞吐量
        '''
        cycle, yellow = config['cycle_time'], config['yellow_duration']
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
            slot_green_start: List[gp.Var] = []     # green start time of each slot
            slot_green_duration: List[gp.Var] = []  # green duration of each slot
            slot_stage_selection: Dict[str, List[gp.Var]] = {}
            slot_movement_selection: Dict[str, List[gp.Var]] = {}
            slot_movement_merge: Dict[str, List[gp.Var]] = {}
            slot_movement_green_duration: Dict[str, List[gp.Var]] = {}
            slot_lane_selection: Dict[str, List[gp.Var]] = {}
            slot_lane_queue: Dict[str, List[gp.Var]] = {}       # The lane queue at the end of each time slot
            slot_lane_throughput: Dict[str, List[gp.Var]] = {}  # The real throughput within each time slot
            slot_lane_green_throughput: Dict[str, List[gp.Var]] = {}
            slot_lane_green_duration: Dict[str, List[gp.Var]] = {}
            lane_throughput: Dict[str, gp.Var] = {}

            # Add variables to each container
            for i in range(self.time_slot_num):
                var_slot_green_start = m.addVar(lb=0, name=constr_name_attach('slot' + str(i), 'green_start'))
                var_slot_green_dur = m.addVar(lb=0, name=constr_name_attach('slot' + str(i), 'green_duration'))
                slot_green_start.append(var_slot_green_start)
                slot_green_duration.append(var_slot_green_dur)
                for stage_id in signal.stages.keys():
                    var_slot_stage = m.addVar(vtype=gp.GRB.BINARY, name=constr_name_attach('slot' + str(i), 'stage'))
                    slot_stage_selection.setdefault(stage_id, []).append(var_slot_stage)
                for movement_id in signal.movements.keys():
                    var_slot_movement = m.addVar(vtype=gp.GRB.BINARY,
                                                 name=constr_name_attach('slot' + str(i), movement_id, 'movement'))
                    var_slot_movement_merge = m.addVar(vtype=gp.GRB.BINARY,
                                                       name=constr_name_attach('slot' + str(i), movement_id, 'merge'))
                    var_slot_movement_green_dur = m.addVar(lb=0,
                                                           name=constr_name_attach('slot' + str(i), movement_id, 'green'))
                    slot_movement_selection.setdefault(movement_id, []).append(var_slot_movement)
                    slot_movement_merge.setdefault(movement_id, []).append(var_slot_movement_merge)
                    slot_movement_green_duration.setdefault(movement_id, []).append(var_slot_movement_green_dur)
                for lane_id in signal.in_lanes.keys():
                    var_slot_queue = m.addVar(lb=0, name=constr_name_attach('slot'+str(i), lane_id, 'queue'))
                    var_slot_throughput = m.addVar(lb=0, name=constr_name_attach('slot'+str(i), lane_id, 'throughput'))     # 真实output
                    var_slot_green_throughput = m.addVar(
                        lb=0, name=constr_name_attach('slot'+str(i), lane_id, 'green_throughput'))      # 如果是绿灯的output
                    var_slot_lane_green_dur = m.addVar(lb=0, name=constr_name_attach('slot' + str(i), lane_id, 'green'))
                    slot_lane_queue.setdefault(lane_id, []).append(var_slot_queue)
                    slot_lane_throughput.setdefault(lane_id, []).append(var_slot_throughput)
                    slot_lane_green_throughput.setdefault(lane_id, []).append(var_slot_green_throughput)
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
                    if i == 0:      # The first time slot
                        m.addConstr(
                            slot_lane_queue[lane_id][i] == lane.queue + lane.arrival_rate * slot_green_duration[i] -
                            slot_lane_throughput[lane_id][i],
                            name=constr_name_attach('slot' + str(i), lane_id, 'queue'))
                    else:
                        m.addConstr(
                            slot_lane_queue[lane_id][i] == slot_lane_queue[lane_id][i-1] + lane.arrival_rate *
                            slot_green_duration[i] - slot_lane_throughput[lane_id][i-1],
                            name=constr_name_attach('slot' + str(i), lane_id, 'queue'))
            # Constraint 2: Throughput of each time slot
            for i in range(self.time_slot_num):
                for lane_id, lane in signal.in_lanes.items():
                    if i == 0: # The first time slot
                        m.addConstr(slot_lane_green_throughput[lane_id][i] == gp.min_(
                            slot_green_duration[i] * lane.saturation_flow_rate,
                            lane.queue + lane.arrival_rate * slot_green_duration[i]),
                                    name=constr_name_attach('slot' + str(i), lane_id, 'green_throughput'))
                    else:
                        m.addConstr(slot_lane_green_throughput[lane_id][i] == gp.min_(
                            slot_green_duration[i] * lane.saturation_flow_rate,
                            slot_lane_queue[lane_id][i] + lane.arrival_rate * slot_green_duration[i]),
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
                m.addConstr(gp.quicksum(stage_list) == 1, name=constr_name_attach('slot' + str(i), 'stage_selection'))
            # Constraint 7: The same stage is not selected consequently
            for i in range(self.time_slot_num - 1):
                for stage_id in signal.stages.keys():
                    m.addConstr(slot_stage_selection[stage_id][i] <= M * (1 - slot_stage_selection[stage_id][i + 1]),
                                name=constr_name_attach('slot' + str(i), 'stage_change1'))
                    m.addConstr(slot_stage_selection[stage_id][i] >= -M * (1 - slot_stage_selection[stage_id][i + 1]),
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
                    m.addConstr(1 - slot_movement_merge[movement_id][i] == gp.abs_(
                        slot_movement_selection[movement_id][i] - slot_movement_selection[movement_id][i + 1]),
                                name=constr_name_attach('slot' + str(i), movement_id, 'merge'))
            # Constraint 10: The green time sequence
            for i in range(self.time_slot_num - 1):
                m.addConstr(slot_green_start[i + 1] == slot_green_start[i] + slot_green_duration[i],
                            name=constr_name_attach('slot' + str(i), 'slot_sequence'))
            # Constraint 11: Boundary
            m.addConstr(slot_green_start[0] == 0, name='first_slot_green_start')
            m.addConstr(slot_green_start[self.time_slot_num - 1] + slot_green_duration[
                self.time_slot_num - 1] == self.cycle_num * signal.cycle,
                        name='last_slot_green_end')
            # Constraint 12: Minimum green
            for i in range(self.time_slot_num):
                m.addConstr(slot_green_duration[i] >= config['min_green'] + yellow,
                            name=constr_name_attach('slot' + str(i), 'min_slot_duration'))
            # Constraint 13: The green time of movement and lane
            for i in range(self.time_slot_num):
                for movement_id in signal.movements.keys():
                    # No right-of-way movement
                    m.addConstr(
                        slot_movement_green_duration[movement_id][i] <= M * slot_movement_selection[movement_id][i],
                        name=constr_name_attach('slot' + str(i), movement_id, 'no_green1'))
                    m.addConstr(
                        slot_movement_green_duration[movement_id][i] >= -M * slot_movement_selection[movement_id][i],
                        name=constr_name_attach('slot' + str(i), movement_id, 'no_green2'))
                    # Right-of-way movement
                    if i < self.time_slot_num - 1:      # Merged slot have no yellow time
                        m.addConstr(slot_movement_green_duration[movement_id][i] - slot_green_duration[i] <= M * (
                            2 - slot_movement_merge[movement_id][i] - slot_movement_selection[movement_id][i]),
                                    name=constr_name_attach('slot' + str(i), movement_id, 'merge_green1'))
                        m.addConstr(slot_movement_green_duration[movement_id][i] - slot_green_duration[i] >= -M * (
                            2 - slot_movement_merge[movement_id][i] - slot_movement_selection[movement_id][i]),
                                    name=constr_name_attach('slot' + str(i), movement_id, 'merge_green2'))
                        m.addConstr(
                            slot_movement_green_duration[movement_id][i] - slot_green_duration[i] + yellow <= M * (
                                slot_movement_merge[movement_id][i] + 1 - slot_movement_selection[movement_id][i]),
                            name=constr_name_attach('slot' + str(i), movement_id, 'unmerge_green1'))
                        m.addConstr(
                            slot_movement_green_duration[movement_id][i] - slot_green_duration[i] + yellow >= -M * (
                                slot_movement_merge[movement_id][i] + 1 - slot_movement_selection[movement_id][i]),
                            name=constr_name_attach('slot' + str(i), movement_id, 'unmerge_green2'))
                    else:       # The last slot always have yellow time (can be improved)
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
                            m.addConstr(slot_lane_green_duration[lane_id][i] == slot_movement_green_duration[movement_id][i],
                                        name=constr_name_attach('slot' + str(i), lane_id, 'green'))
            # Constraint 14: Fixed the duration of inflow movement
            for movement_id, movement in signal.movements.items():
                if movement.type == 'inflow':
                    m.addConstr(gp.quicksum(slot_movement_green_duration[movement_id]) == movement.green_duration,
                                name=constr_name_attach(movement_id, 'inflow_fixed_green_duration'))

            m.optimize()

            if m.status == gp.GRB.OPTIMAL:
                # get the objective value
                throughput_result = total_throughput.x
                perimeter_total_throughput += throughput_result
                # extract the optimal solution
                for i in range(self.time_slot_num):
                    green_start = slot_green_start[i].x
                    for movement_id, movement in signal.movements.items():
                        # get the green time for the right-of-way movements
                        if slot_movement_selection[movement_id][i].x == 1:
                            if not isinstance(movement.green_start, list):
                                movement.green_start = []
                            movement.green_start.append(green_start)
                            green_duration = slot_movement_green_duration[movement_id][i].x
                            if not isinstance(movement.green_duration, list):
                                movement.green_duration = []
                            movement.green_duration.append(green_duration)
                            # get the yellow time for the end of the right-of-way movements
                            if slot_movement_merge[movement_id][i].x == 0:
                                movement.yellow_start.append(green_start + green_duration)
                    for lane_id, lane in signal.in_lanes.items():
                        # get the green time for the right-of-way lanes
                        if slot_lane_selection[lane_id][i].x == 1:
                            if not isinstance(lane.green_start, list):
                                lane.green_start = []
                            lane.green_start.append(green_start)
                            green_duration = slot_lane_green_duration[lane_id][i].x
                            if not isinstance(lane.green_duration, list):
                                lane.green_duration = []
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






if __name__ == '__main__':

    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    peridata = PeriSignals(config['netfile_dir'], sumo_cmd)
    peridata.get_basic_inform()
    peridata.get_conflict_matrix()



    cycle = 100
    total_flow = 0
    # # conflict_matrix = np.array([[0, 0, 0, 0, 0, 0],
    # #                             [0, 0, 0, 0, 0, 0],
    # #                             [0, 0, 0, 0, 0, 0],
    # #                             [0, 0, 0, 0, 0, 0],
    # #                             [0, 0, 0, 0, 0, 0],
    # #                             [0, 0, 0, 0, 0, 0]], np.int32)  # 交叉口A直/左 交叉口B直/左 交叉口C直 交叉口D直
    # sfr_array = np.array([1200] * 6, np.int32).reshape(-1, 1) / 3600  # in veh/s
    # demand_array = np.array([300] * 6, np.int32).reshape(-1, 1) / 3600  # in veh/s
    # queue_array = np.array([5, 10, 15] * 2, np.int32).reshape(-1, 1)
    #
    # time_start = time.time()
    # green_duration = gated_green_duration(cycle=cycle, total_inflow=total_flow,
    #                                       gated_sfr=sfr_array, gated_demand=demand_array, initial_queue=queue_array,
    #                                       min_green=10, max_green=50)
    # green_duration = np.array([6, 49, 24, 5, 41, 44, 6, 49, 24]).T
    # # standard_green_duration = np.array([1, 50, 24, 8, 41, 46]).T
    # conflict_matrix = np.zeros((9, 9), np.int32)
    # clearance_time = 5.0
    # sfr_array = np.array([1600, 1800, 1600] * 3, np.int32).reshape(-1, 1) / 3600  # in veh/s
    # green_start = gated_green_start(cycle=cycle, gated_sfr=sfr_array, green_duration=green_duration,
    #                                 conflict_matrix=conflict_matrix, clearance_time=clearance_time)
    # time_end = time.time()
    # time_duration = time_end - time_start
    # print("Calculation time: ", time_duration)
    # print("Gated movement green duration: ", green_duration)
    # print("Gated movement green start: ", green_start)
    # # print("Gated movement green overlap: ", green_overlap)
    # # print("gamma: ", g)
    # # print("delta: ", d)
    # # print("omega: ", o)
    # # print("theta: ", t)
