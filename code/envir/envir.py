from copy import copy
import sumolib
import traci
from utils.trafficsignalcontroller import TrafficSignalController
from peritsc.perimeterdata import PeriSignals
from utils.result_processing import plot_MFD
import numpy as np
from utils.genDemandBuffer import TrafficGenerator
from utils.genDemandFromTurn import TrafficGeneratorFromTurn
from envir.perimeter import Peri_Agent
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt
from collections import deque
from metric.metric import Metric
import datetime
import platform
import time
from typing import Dict, Union
from copy import deepcopy

class Simulator():
    def __init__(self, TrafficGen, netdata, peridata: PeriSignals, config):
        # self.min_green = config['min_green']
        # self.max_green = config['max_green']
        # self.states = config['states']
        self.yellow_duration = config['yellow_duration']
        self.max_steps = config['max_steps']
        self.TrafficGen: Union[TrafficGenerator, TrafficGeneratorFromTurn] = TrafficGen
        self.state_steps = config['state_steps']
        self.netdata = netdata
        self.peridata: PeriSignals = peridata
        self.config = config

        ''' -info_interval:  10 sec for data collection 
            -control_interval: 100sec for a new action
            -simulation_interval: 1sec for SUMO simulation
            -peri_record_interval: 1sec for peri data collection
        '''
        self.control_interval = config['control_interval']
        self.info_interval = config['infostep']

        self.lower_mode = config['lower_mode']

        self.vehicle_loc = {}
        self.vehicle_routes = {}

    def simu_start(self, sumo_cmd):
        traci.start(sumo_cmd)
        # print("Simulating...")
        return None

    def reset(self, sumo_cmd, config):
        """ reset env: initialize the perimeter control, generate demand, and start sumo
        """
        ######## must generate demand before traci.start ########

        ## 1. reset peremiters
        # self.Perimeter = Peri_Agent()
        # self.peri_num = len(self.Perimeter.info)
        # self.info_update_index = 0

        ## 1. reset demand
        if config['network'] == 'FullGrid':
            self.TrafficGen.generate_flow(config)
            self.TrafficGen.generate_flow_file(config)
            self.TrafficGen.generate_trip_file(config)
        else:
            self.TrafficGen.gen_demand(config)


        ## 2. start sumo
        self.simu_start(sumo_cmd)

        ## 3. reset subscribe
        self.set_subscribe()

    def set_subscribe(self):
        ## lane data subscribe
        pass
        # for lane in (self.netdata['lane']).keys():
        #     traci.lane.subscribe(lane, [traci.constants.LAST_STEP_MEAN_SPEED, 
        #                                 traci.constants.LAST_STEP_VEHICLE_NUMBER,
        #                                 traci.constants.LAST_STEP_OCCUPANCY])


        # ## edge data subscribe
        # for edge in (self.netdata['edge']).keys():
        #     traci.edge.subscribe(edge, [traci.constants.LAST_STEP_MEAN_SPEED, 
        #                                 traci.constants.LAST_STEP_VEHICLE_NUMBER,
        #                                 traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER])


    def simu_run(self, step):
        """ simulation with one lower level control interval
        """

        self._step = step   # 当前仿真step

        ## 1. simulate yellow phase
        if self.lower_mode != 'FixTime':
            self._set_phase('yellow', self.yellow_duration)
        self._simulate(int(self.yellow_duration))

        ## 2. simulate green phase
        rest_interval_sec = self.control_interval - self.yellow_duration
        if self.lower_mode != 'FixTime':
            self._set_phase('green', rest_interval_sec)
        self._simulate(int(rest_interval_sec))

        done = False
        if self._step >= self.max_steps:
            done = True
            # print(self.vehicle_loc)
            # for signal_id, signal in self.peridata.peri_signals.items():
            #     for lane_id, lane in signal.in_lanes.items():
            #         print(f'Queue on lane {lane_id} at the end of simulation: {lane.queueing_vehicles}')

        ## get data output
        #  1. for OAM
        #  2. for upper level update   
        lane_data, edge_data = None, None

        # lane_data = traci.lane.getAllSubscriptionResults()
        # edge_data = traci.edge.getAllSubscriptionResults()


        return self._step, done, lane_data, edge_data


    def _set_phase(self, phase_type, phase_duration):
        for tl_id in self.agent_lower.controled_light:
            phase = self._get_node_phase(tl_id, phase_type)
            traci.trafficlight.setRedYellowGreenState(tl_id, phase)     # programID will be set to 'online'
            traci.trafficlight.setPhaseDuration(tl_id, phase_duration)

    def _get_node_phase(self, tl_id, phase_type):
        ''' get the phase of the node given the signal type 
        '''

        tsc = self.agent_lower.tsc[tl_id]
        cur_phase = tsc.cur_phase
        prev_phase = tsc.prev_phase

        ## 1. green phase -- keep the current phase
        if phase_type == 'green':
            return cur_phase

        ## 2. no change of the phase -- no amber, maintain current
        if prev_phase == cur_phase:  # no amber, maintain
            return cur_phase

        ## 3. signal changed, set the yellow phase
        switch_reds = []
        switch_greens = []
        for i, (p0, p1) in enumerate(zip(prev_phase, cur_phase)):
            if (p0 in 'Gg') and (p1 == 'r'):
                switch_reds.append(i)
            elif (p0 in 'r') and (p1 in 'Gg'):
                switch_greens.append(i)
        if not len(switch_reds):
            return cur_phase

        yellow_phase = list(cur_phase)
        for i in switch_reds:
            yellow_phase[i] = 'y'
        for i in switch_greens:
            yellow_phase[i] = 'r'
        return ''.join(yellow_phase)

    def _get_peri_state(self):
        '''
        Update the halting vehicle and inflow of peri lanes at each step within one interval.
        Update the (approximate) outflow of peri lanes of each interval with induction loops.
        '''
        if self._step % 10 == 0:
            for signal_id, signal in self.peridata.peri_signals.items():
                for lane_id, lane in signal.in_lanes.items():
                    # 记录停车车辆数 (用于更新critical point)
                    halt_num = traci.lane.getLastStepHaltingNumber(lane_id)
                    if halt_num > lane.this_interval_max_halting_number:
                        lane.this_interval_max_halting_number = halt_num

                    edge = lane.edge
                    # 更新个体车辆信息
                    vehicles: list = traci.lane.getLastStepVehicleIDs(lane_id)
                    for vehicle_id in vehicles:
                        # 车辆信息
                        speed = traci.vehicle.getSpeed(vehicle_id)
                        pos = traci.vehicle.getLanePosition(vehicle_id)
                        if vehicle_id not in self.vehicle_routes:
                            self.vehicle_routes[vehicle_id] = traci.vehicle.getRoute(vehicle_id)

                        if speed < 0.1 or pos > lane.length - self.config['record_distance']:
                            route = self.vehicle_routes[vehicle_id]
                            assert edge in route
                            edge_idx = route.index(edge)
                            if edge_idx == len(route) - 1:      # 到达终点edge
                                self.vehicle_loc.pop(vehicle_id, None)
                                self.vehicle_routes.pop(vehicle_id, None)
                                continue

                            next_edge = route[edge_idx + 1]  # 该车辆下一个edge
                            # 根据route确定车辆实际所属车道
                            if next_edge in lane.downstream_edges:
                                target_lane_id, target_lane = lane_id, lane
                            else:
                                for idx in range(2, -1, -1):
                                    test_lane_id = '_'.join((edge, str(idx)))
                                    test_lane = signal.in_lanes[test_lane_id]
                                    if next_edge in test_lane.downstream_edges:
                                        target_lane_id, target_lane = test_lane_id, test_lane
                                        break

                            # queue veh
                            # 可能导致同一车辆同时出现在queue和virtual queue, 但可通过后面更新过程清理
                            if speed < 0.1 and pos > 10:
                                if target_lane_id == lane_id:
                                    target_lane.queueing_vehicles.add(vehicle_id)
                                else:
                                    target_lane.virtual_queue_vehicles.add(vehicle_id)

                            # entered veh (arrival rate)
                            if pos > lane.length - self.config['record_distance']:  # 不记录过于上游的车辆
                                if vehicle_id not in self.vehicle_loc:
                                    target_lane.entered_vehicles.add(vehicle_id)    # 直接计入target lane, 避免变道问题
                                    self.vehicle_loc[vehicle_id] = [signal_id, target_lane_id]
                                else:  # 已在路网上的车辆
                                    former_signal_id, former_lane_id = self.vehicle_loc[vehicle_id]
                                    if former_signal_id != signal_id:  # 进入下一个link
                                        target_lane.entered_vehicles.add(vehicle_id)
                                        self.vehicle_loc[vehicle_id] = [signal_id, target_lane_id]

        if self._step % self.config['updatestep'] == 0:
            # 每个interval结束时, 更新queue_vehicle, 删除离开该lane的车辆
            for signal_id, signal in self.peridata.peri_signals.items():
                for lane_id, lane in signal.in_lanes.items():
                    lane_current_vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                    edge_current_vehicles = traci.edge.getLastStepVehicleIDs(lane_id.split('_')[0])

                    for vehicle in set(lane.queueing_vehicles):
                        if vehicle not in edge_current_vehicles:      # 进入下一个link
                            lane.queueing_vehicles.discard(vehicle)
                        else:
                            if vehicle not in lane_current_vehicles:        # 变道到其他lane
                                lane.queueing_vehicles.discard(vehicle)
                    for vehicle in set(lane.virtual_queue_vehicles):
                        if vehicle not in edge_current_vehicles:      # 进入下一个link
                            lane.virtual_queue_vehicles.discard(vehicle)
                        else:
                            if vehicle in lane_current_vehicles:            # 变道到target lane
                                lane.virtual_queue_vehicles.discard(vehicle)
                                lane.queueing_vehicles.add(vehicle)

        if self._step % self.config['detector_interval'] == 0:
            # 每个检测器interval结束时记录车道的驶离车辆数（用于判断特殊情况, 可以有1-2辆的误差）
            for signal_id, signal in self.peridata.peri_signals.items():
                for lane_id, lane in signal.in_lanes.items():
                    detector_id = lane_id + '_out'
                    lane_outflow = traci.inductionloop.getLastIntervalVehicleNumber(detector_id)
                    lane.outflow_vehicle_num += lane_outflow
            # out-out/in-out类车辆驶出PN后删除route记录
            for edge in self.config['Edge_Peri_out']:
                for lane in range(3):
                    detector_id = '_'.join((str(edge), str(lane), 'in'))
                    pass_vehs = traci.inductionloop.getLastIntervalVehicleIDs(detector_id)
                    for veh in pass_vehs:
                        self.vehicle_loc.pop(veh, None)
                        self.vehicle_routes.pop(veh, None)

        if self._step % self.config['cycle_time'] == 0:
            # 周期结束时获取出口道的最末端停车车辆位置
            for signal_id, signal in self.peridata.peri_signals.items():
                for lane_id, lane in signal.out_lanes.items():
                    vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                    for vehicle_id in vehicles:
                        speed = traci.vehicle.getSpeed(vehicle_id)
                        pos = traci.vehicle.getLanePosition(vehicle_id)
                        if speed < 0.1 and pos > 10:    # 排除刚生成的车辆
                            if pos < lane.last_halt_position:
                                lane.last_halt_position = pos

    def _simulate(self, num_step):
        # starttime = time.perf_counter()

        for _ in range(num_step):
            traci.simulationStep()
            self._step += 1
            self._get_peri_state()

        # endtime = time.perf_counter()
        # # endtime = datetime.datetime.now()
        # run_time = (endtime - starttime)*1000

        # print('程序运行时间:%s毫秒' % (run_time))

    ### Env: action, reward, state
    def get_stacked_states(self):
        ''' obtain the stacked state of the controller: list as input to NN 
        '''
        # get stacked states
        state = []
        for k in self.states:
            state += list(self.state_dict[k])
        return state


if False:
    def get_state(self):
        ''' obtain state of the controller: store in the state_dict
        '''
        state = []
        for state_type in self.states:

            ### 1.accumulation of PN after normalization
            if state_type == 'accu':
                accu_PN, _ = self.Metric.get_accu(info_inter_flag=True)
                accu_PN = accu_PN / self.config['accu_max'] # normalize
                # self.state_dict['accu'].append(accu_PN)
                state.append(accu_PN)

            ### 2. accumulation of buffer after normalization
            elif state_type == 'accu_buffer':
                _, accu_buffer = self.Metric.get_accu(info_inter_flag=True)
                accu_buffer = accu_buffer / self.config['accu_buffer_max'] # normalize
                # self.state_dict['accu_buffer'].append(accu_buffer)
                state.append(accu_buffer)

            ### 3. occupancy of the downlane in the perimeter
            elif state_type == 'down_edge_occupancy':
                self.Perimeter.get_down_edge_occupancy()
                # self.state_dict['down_edge_occupancy'].append(
                #     self.Perimeter.down_edge_occupancy)
                state.extend(self.Perimeter.down_edge_occupancy)


            ### 4. average occupancy of the buffer links
            elif state_type == 'buffer_aver_occupancy':
                self.Perimeter.get_buffer_average_occupancy()
                state.extend(self.Perimeter.buffer_average_occupancy)
                # self.state_dict['buffer_aver_occupancy'].append(
                #     self.Perimeter.buffer_average_occupancy)

            ### 5. demand of next step
            elif state_type == 'future_demand':
                cycle_index = int(self.info_update_index/self.Metric.info_length)
                demand_nextstep = self.Metric.get_demand_nextstep(cycle_index)
                demand_nextstep = demand_nextstep / self.config[
                    'Demand_state_max']  # normalization
                # self.state_dict['future_demand'].append(demand_nextstep)
                state.append(demand_nextstep)

            ### 6. entered vehicles from perimeter
            elif state_type == 'entered_vehs':
                entered_veh = self.Metric.get_entered_veh_control_interval()
                entered_veh = entered_veh / self.config['entered_veh_max'] # normalize
                # self.state_dict['entered_vehs'].append(entered_veh) 
                state.append(entered_veh)


            ### 7. network mean speed
            elif state_type == 'network_mean_speed':
                network_mean_speed, _ = self.Metric.get_PN_speed_production(info_inter_flag=True)
                network_mean_speed = network_mean_speed / self.config['network_mean_speed_max'] # normalize
                # self.state_dict['network_mean_speed'].append(network_mean_speed)
                state.append(network_mean_speed)


            ### 8. network PN halting vehicles 
            elif state_type == 'network_halting_vehicles':
                _, PN_halt_vehs = self.Metric.get_halting_vehs(info_inter_flag=True)
                PN_halt_vehs = PN_halt_vehs/self.config['PN_halt_vehs_max']
                # self.state_dict['network_halting_vehicles'].append(PN_halt_vehs)
                state.append(PN_halt_vehs)


            ### 9. buffer halting vehicles 
            elif state_type == 'buffer_halting_vehicles':
                buffer_halt_vehs, _ = self.Metric.get_halting_vehs(info_inter_flag=True)
                # print(f'buffer_halt_vehs = {buffer_halt_vehs}')
                buffer_halt_vehs = buffer_halt_vehs/self.config['buffer_halt_vehs_max']
                # self.state_dict['network_halting_vehicles'].append(buffer_halt_vehs)
                state.append(buffer_halt_vehs)

        return state

    def action_coordinate_transformation(self, upper_bound, action):
        """ Conduct the coordinate transform given the bounded action
        """
        ## [-1 1]
        # medium =  (upper_bound -lower_bound) / 2 +lower_bound
        # action  = medium + action * (medium - lower_bound)
        # assert (action>=0).all()==True, 'the action is less than 0'
        # assert (action<=1).all()==True, 'the action is greater than 1'
        ## [0,1]
        if self.Perimeter.peri_action_mode == 'centralize': # centralized actions
            action = action * upper_bound * self.peri_num

        else: # decentralized actions
            action = action * upper_bound

        # print(f"actual ation: {action}")
        return action

    def simu_static_run(self, e, sumo_cmd):
        # self.TrafficGen.gen_demand()
        # self.simu_start(sumo_cmd)

        cycle_index = 0
        for t in range(self.max_steps):
            traci.simulationStep()
            # print(t)
            if t != 0 and (self.max_steps -
                           t) <= 10000 and (t) % self.cycle_time == 0:
                # process output file to get observations
                self.process_output(cycle_index)
                cycle_index += 1

                # get veh num on each edge
                self.get_edge_cur_veh_num()

                # get accumulation of the PN
                self.get_accu()

                # get reward
                reward = self.get_throughput()

        # print(self.accu_list)
        # print(self.reward_list)
        traci.close()
        plot_MFD(self.accu_list, self.throuput_list, self.cycle_time, e,
                 self.config['Peri_mode'])

    def simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        done = False

        # do not do more steps than the maximum allowed number of steps
        if (self._step + steps_todo) >= self.max_steps:
            steps_todo = self.max_steps - self._step
            done = True

        while steps_todo > 0:

            ## check the signal to switch
            self.Perimeter.switch_phase(self._step)

            ## simulate one step
            traci.simulationStep()  # simulate 1 step in sumo
            # update time index
            self._step += 1  # update the step counter
            # print(self._step)
            steps_todo -= 1

            ## Metric update 
            if self._step % self.info_interval == 0:
                self.Metric.update_info_interval(self.info_update_index, self.Perimeter.info, self.outputfile)
                self.info_update_index += 1

        # update metrics of the control interval
        self.Metric.update_control_interval()

        # get actual entered vehs from the perimeter
        entered_veh_control_interval = self.Metric.get_entered_veh_control_interval()

        # get reward
        reward = self.get_reward()

        # get penalty
        penalty = self.get_penalty(entered_veh_control_interval)

        # get state
        state = self.get_state()
        # state = self.get_stacked_states()


        return state, reward, penalty, done, entered_veh_control_interval

    def get_reward(self):
        ''' Collect reward for each simulation step
        '''
        ## 1. production within control interval ( speed * veh )
        _, production_control_interval  = self.Metric.get_PN_speed_production()
        # print(f'Production = {production_control_interval}')
        reward = production_control_interval / self.config['production_control_interval_max']

        return reward

    def get_penalty(self, entered_veh_control_interval):
        ''' Collect reward for each simulation step
        '''
        halveh_buffer, halveh_PN = self.Metric.get_halting_vehs()
        penalty = -(halveh_buffer/1000)**2

        penalty = np.clip(penalty, -4, 0)
        return penalty
