import traci
import numpy as np
from utils.utilize import config
from peritsc.perimeterdata import PeriSignals
from peritsc.signal_controller import PeriSignalController
from typing import Dict

# EdgeCross = [33034, 53054]


class Peri_Agent():
    def __init__(self, tsc_peri, peridata: PeriSignals):
        # control mode
        self.distribution_mode = config['peri_distribution_mode']
        self.signal_phase_mode = config['peri_signal_phase_mode']
        self.optimization_mode = config['peri_optimization_mode']

        # signal constraints
        self.min_green = config['min_green']
        self.max_green = config['max_green']
        self.yellow_duration = config['yellow_duration']
        self.cycle_time = config['cycle_time']
        self.downedge_maxveh = config['max_queue']

        # perimeter infos
        self.info = config['Peri_info']
        self.peri_num = len(self.info)
        self.splitmode = config['splitmode']  # 0 = waittime, 1 = waitveh
        self.peri_action_mode = config['peri_action_mode']
        self.flowrate = config['flow_rate']  # 1 veh /cross time
        self.tsc_peri = tsc_peri
        self.peridata = peridata
        # list to record datas
        self.green_time = np.array([])
        self.inflow_movements = []

        # self.entered_veh = np.array([])

    def get_waitfeature(self):
        """ get waiting features on the perimeters
        """
        waittime, waitveh = {}, {}
        for tlsID in self.info.keys():

            edge_num = self.info[tlsID]['edge']  # get the controlled edge
            waittime[edge_num] = traci.edge.getWaitingTime(
                str(edge_num))  # waiting times on the edge
            waitveh[edge_num] = traci.edge.getLastStepHaltingNumber(
                str(edge_num))  # waiting vehs

        # print(waittime, waitveh)
        self.wait_feature = (waittime, waitveh
                             )  # store all the dicts into the tuple

        # return wait_feature

    def get_greensplit(self, action, steps):
        """
        distribute the vehicles on perimeter nodes with green split
        """
        green_time = np.zeros(len(self.info))

        if self.peri_action_mode == 'centralize':
            ############ consider upstream waiting features
            if False:
                # First, get wait feature
                self.get_waitfeature()

                # Secend, calcualte green split
                ### waitfeature i\
                # s veh number
                wait_feature = self.wait_feature[self.splitmode]  # choose split mode
                print(f"wating vehicles: {wait_feature}")

                wait_value = np.array(list(
                    wait_feature.values()))  # get values of each crossedge

                wait_value = np.maximum(wait_value, self.min_green / self.flowrate)  # minimum green constraint, in case of NaN
                # print(wait_value)
                ratio = wait_value / sum(
                    wait_value)  # split ratio of the permitted vehs
                # print(ratio)
                permit_vehnum = action * ratio  # calculate the permit vehs on each perimeter
                # print(permit_vehnum)
                green_time = (
                    permit_vehnum *
                    self.flowrate).round()  # transfer: veh num --> green time

            ############# directly split green time given total inflow vehicles
            elif False:
                green_time[:] = (action * self.flowrate / len(self.info)).round()

            ############# directly split green time given total green time
            else:
                green_time[:] = round((action / len(self.info)), 0)

        else:  # decentralized
            green_time[:] = np.around(action)

        # print(green_time)
        green_time = np.minimum(green_time, self.max_green)
        green_time = np.maximum(green_time, self.min_green)
        # green_time = np.maximum(green_time, 0)

        red_time = self.cycle_time - green_time - 2 * self.yellow_duration

        self.green_time = np.reshape(np.append(self.green_time, green_time), [-1, len(self.info)])
        # print(self.green_time)

        green_time = dict(zip(config['Peri_info'].keys(), green_time))
        red_time = dict(zip(config['Peri_info'].keys(), red_time))

        return green_time, red_time
        # print(f"action of greentime :{green_time}")

    def get_full_greensplit(self, action, peri_state: Dict[str, Dict[str, list]]):
        """
        distribute the vehicles on perimeter nodes with green split
        """
        # 1. Collect critical parameters (demand & initial queue)
        for signal_id, signal in self.peridata.peri_signals.items():
            for lane_id, lane in signal.in_lanes.items():
                if len(peri_state):
                    # calculate the real queue length at the implementation step
                    # assume that the queue length reaches maximum at the end of red
                    halting_veh_evolution = peri_state['halting'][lane_id]
                    entered_veh_evolution = peri_state['entered'][lane_id]
                    left_veh_evolution = peri_state['left'][lane_id]
                    # print(f'The evolution of halting vehicles on lane {lane_id}: {halting_veh_evolution}')
                    # print(f'The evolution of entered vehicles on lane {lane_id}: {entered_veh_evolution}')
                    # print(f'The evolution of left vehicles on lane {lane_id}: {left_veh_evolution}')
                    red_end_step = halting_veh_evolution.index(max(halting_veh_evolution))
                    total_newly_entered_veh = sum(entered_veh_evolution[red_end_step:])     # 假设一个cycle内到达率不变
                    total_newly_left_veh = sum(left_veh_evolution[red_end_step:])
                    flow = sum(entered_veh_evolution) / self.cycle_time     # veh/s
                    queue = max(halting_veh_evolution) + total_newly_entered_veh - total_newly_left_veh
                    queue = max(queue, 0)
                else:   # step = 0
                    density = traci.lane.getLastStepVehicleNumber(lane_id) / lane.length
                    speed = traci.lane.getLastStepMeanSpeed(lane_id)
                    flow = density * speed      # veh/s
                    queue = traci.lane.getLastStepHaltingNumber(lane_id)
                avg_vehicle_length = traci.lane.getLastStepLength(lane_id)
                lane.set_arrival_rate(flow)
                lane.set_queue(queue)
                lane.set_vehicle_length(avg_vehicle_length)
            for movement_id, movement in signal.movements.items():
                downlink_id = movement.down_link.id
                queue = traci.edge.getLastStepHaltingNumber(downlink_id)
                signal.downLinks[downlink_id].update_state(queue)

        # 2. Optimize the signal plan of all perimeter intersections
        controller = PeriSignalController(self.peridata, action)
        estimate_inflow = controller.signal_optimize()

        # # centralized method
        # # set_inflow_local_green(peri_data=self.peridata, required_total_inflow=action)
        #
        # # decentralized method
        # set_inflow_green(peri_data=self.peridata, required_total_inflow=action)
        # set_local_green(peri_data=self.peridata)

        # 3. record green time data
        inflow_movement_green_time = self.peridata.get_inflow_green_duration()
        self.inflow_movements = list(inflow_movement_green_time.keys())
        self.green_time = np.reshape(np.append(self.green_time, list(inflow_movement_green_time.values())),
                                     [-1, len(inflow_movement_green_time)])

        return estimate_inflow

    def set_program(self, green_time, red_time):
        for peri_id, peri in dict.items(self.info):
            tsc = peri['tsc']
            logic = tsc.logic

            ## set green and red time
            logic.phases[tsc.green_phase_index].duration = green_time[peri_id]
            logic.phases[tsc.red_phase_index].duration = red_time[peri_id]

            ## set program
            traci.trafficlight.setProgramLogic(peri_id, logic) # set the new program
            # print(traci.trafficlight.getAllProgramLogics(peri_id))

    def set_full_program(self):

        signal_plans = self.peridata.plan2program()
        # self.peridata.print_program()
        for signal_id, signal_plan in signal_plans.items():
            phases = []
            for state, duration in signal_plan.items():
                phase = traci.trafficlight.Phase(duration, state)
                phases.append(phase)

            tsc = self.info[signal_id]['tsc']
            logic = tsc.logic
            # if logic.programID == '0':
            #     logic.programID = '1'
            logic.phases = tuple(phases)
            traci.trafficlight.setProgramLogic(signal_id, logic)
            # traci.trafficlight.setProgram(signal_id, logic.programID)
            traci.trafficlight.setPhase(signal_id, 0)

        # for peri_id, peri in dict.items(self.info):
        #     peri_edge = peri['edge']
        #     peri_approaches = peri['in_edges']
        #     gated_edge_idx = peri_approaches.index(peri_edge)
        #
        #     tsc = peri['tsc']
        #     logic = tsc.logic
        #     signal_plan = signal_plan_info[peri_id]
        #     if signal_plan == -1:
        #         print(f'Node {peri_id} failed to get optimal signal plan. ')
        #         traci.trafficlight.setProgramLogic(peri_id, logic)  # use current program
        #     else:
        #         signal_program = plan2program(cycle=config['cycle_time'], green_start=signal_plan[0],
        #                                       green_duration=signal_plan[1], clearance_time=config['yellow_duration'],
        #                                       gated_idx=gated_edge_idx)
        #         # set signal program
        #         phases = []
        #         for state, duration in signal_program.items():
        #             phase = traci.trafficlight.Phase(duration, state)
        #             phases.append(phase)
        #         logic.phases = tuple(phases)
        #         traci.trafficlight.setProgramLogic(peri_id, logic) # set the new program



    def switch_phase(self, steps):
        """ check signal to switch
        """
        for tlsID in self.phase_swithtime.keys():
            if steps in self.phase_swithtime[tlsID]:
                next_phase = (np.where(self.phase_swithtime[tlsID] == steps)[0]
                              + 1) % (np.shape(self.phase_swithtime[tlsID])[0])
                traci.trafficlight.setPhase(tlsID, next_phase)

            # logic, = traci.trafficlight.getAllProgramLogics(tlsID) # get the current program
            # # print(logic)

            # # modify new program
            # phase_index = config['Peri_info'][tlsID][1]
            # logic.phases[phase_index].duration = green_time[tlsID] # the controlled phase
            # logic.phases[3-phase_index].duration = self.cycle_time - green_time[tlsID] - 2 * self.yellow_duration # the side phase

            # traci.trafficlight.setProgramLogic(tlsID, logic) # set the new program

            # # print(traci.trafficlight.getAllProgramLogics(tlsID))
        # return

    def get_down_edge_veh(self):
        down_edge_veh = []
        for tlsID in self.info.keys():
            downedge = self.info[tlsID]['down_edge']
            # downedge_veh.append(edge_infos[downedge]['total'])
            down_edge_veh.append(traci.edge.getLastStepHaltingNumber(str(downedge)))
        # print(downedge_veh) 

        # print(f'waiting vehicles on the down edge: {down_edge_veh}')
        return down_edge_veh

    def get_down_edge_occupancy(self):
        ''' obtain the down edge occupancy of the perimeters
        '''
        down_edge_occupancy = []

        for tlsID in self.info.keys():
            downedge = self.info[tlsID]['down_edge']
            down_edge_occupancy.append(traci.edge.getLastStepOccupancy(str(downedge)))
        
        self.down_edge_occupancy = down_edge_occupancy

        # return down_edge_occupancy

    def get_buffer_average_occupancy(self):
        ''' obtain the average occupancy of the buffer links
        '''
        self.buffer_average_occupancy = [0] * self.peri_num 

        for idx, tlsID in enumerate(self.info.keys()):
            buffer_occupancy = []
            buffer_edges = self.info[tlsID]['buffer_edges']
            
            for edge in buffer_edges:
                buffer_occupancy.append(traci.edge.getLastStepOccupancy(str(edge)))
        
            self.buffer_average_occupancy[idx] = np.mean(buffer_occupancy)

    def get_buffer_edge_veh(self):
        ''' obtain the waiting vehicles of the buffer links to calculate penalty
        '''
        buffer_edge_wait_veh = []

        for tlsID in sorted(list(self.info.keys())):
            buffer_edges = self.info[tlsID]['buffer_edges']
            for edge in buffer_edges:
            # downedge_veh.append(edge_infos[downedge]['total'])
                buffer_edge_wait_veh.append(traci.edge.getLastStepHaltingNumber(str(edge)))
        
        ## total wait vehs on the buffer
        buffer_wait_veh_tot = sum(buffer_edge_wait_veh)

        ## calculate the difference between t and t+1
        delta_buffer_wait_veh_tot = buffer_wait_veh_tot- self.buffer_wait_vehs_tot[-1]  # calculate delta: increment of queue in the buffer

        ## record the queue of current time
        self.buffer_wait_vehs_tot = np.append(self.buffer_wait_vehs_tot, buffer_wait_veh_tot) 
        
        # print(f'waiting vehicles on the buffer edge: {buffer_wait_veh_tot}')
        # print(f'increment waiting vehicles on the buffer edge: {delta_buffer_wait_veh_tot}')

        return delta_buffer_wait_veh_tot, buffer_wait_veh_tot