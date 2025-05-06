from matplotlib import pyplot as plt
import numpy as np
import random
import timeit
from time import time
from utils.result_processing import plot_MFD, plot_accu, plot_actions, plot_flow_MFD, plot_flow_progression, \
    plot_phase_mean_time, plot_tsc_delay, plot_peri_waiting, plot_controlled_tls_delay_epis, \
    plot_peri_queue_progression, plot_feature_progression, save_stats, plot_ordered_real_action, \
    plot_link_status, plot_trip_MFD
import traci


# from train_main import Agent_upper


class Trainer():
    ''' A class to manage the training 
    '''

    def __init__(self, env, agent_lower, agent_upper, sumo_cmd_e, e, n_jobs, config):
        self.env = env
        self.agent_upper = agent_upper
        self.agent_lower = agent_lower
        self.cur_epis = e
        self.sumo_cmd = sumo_cmd_e
        self.n_jobs = n_jobs
        self.config = config
        self.model_dir = self.config['models_path_name'] if config['mode'] == 'train' else config['cache_path_name']
        self.info_interval = self.config['infostep']
        self.cycle_time = self.config['cycle_time']
        self.lower_mode = config['lower_mode']
        self.lower_extend_flag = False
        self.mode = config['mode']
        self.record_fields = {}     # used for plot and save

    def run(self):
        ''' run the training for one epsiode
        '''
        # print(self.agent_upper.accu_crit)

        ## set random seeds
        self.set_seeds()

        ## clear buffer, set output files
        if self.n_jobs > 0:
            self.clear_buffer()
            self.set_output_file()
        else:  ## This is to change the single process 0 to 1
            self.n_jobs = 1

        self.set_action_type()

        ## Set epsilon
        self.set_epsilon()
        print(
            f'############# Episode {self.cur_epis + 1}: {self.agent_upper.action_type}, epsilon = {self.agent_upper.epsilon}, \
         {self.agent_lower.action_type}, epsilon = {self.agent_lower.epsilon} ##############')

        ## Load weights of NN
        self.load_weights(self.cur_epis, self.n_jobs)

        ## run the simulation to get memories
        self.explore()

        ## 4. after simulation
        edge_data, lane_data, queue_data, trip_data = self.process_output()
        upper_metric, lower_metric = self.get_metric(edge_data, lane_data, queue_data, trip_data)

        ## 4.1 get upper reward
        upper_metric['upper_reward_epis'] = self.get_upper_reward(upper_metric['flow'],
                                                                  upper_metric['peri_entered_vehs'])
        lower_metric['lower_reward_epis'] = self.get_lower_reward(lower_metric['tsc_metrics'])

        ## 4.3 fill the upper reward and metrics
        upper_metric['upper_penalty_epis'] = self.fill_upper_buffer_reward(upper_metric['upper_reward_epis'],
                                                                           upper_metric['PN_waiting'])

        upper_metric['cul_reward'] = [sum(upper_metric['upper_reward_epis'])]
        upper_metric['cul_penalty'] = [sum(upper_metric['upper_penalty_epis'])]
        upper_metric['cul_obj'] = [sum(upper_metric['cul_reward'] + upper_metric['cul_penalty'])]

        ## 4.4 record
        self.record(upper_metric, lower_metric)
        self.agent_upper.record(upper_metric)
        self.agent_lower.record(lower_metric)
        self.print_result(upper_metric, lower_metric)

        ## 4.5 plot
        self.plot()

        ## 4.6 save
        self.save()

        ## 4.7 process output
        if self.mode == 'train':
            ## test and train
            if self.cur_epis % self.n_jobs == 0:
                ## if it's not the testing /single process
                upper_metric['test'] = True
            else:
                upper_metric['test'] = False
        else:
            upper_metric['test'] = True

        ## expert and explore
        upper_metric['expert'] = True \
            if self.agent_upper.action_type == 'Expert' else False

        return self.agent_upper.buffer.buffer, self.agent_lower.memory, \
               upper_metric, lower_metric

    def explore(self):
        ''' explore within the episode
        '''

        step, done = self.reset()

        ## 3.simulate the episode
        while not done:
            ## output bar for multiple process
            print('\t' * (self.cur_epis % self.n_jobs) * 3 + f"{self.cur_epis + 1} current step: {step}", end='\r')

            ## 3.2 get action
            self.get_actions(step)      # 获取上层和下层控制方案

            ## 3.3 simulation of one control interval
            step, done, _, _ = self.env.simu_run(step)      # 运行一个下层interval

            ## 3.4 upper metric update
            if step % self.info_interval == 0:
                self.agent_upper.get_metric_each_interval()

            ## 3.5 memory
            done = self.memorize(step, done)        # 检查异常事件并保存状态
            # plot_tsc_delay(self.agent_lower.tsc)

            if done == True:
                traci.close()

    ### helper funcs
    def set_seeds(self):
        # mod_num = random.randint(0, 9999)
        # random.seed(int(time() * 1e10 % mod_num))
        # np.random.seed(int(time() * 1e10 % mod_num))
        random.seed(47 + 86 * self.cur_epis)
        np.random.seed(47 + 86 * self.cur_epis)

    def clear_buffer(self):
        ''' clear buffer of upper/lower agent
        '''
        ## clear buffer, only save new buffers
        self.agent_upper.buffer.clear()
        self.agent_lower.init_memory()

    def set_output_file(self):
        ''' set output file direction with paraller computing
        '''
        ## redefine upper-level edge output file
        agent_upper_outputfile = self.agent_upper.outputfile.split('/')
        agent_upper_outputfile[1] += str(self.cur_epis % self.n_jobs + 1)
        self.agent_upper.outputfile = '/'.join(agent_upper_outputfile)

        ## redefine lower-level edge output file
        agent_lower_outputfile = self.agent_lower.outputfile.split('/')
        agent_lower_outputfile[1] += str(self.cur_epis % self.n_jobs + 1)
        self.agent_lower.outputfile = '/'.join(agent_lower_outputfile)
        # print(self.agent_lower.outputfile)

    def set_action_type(self):
        self.agent_upper.set_action_type(self.n_jobs, self.cur_epis)
        self.agent_lower.set_action_type(self.n_jobs, self.cur_epis)

    def set_epsilon(self):
        self.agent_upper.set_epsilon(self.n_jobs, self.cur_epis)
        self.agent_lower.set_epsilon(self.n_jobs, self.cur_epis)

    def load_weights(self, e, njobs):
        if e // njobs == 0:
            path = self.config['cache_path_name']
        else:
            path = self.model_dir

        if self.agent_upper.peri_mode in ['DQN', 'DDPG', 'C_DQN']:
            self.agent_upper.load_weights(path)

        if self.agent_lower.lower_mode in ['OAM']:
            self.agent_lower.load_weights(path)

    def reset(self):
        ''' reset to start the simulation
        '''

        ######## must generate demand before traci.start ########
        ## 1. reset envir
        self.env.reset(self.sumo_cmd, self.config)

        ## 2. reset upper agent
        self.agent_upper.reset()

        ## 3. reset lower agent
        self.agent_lower.reset()

        ## lower agent fixtime plan
        if self.lower_mode == 'FixTime':
            self.agent_lower.set_program()
        ## Set the action type

        ## 4. 
        self.env.agent_lower = self.agent_lower

        ## training
        step, done = 0, False
        return step, done

    def get_actions(self, step):
        ''' get actions of upper and lower
        '''
        ## upper
        if step % self.cycle_time == 0:
            print(f'-------- Current time step: {step} --------')
            new_extend_flag = self.agent_upper.implem_action_all(step)
        else:
            new_extend_flag = self.lower_extend_flag

        ## lower
        if new_extend_flag == self.lower_extend_flag:
            extend_index = 0
        elif new_extend_flag and not self.lower_extend_flag:
            extend_index = 1
        else:
            extend_index = -1
        self.lower_extend_flag = new_extend_flag
        self.agent_lower.implem_action_all(extend_index)

    def memorize(self, step, done):
        ''' save the memory for upper and lower levels after each step
        '''
        ## upper level
        if step % self.cycle_time == 0:

            # check incidents
            done = self.check_gridlock(done, step)
            self.agent_upper.perimeter.check_green_waste()

            upper_old_state = self.agent_upper.get_memory(done)
            # record the PN accumulation at the lower agent for N-MP
            if self.agent_upper.peri_mode in ['Expert', 'PI', 'N-MP']:
                PN_accu = upper_old_state[0] * self.config['accu_max']
                print(f'PN vehicle accumulation after this interval: {PN_accu}. ')
                if self.agent_upper.peri_mode == 'N-MP':
                    self.agent_lower.set_tsc_state(PN_accu)

        ## lower level
        # if self.lower_mode != 'FixTime':
        self.agent_lower.store_memory()
        return done

    def check_gridlock(self, done, step):
        ''' check gridlock
            Upper: DQN and DDPG need gridlock break done
        '''
        ## break the episode if grid-lock
        # if self.lower_mode == 'OAM':
        if self.agent_upper.peri_mode in ['DQN', 'DDPG']:
            _, PN_halt_vehs = self.agent_upper.metric.get_halting_vehs(info_inter_flag=True)
            PN_halt_vehs = PN_halt_vehs / self.config['PN_halt_vehs_max']
            if not done and PN_halt_vehs > 0.7:
                done = True  # terminate the episode
                print(f'########## Grid lock of episode {self.cur_epis} at {step} ##########')
                # ## end traci
                # traci.close()

        return done

    def process_output(self):
        '''  process edge_data and lane_data after each epis
        '''
        ## output for entered vehicles
        # self.agent_upper.metric.process_output(self.agent_upper.outputfile)

        ## check the lower lane data
        self.agent_upper.metric.process_lane_xml(self.agent_lower.outputfile)

        ## output for csv
        self.agent_upper.metric.xml2csv(self.agent_upper.outputfile)  ## edge
        self.agent_upper.metric.xml2csv(self.agent_lower.outputfile)  ## lane
        self.agent_upper.metric.xml2csv(self.agent_lower.queuefile)   ## queue
        self.agent_upper.metric.xml2csv(self.agent_lower.tripfile)    ## vehicle trip

        ## metric data
        edge_data = self.agent_upper.metric.process_edge_output(self.agent_upper.outputfile)
        lane_data = self.agent_upper.metric.process_lane_output(self.agent_lower.outputfile)
        queue_data = self.agent_upper.metric.process_queue_output(self.agent_lower.queuefile)
        trip_data = self.agent_upper.metric.process_trip_output(self.agent_lower.tripfile)

        return edge_data, lane_data, queue_data, trip_data

    def get_metric(self, edge_data, lane_data, queue_data, trip_data):
        ''' get metric of upper / lower level after each epis
        '''
        ## upper
        upper_metric = self.agent_upper.get_metric_each_epis(edge_data)

        ## lower
        lower_metric = self.agent_lower.get_metric_each_epis(lane_data, queue_data, trip_data)

        return upper_metric, lower_metric

    def get_upper_reward(self, flow_epis, entered_vehs):
        ''' get upper reward of each step after each epis
        '''

        ## get reward
        upper_reward_epis = flow_epis / self.config['reward_max']
        for i in range(len(upper_reward_epis)):
            if upper_reward_epis[i] > 1:
                upper_reward_epis[i] = upper_reward_epis[i] ** 2

        entered_vehs = entered_vehs / 600
        upper_reward_epis += entered_vehs
        return upper_reward_epis

    def get_lower_reward(self, tsc_metrics):
        ''' get lower reward of each agent after each epis
        '''
        if self.agent_lower.lower_mode == 'OAM':
            reward = self.agent_lower.get_reward(tsc_metrics)

        else:
            return None

        return reward

    def print_result(self, upper_metric, lower_metric):
        ''' print result after simulation
        '''
        print(
            f"\r\n### Episode {self.cur_epis + 1} Finish --- total obj upper: {upper_metric['cul_obj'][0]} = {upper_metric['cul_reward'][0]}+{upper_metric['cul_penalty'][0]}; #############\r\n network delay mean: {np.around(lower_metric['network_delay_mean'], 2)} sec; tsc delay mean:{np.around(lower_metric['tsc_delay_mean'], 2)} sec \r\n network perveh delay mean: {np.around(lower_metric['network_perveh_delay_mean'], 4)} sec; tsc perveh delay mean:{np.around(lower_metric['tsc_perveh_delay_mean'], 4)} sec \r\n tsc through mean: {np.around(lower_metric['tsc_through_mean'], 0)} veh/hr*tsc \r\n")
        print(f"There are {lower_metric['PN_accu'][-1]} vehicles in the PN at the end of simulation. ")

    def record(self, upper_metric: dict, lower_metric: dict):
        '''
        record necessary data fields for plot and save
        '''
        # upper level data (interval: cycle length)
        self.record_fields['cumul_obj_upper'] = upper_metric['cul_reward'][0]
        self.record_fields['peri_entered_vehs'] = upper_metric['peri_entered_vehs']
        self.record_fields['peri_outflow_vehs'] = upper_metric['peri_outflow_vehs']
        for direction in ['north', 'east', 'west', 'south']:
            inflow_field_name = 'peri_entered_vehs_' + direction
            outflow_field_name = 'peri_outflow_vehs_' + direction
            self.record_fields[inflow_field_name] = upper_metric[inflow_field_name]
            self.record_fields[outflow_field_name] = upper_metric[outflow_field_name]
        self.record_fields['peri_waiting_tot'] = upper_metric['peri_waiting_tot']
        self.record_fields['peri_waiting_mean'] = upper_metric['peri_waiting_mean']
        self.record_fields['peri_ordered_inflow'] = upper_metric['ordered_inflow']
        self.record_fields['peri_cumul_green'] = upper_metric['cumul_green']
        self.record_fields['peri_cumul_throughput'] = upper_metric['cumul_throughput']
        # lower level data (interval: config['lower_agent_step'])
        self.record_fields['cumul_reward_lower'] = lower_metric['tsc_perveh_delay_mean']
        self.record_fields['PN_accu_epis'] = lower_metric['PN_accu']
        self.record_fields['PN_flow_epis'] = lower_metric['PN_flow']
        self.record_fields['PN_speed_epis'] = lower_metric['PN_speed']
        self.record_fields['peri_spillover'] = lower_metric['peri_spillover']
        self.record_fields['peri_queue'] = lower_metric['peri_queue']
        for direction in ['north', 'east', 'west', 'south']:
            queue_field_name = 'peri_queue_' + direction
            spillover_field_name = 'peri_spillover_' + direction
            self.record_fields[queue_field_name] = lower_metric[queue_field_name]
            self.record_fields[spillover_field_name] = lower_metric[spillover_field_name]
        self.record_fields['peri_throughput'] = lower_metric['peri_throughput']
        for node_type in self.config['peri_node_types']:
            demand_type = node_type.split('_')[-1]
            field_name = 'peri_throughput_' + demand_type
            self.record_fields[field_name] = lower_metric[field_name]
        self.record_fields['peri_throughput_each_tsc'] = lower_metric['peri_throughput_each_tsc']
        self.record_fields['peri_delay'] = lower_metric['peri_delay']
        self.record_fields['tsc_delay_step'] = lower_metric['tsc_delay_step']
        self.record_fields['tsc_perveh_delay_step'] = lower_metric['tsc_perveh_delay_step']
        self.record_fields['tsc_metrics'] = lower_metric['tsc_metrics']
        self.record_fields['trip_avg_travel_time'] = lower_metric['avg_travel_time']
        self.record_fields['trip_completion'] = lower_metric['vehicle_completion']
        self.record_fields['PN_link_density'] = lower_metric['PN_link_density']

    def plot(self):
        ''' plot after one episode
        02.10 update: the following performance indices are plotted and saved
        1) perimeter level: queue length on each gated link & summation of all queue length
        2) PN level: average speed of the PN / MFD of the PN
        3) overall level: cumulative trip completion (of each type of demand)
        '''

        ''' perimeter level: queue length '''
        max_lane_length = max([lanegroup.length for lanegroup in self.agent_lower.peridata.peri_lane_groups.values()])
        # queue on each gated movement
        plot_peri_queue_progression(self.config, self.record_fields['peri_queue'], max_lane_length, self.cur_epis, self.n_jobs)
        # total queue length on all the gated links
        peri_total_queue_evolve = [sum(queue) for queue in zip(*self.record_fields['peri_queue'].values())]
        plot_feature_progression(self.config, peri_total_queue_evolve, self.cur_epis, self.n_jobs,
                                 'peri_total_queue', 'queue length (m)')

        # other indices: throughput, spillover, and delay
        plot_feature_progression(self.config, self.record_fields['peri_throughput'], self.cur_epis, self.n_jobs,
                                 'peri_throughput', 'Throughput (veh)', accumulated=True)
        plot_feature_progression(self.config, self.record_fields['peri_spillover'], self.cur_epis, self.n_jobs,
                                 'peri_spillover', 'Spillover times', accumulated=True)
        plot_feature_progression(self.config, self.record_fields['peri_delay'], self.cur_epis, self.n_jobs,
                                 'peri_delay', 'Delay (s)')

        ## upper actions
        # if self.agent_upper.peri_mode not in ['Static', 'MaxPressure']:
        #     plot_actions(self.config, self.agent_upper.perimeter.green_time, \
        #                  peri_entered_vehs, self.cur_epis, \
        #                  self.agent_upper.action_type, self.n_jobs, self.agent_upper.perimeter.inflow_movements)

        ''' ordered inflow and actual inflow '''
        plot_ordered_real_action(self.config, self.record_fields['peri_ordered_inflow'],
                                 self.record_fields['peri_entered_vehs'], self.cur_epis, self.n_jobs)

        ''' PN level '''
        # MFD
        plot_MFD(self.config, self.record_fields['PN_accu_epis'], self.record_fields['PN_flow_epis'], self.cycle_time,
                 self.cur_epis, self.record_fields['cumul_obj_upper'],
                 self.n_jobs, self.record_fields['cumul_reward_lower'])

        # trip MFD
        # plot_trip_MFD(self.config, accu_epis, trip_completion, self.cur_epis, self.n_jobs)

        # average speed in PN
        plot_feature_progression(self.config, self.record_fields['PN_speed_epis'], self.cur_epis, self.n_jobs,
                                 feature_title='PN_average_speed', ylabel_name='Average speed (m/s)')

        # link density
        plot_link_status(self.config, self.agent_upper.netdata, self.record_fields['PN_link_density'],
                         self.cur_epis, self.n_jobs)

        # accumulation
        plot_feature_progression(self.config, self.record_fields['PN_accu_epis'], self.cur_epis, self.n_jobs,
                                 feature_title='PN_accu', ylabel_name='accumulation (veh)')

        # PN total delay (not used)
        # TODO: currently not PN but the whole network
        plot_feature_progression(self.config, self.record_fields['tsc_delay_step'], self.cur_epis, self.n_jobs,
                                 feature_title='delay', ylabel_name='delay (s)')
        plot_feature_progression(self.config, self.record_fields['tsc_perveh_delay_step'], self.cur_epis, self.n_jobs,
                                 feature_title='delay_per_vehicle', ylabel_name='delay (s)')
        # plot_controlled_tls_delay_epis(self.config, self.record_fields['tsc_delay_step'],
        #                                self.cur_epis, self.n_jobs, 2000, 'tsc_delay')
        # plot_controlled_tls_delay_epis(self.config, self.record_fields['tsc_perveh_delay_step'],
        #                                self.cur_epis, self.n_jobs, 1, 'tsc_perveh_delay')

        ''' overall level: cumulative trip completion '''
        # cumulative completion (used when not all demands are served)
        plot_feature_progression(self.config, self.record_fields['trip_completion'], self.cur_epis, self.n_jobs,
                                 'trip_completion', 'Cumulative completion trips')
        # average travel time progression (used when all demands are served)
        plot_feature_progression(self.config, self.record_fields['trip_avg_travel_time'], self.cur_epis, self.n_jobs,
                                 'travel_time', 'Time (s)')

        # total delay over the simulation process of each junction
        plot_tsc_delay(self.config, self.record_fields['tsc_metrics'], self.cur_epis, self.n_jobs)

    def save(self):

        # PN level
        save_stats(self.config, self.record_fields['PN_accu_epis'], self.cur_epis, self.n_jobs, 'PN_accu')
        save_stats(self.config, self.record_fields['PN_flow_epis'], self.cur_epis, self.n_jobs, 'PN_flow')
        save_stats(self.config, self.record_fields['PN_speed_epis'], self.cur_epis, self.n_jobs, 'PN_speed')
        # perimeter level
        save_stats(self.config, self.record_fields['peri_throughput'], self.cur_epis, self.n_jobs, 'peri_throughput')
        for node_type in self.config['peri_node_types']:
            demand_type = node_type.split('_')[-1]
            field_name = 'peri_throughput_' + demand_type
            save_stats(self.config, self.record_fields[field_name], self.cur_epis, self.n_jobs, field_name)
        save_stats(self.config, self.record_fields['peri_throughput_each_tsc'], self.cur_epis, self.n_jobs,
                   'peri_throughput_each_tsc')
        save_stats(self.config, self.record_fields['peri_queue'], self.cur_epis, self.n_jobs, 'peri_queue')
        save_stats(self.config, self.record_fields['peri_spillover'], self.cur_epis, self.n_jobs, 'peri_spillover')
        save_stats(self.config, self.record_fields['peri_entered_vehs'], self.cur_epis, self.n_jobs, 'peri_inflow')
        save_stats(self.config, self.record_fields['peri_outflow_vehs'], self.cur_epis, self.n_jobs, 'peri_outflow')
        for direction in ['south', 'north', 'east', 'west']:
            inflow_field_name = 'peri_entered_vehs_' + direction
            outflow_field_name = 'peri_outflow_vehs_' + direction
            queue_field_name = 'peri_queue_' + direction
            spillover_field_name = 'peri_spillover_' + direction
            save_stats(self.config, self.record_fields[inflow_field_name], self.cur_epis, self.n_jobs,
                       'peri_inflow_' + direction)
            save_stats(self.config, self.record_fields[outflow_field_name], self.cur_epis, self.n_jobs,
                       'peri_outflow_' + direction)
            save_stats(self.config, self.record_fields[queue_field_name], self.cur_epis, self.n_jobs, queue_field_name)
            save_stats(self.config, self.record_fields[spillover_field_name], self.cur_epis, self.n_jobs,
                       spillover_field_name)
        save_stats(self.config, self.record_fields['peri_delay'], self.cur_epis, self.n_jobs, 'peri_delay')
        for mov_type, green_stats in self.record_fields['peri_cumul_green'].items():
            save_stats(self.config, green_stats, self.cur_epis, self.n_jobs, '_'.join(('peri', mov_type, 'cumul_green')))
        for mov_type, throughput_stats in self.record_fields['peri_cumul_throughput'].items():
            save_stats(self.config, throughput_stats, self.cur_epis, self.n_jobs, '_'.join(('peri', mov_type, 'cumul_throughput')))
        # overall level
        for trip_type, trip_avg_time_data in self.record_fields['trip_avg_travel_time'].items():
            save_stats(self.config, trip_avg_time_data, self.cur_epis, self.n_jobs, 'avg_travel_time_' + trip_type)
        for trip_type, trip_completion in self.record_fields['trip_completion'].items():
            save_stats(self.config, trip_completion, self.cur_epis, self.n_jobs, 'trip_completion_' + trip_type)

    def fill_upper_buffer_reward(self, upper_reward_epis, PN_waiting_epis):

        upper_penalty_epis = -PN_waiting_epis / 1.5e5

        if self.agent_upper.buffer.buffer:
            for idx, (memory, reward) in \
                    enumerate(zip(self.agent_upper.buffer.buffer, upper_reward_epis)):

                memory[2] = reward

                # memory[-1] = penalty if memory[0][0] > 0.35 else 0

                ##penalty
                if self.config['network'] == 'Grid':
                    upper_penalty_epis = -PN_waiting_epis / 1.5e5

                    if memory[0][0] <= 0.3:
                        upper_penalty_epis[idx] = 0
                    # elif  0.25<=memory[0][0] and memory[0][0]<= 0.35:
                    #     if reward<0.8:
                    #         upper_penalty_epis[idx] += -1
                    #     else:
                    #         upper_penalty_epis[idx] = 0
                    else:
                        upper_penalty_epis[idx] += -1

                elif self.config['network'] == 'Bloomsbury':
                    upper_penalty_epis = -PN_waiting_epis / 1.5e5

                    if memory[0][0] <= 0.6:
                        upper_penalty_epis[idx] = 0
                    # elif  0.25<=memory[0][0] and memory[0][0]<= 0.35:
                    #     if reward<0.8:
                    #         upper_penalty_epis[idx] += -1
                    #     else:
                    #         upper_penalty_epis[idx] = 0
                    elif memory[0][0] >= 0.6:
                        upper_penalty_epis[idx] += -1

                memory[-1] = upper_penalty_epis[idx]
            print(min(upper_penalty_epis))

        return upper_penalty_epis
