import matplotlib.pyplot as plt
from utils.utilize import config
import pickle
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

################# PLOT #######################
def plot_MFD(config, accu, flow, aggregate_time, e, reward, n_jobs, reward_lower):
    ''' Plot MFD of each simulation
    '''
    # unit reform
    # throughput = np.array(throughput) / aggregate_time * 3600
    plt.xlabel('acc(veh)')
    plt.ylabel('outflow(veh/h)')
    plt.title(f'MFD (episode{e + 1})')
    plt.scatter(accu, flow)
    plt.xlim((0., config['accu_max']))
    plt.ylim(0., 1000)
    # plt.plot(x1, y1, label='整体路网基本图')
    # plt.plot(x2, y2, label='子路网基本图')
    # plt.legend()
    # plt.show()
    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_MFD_{np.around(reward, 2)}_{np.around(reward_lower, 2)}.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)

    else:
        file_name = f"e{e + 1}_MFD_{int(np.around(reward, 2))}_{np.around(reward_lower, 2)}.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        plt.savefig(plot_path)
    plt.close()


def plot_flow_MFD(config, accu, flow, e, n_jobs, traci=False):
    # accu = np.array(accu)
    # mean_speed = np.array(mean_speed)
    # TTD = accu * mean_speed* aggregate_time  # nveh*m/sec*sec = veh*m
    # flow = TTD / PN_road_length_tot / (aggregate_time/3600)
    # density = accu/(PN_road_length_tot/1000)

    plt.xlabel('density(veh/km)')
    plt.ylabel('flow (veh/h)')
    plt.title(f'Flow-density MFD')
    plt.scatter(accu, flow)
    plt.xlim((0., config['accu_max']))
    plt.ylim(0., 400)
    if traci:
        if e % n_jobs == 0:
            plt.savefig(
                f"{config['plots_path_name']}test\e{np.around((e) / n_jobs + 1, 0)}_flow_MFD_traci.png")

        else:
            plt.savefig(
                f"{config['plots_path_name']}e{e + 1}_flow_MFD_traci.png")
    else:
        if e % n_jobs == 0:
            plt.savefig(
                f"{config['plots_path_name']}test\e{np.around((e) / n_jobs + 1, 0)}_flow_MFD_output.png")

        else:
            plt.savefig(
                f"{config['plots_path_name']}e{e + 1}_flow_MFD_output.png")
    plt.close()
    plt.close('all')


def plot_demand(config, demand_config, demand_interval_sec, demand_interval_total):
    Demand_Interval = np.array(
        [i * demand_interval_sec for i in range(demand_interval_total)])
    # Demand_Interval = np.reshape(Demand_Interval, (1,-1))
    # plt.figure()
    plt.xlabel('Time(sec)')
    plt.ylabel('Demand(veh/h)')
    plt.title('Demand profile')

    for DemandType in demand_config:
        plt.plot(Demand_Interval,
                 DemandType['Demand'][0],
                 label=f"{DemandType['DemandType']}")

    plt.legend(loc='center left')
    # plt.show()
    plt.savefig(f"{config['plots_path_name']}Demand.png")
    plt.close()


def plot_reward(reward):
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('Reward')
    plt.plot(range(len(reward)), reward, 'o-')
    plt.savefig(f"{config['plots_path_name']}Reward.png")
    plt.close()


def plot_penalty(penalty):
    plt.xlabel('episode')
    plt.ylabel('penalty')
    plt.title('buffer_queue along training')
    plt.plot(range(len(penalty)), penalty, 'o-')
    plt.savefig(f"{config['plots_path_name']}penalty.png")
    plt.close()


def plot_obj_reward_penalty(obj, penalty, reward):
    # reward
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('reward')
    plt.plot(range(len(reward)), reward, 'o-')
    file_name = "upper_reward.png"
    plot_path = os.path.join(config['plots_path_name'], 'metric', file_name)
    plt.savefig(plot_path)
    # plt.savefig(f"{config['plots_path_name']}metric/reward.png")
    plt.close()

    # penalty
    plt.xlabel('episode')
    plt.ylabel('penalty')
    plt.title('penalty')
    plt.plot(range(len(penalty)), penalty, 'o-')
    file_name = "upper_penalty.png"
    plot_path = os.path.join(config['plots_path_name'], 'metric', file_name)
    plt.savefig(plot_path)
    plt.close()

    # objective
    plt.xlabel('episode')
    plt.ylabel('objective')
    plt.title('total objective')
    plt.plot(range(len(obj)), obj, 'o-')
    file_name = "upper_objective.png"
    plot_path = os.path.join(config['plots_path_name'], 'metric', file_name)
    plt.savefig(plot_path)
    plt.close()

    # together
    plt.xlabel('episode')
    plt.ylabel('objective')
    plt.title('total objective')
    plt.plot(range(len(obj)), obj, 'ko-', label=f"objective")
    plt.plot(range(len(reward)), reward, 'bo-', label=f"reward")
    plt.plot(range(len(penalty)), penalty, 'go-', label=f"penalty")
    plt.legend()
    file_name = "upper_objective_together.png"
    plot_path = os.path.join(config['plots_path_name'], 'metric', file_name)
    plt.savefig(plot_path)
    plt.close()


def plot_accu_critic(accu_list):
    # accu_critic
    plt.xlabel('episode')
    plt.ylabel('n_critical')
    plt.title('critical accumulation')
    plt.plot(range(len(accu_list)), accu_list, 'o-')

    file_name = "n_critic.png"
    plot_path = os.path.join(config['plots_path_name'], 'metric', file_name)
    plt.savefig(plot_path)

    # plt.savefig(f"{config['plots_path_name']}metric/n_critic.png")
    plt.close()


def plot_computime(computime):
    plt.xlabel('episode')
    plt.ylabel('computational time (secs)')
    plt.title('computational time of each episode')
    plt.plot(range(len(computime)), computime, 'o-')

    file_name = "Computational time.png"
    plot_path = os.path.join(config['plots_path_name'], 'metric', file_name)
    plt.savefig(plot_path)

    # plt.savefig(f"{config['plots_path_name']}metric\Computational time.png")
    plt.close()


def plot_accu(config, accu, e, n_jobs):
    plt.xlabel('cycle')
    plt.ylabel('accumulation (veh)')
    plt.title(f'PN vehicle accumulation progression of episode{e + 1}')
    plt.plot(range(len(accu)), accu, 'o-', label=f"accumulation")
    plt.legend()
    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_estimate_actual_inflow.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)
    else:
        file_name = f"e{e + 1}_estimate_actual_inflow.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        plt.savefig(plot_path)
        # plt.savefig(f"{plot_path}e{e+1}_tls_delay.png")
    pass


def plot_accu(config, accu, throughput, buffer_queue, e):
    ''' Plot progression of accumulation and throughput of each simulation
    '''
    plt.subplot(3, 1, 1)
    # plt.xlabel('cycle')
    plt.ylabel('accumulation (veh)')
    plt.title(f'progression of (episode{e + 1})')
    plt.plot(range(len(accu)), accu, 'o-', label=f"accumulation")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.xlabel('cycle')
    plt.ylabel('throughput (veh/cycle)')
    plt.plot(range(len(throughput)), throughput, 'g>-', label=f"throughput")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.xlabel('cycle')
    plt.ylabel('vehicle number (veh/cycle)')
    plt.plot(range(len(buffer_queue)), buffer_queue,
             'k>-', label=f"buffer queue")
    plt.legend()

    plt.savefig(f"{config['plots_path_name']}e{e + 1}_accu_throuput.png")
    plt.close()


def plot_critic_loss(critic_loss, level, mode):
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.title('Critic loss')
    plt.plot(range(len(critic_loss)), critic_loss)

    file_name = f"{level}_{mode}_CriticLoss.png"
    plot_path = os.path.join(config['plots_path_name'], 'critic', file_name)
    plt.savefig(plot_path)

    # plt.savefig(f"{config['plots_path_name']}critic\{level}_{mode}_CriticLoss.png")

    plt.close()


def plot_critic_loss_cur_epis(critic_loss, cur_epis, lr):
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.title('Critic loss')
    plt.plot(range(len(critic_loss)), critic_loss)

    file_name = f"CriticLoss_e{cur_epis}.png"
    plot_path = os.path.join(config['plots_path_name'], 'critic', file_name)
    plt.savefig(plot_path)

    plt.close()


def plot_last_critic_loss(last_critic_loss, level):
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.title('Last Critic loss')
    plt.plot(range(len(last_critic_loss)), last_critic_loss)

    file_name = f"LastCriticLoss_{level}.png"
    plot_path = os.path.join(config['plots_path_name'], 'critic', file_name)
    plt.savefig(plot_path)

    plt.close()


def plot_throughput(throughput):
    ''' Plot throughput in the training for each episode
    '''
    sum_throughput = [sum(i) for i in throughput]
    plt.xlabel('epoch')
    plt.ylabel('throughput (veh/hour)')
    plt.title('Throughput')
    plt.plot(range(len(sum_throughput)), sum_throughput, 'o-')
    plt.savefig(f"{config['plots_path_name']}metric/throughput.png")
    plt.close()


def plot_actions(config, actions, actions_excuted, e, action_type, n_jobs, category_names=None):
    # if peri_action_mode =='centralize' or action_type == 'Expert':
    #     ''' for centralized actions
    #     '''
    #     # plt.xlabel('')
    #     plt.ylabel('allowed vehicles')
    #     plt.title(f'action v.s. executed (episode{e+1})')
    #     plt.bar(range(len(actions)), actions)
    #     plt.bar(range(len(actions)), expert_action_list)
    #     plt.plot(range(len(actions_excuted)),
    #             actions_excuted,
    #             'ko-',
    #             label=f"excuted actions")
    #     plt.legend()
    #     plt.ylim((0., config['max_green'] * 4))
    #     # plt.show()
    #     plt.savefig(f"{config['plots_path_name']}e{e+1}_actions.png")
    #     plt.close()

    # else:
    ''' for decentralized actions
    '''
    actions = np.array(actions)
    actions_cum = actions.cumsum(axis=1)

    if category_names is None:
        category_names = list(config['Peri_info'].keys())

    if action_type == 'Expert':
        category_colors = plt.get_cmap('seismic')(np.linspace(0.15, 0.85, actions_cum.shape[1]))
    else:
        category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, actions_cum.shape[1]))

    fig, ax = plt.subplots()
    ax.yaxis.set_visible(True)
    ax.set_ylim(0, config['max_green'] * len(config['Peri_info']))

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        heights = actions[:, i]
        # 取第一列数值
        starts = actions_cum[:, i] - heights
        # 取每段的起始点
        ax.bar(range(len(actions)), heights, bottom=starts,
               label=colname, color=color)
        # xcenters = starts + heights / 2
        # r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # for y, (x, c) in enumerate(zip(xcenters, heights)):
        #     ax.text(y, x, str(int(c)), ha='center', va='center',
        #             color=text_color, rotation = 90)
        # fig.savefig(f"{config['plots_path_name']}e{e+1}_actions.png")
    plt.plot(range(len(actions_excuted)),
             actions_excuted,
             'ko-',
             label=f"excuted actions")
    ax.legend()
    if action_type == 'Expert':
        file_name = f"e{e + 1}_actions_expert.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        fig.savefig(plot_path)
    else:
        if e % n_jobs == 0:
            file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_actions.png"
            plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
            fig.savefig(plot_path)
        else:
            file_name = f"e{e + 1}_actions.png"
            plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
            fig.savefig(plot_path)
    plt.close()


def plot_ordered_real_action(config, estimated_inflow, actual_inflow, e, n_jobs, ordered_inflow=None):
    plt.xlabel('time')
    plt.ylabel('inflow (veh)')
    # plt.title('Estimated and actual inflow')
    plt.plot(np.arange(0, len(estimated_inflow) * 100, 100), estimated_inflow, '-', c='b', label='Estimated inflow')
    plt.plot(np.arange(0, len(actual_inflow) * 100, 100), actual_inflow, '-', c='g', label='Actual inflow')
    if ordered_inflow:
        plt.plot(np.arange(0, len(ordered_inflow) * 100, 100), ordered_inflow, '-', c='black', label='Ordered inflow')
    plt.ylim((0, 250))  # 统一scale
    plt.legend()
    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_estimate_actual_inflow.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)
    else:
        file_name = f"e{e + 1}_estimate_actual_inflow.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        plt.savefig(plot_path)
        # plt.savefig(f"{plot_path}e{e+1}_tls_delay.png")
    plt.close()


def plot_q_value(q_value):
    plt.xlabel('epoch')
    plt.ylabel('q_value')
    plt.title('Actor loss')
    plt.plot(range(len(q_value)), q_value)
    plt.savefig(f"{config['plots_path_name']}critic\ActorLoss.png")
    plt.close()


def plot_q_value_improve(q_value_improve):
    plt.xlabel('epoch')
    plt.ylabel('q_value_improve')
    plt.title('q_value improvement of each update')
    plt.plot(range(len(q_value_improve)), q_value_improve)

    file_name = "q_improve.png"
    plot_path = os.path.join(config['plots_path_name'], 'critic', file_name)
    plt.savefig(plot_path)
    plt.close()





def plot_tsc_delay(config, tsc_all, e, n_jobs):
    ''' plot delay for each junction
    '''
    plt.xlabel('tsc_id')
    plt.ylabel('Delay(1e5 sec)')
    plt.title('Total delay for each tsc')
    delay = {}
    for t_id, t_value in tsc_all.items():
        delay[t_id] = sum(t_value['delay_step']) / 1e5

    # delay=sorted(delay)
    sorted_tuples = sorted(delay.items(), key=lambda item: item[1])
    sort_delay = {k: v for k, v in sorted_tuples}
    plt.bar(sort_delay.keys(), sort_delay.values())
    plt.ylim((0., 10))

    # plt.legend()
    # plt.show()
    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_each_tsc_delay.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)
    else:
        file_name = f"e{e + 1}_each_tsc_delay.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        plt.savefig(plot_path)
    plt.close()


def plot_lower_reward_epis(reward_epis):
    ''' plot lower level reward of each epis
    '''
    # reward
    plt.xlabel('episode')
    plt.ylabel('lower level reward')
    plt.title('reward')
    plt.plot(range(len(reward_epis)), reward_epis, 'o-')

    file_name = "lower_reward.png"
    plot_path = os.path.join(config['plots_path_name'], 'metric', file_name)

    plt.savefig(plot_path)
    plt.close()


def plot_phase_mean_time(config, controled_light, tsc, e, n_jobs):
    ''' plot the mean time of each phase of each tsc
    '''
    phase_time = {}
    ## calculate mean phase time
    for tl_id in controled_light:
        phase_time[tl_id] = np.mean(tsc[tl_id].phase_time_list)

    # plt.bar(range(len(phase_time)), phase_time.values())
    plt.bar(phase_time.keys(), phase_time.values())

    plt.xlabel('tl_id')
    plt.ylabel("phase mean time (sec)")
    plt.title(f"phase mean time of episode {e}")
    # for a, b in phase_time.items():
    #     plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=11)

    if e % n_jobs == 0:
        plt.savefig(
            f"{config['plots_path_name']}test\e{np.around((e) / n_jobs + 1, 0)}_phase_mean_time.png")

    else:
        plt.savefig(
            f"{config['plots_path_name']}e{e + 1}_phase_mean_time.png")
    plt.close()


def plot_flow_progression(config, flow, e, n_jobs):
    plt.subplot(2, 1, 1)
    plt.ylabel('flow')
    plt.title(f'flow progression of (episode{e + 1})')
    plt.plot(range(len(flow)), flow, 'o-')
    plt.ylim((0., 350))

    # plt.legend()

    plt.subplot(2, 1, 2)
    plt.ylabel('total flow')
    # plt.title(f'penalty credit assignment of (episode{e+1})')
    plt.plot(range(len(flow)), np.cumsum(flow / 400), 'o-')
    plt.ylim((0., 60))

    # plt.legend()

    if e % n_jobs == 0:
        plt.savefig(
            f"{config['plots_path_name']}test\e{np.around((e) / n_jobs + 1, 0)}_flow_progression.png")

    else:
        plt.savefig(
            f"{config['plots_path_name']}e{e + 1}_flow_progression.png")
    plt.close()


def plot_peri_waiting(config, peri_waiting_tot, peri_waiting_mean, e, n_jobs):
    ''' Plot perimeter waiting time during the simulation
    '''

    '''total perimeter waiting '''
    sum_peri_waiting = np.sum(peri_waiting_tot, 1) / 3600
    plt.xlabel('time')
    plt.ylabel('Perimeter delay (hour)')
    plt.title('Total perimeter delay in each cycle')
    plt.plot(range(len(sum_peri_waiting)), sum_peri_waiting)
    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_periwait_tot.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)
    else:
        file_name = f"e{e + 1}_periwait_tot.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        plt.savefig(plot_path)
    plt.close()

    ''' mean perimeter waiting'''
    plt.xlabel('time')
    plt.ylabel('Delay (veh.sec)')
    plt.title('Average perimeter delay in each cycle')
    plt.plot(range(len(peri_waiting_mean)), peri_waiting_mean)
    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_periwait_mean.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)
    else:
        file_name = f"e{e + 1}_periwait_mean.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        plt.savefig(plot_path)
    plt.close()


def plot_controlled_tls_delay_epis(config, tls_delay_epis, e, n_jobs, ylim, title):
    ''' plot controlled tls delay progressions along the epis
    '''
    plt.xlabel('time')
    plt.ylabel('delay (sec)')
    plt.title(f'{title} progression')
    plt.plot(range(len(tls_delay_epis)), tls_delay_epis, '-')
    plt.ylim((0, ylim))
    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_{title}.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)
    else:
        file_name = f"e{e + 1}_{title}.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        plt.savefig(plot_path)
        # plt.savefig(f"{plot_path}e{e+1}_tls_delay.png")
    plt.close()


def plot_peri_feature_progression(config, feature_epis, e, n_jobs, feature_title, ylim):
    """
    plot the progression of the total throughput of all perimeter intersections
    """
    plt.xlabel('time')
    plt.ylabel(ylim)
    plt.title(f'perimeter {feature_title} progression')
    plt.plot(range(len(feature_epis)), feature_epis, '-')
    plt.ylim((0, max(feature_epis) * 1.2 + 10))
    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_peri_{feature_title}.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)
    else:
        file_name = f"e{e + 1}_peri_{feature_title}.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        plt.savefig(plot_path)
    plt.close()


def plot_peri_queue_progression(config, queue_epis: dict, e, n_jobs):
    """
    plot the progression of queue length on each perimeter inflows
    """
    plt.xlabel('time')
    plt.ylabel('queue length (m)')
    plt.title('perimeter queue progression')
    max_queue = 0
    for inflow_id, inflow_queue in queue_epis.items():
        plt.plot(range(len(inflow_queue)), inflow_queue, '-', label=inflow_id)
        max_queue = max(max_queue, max(inflow_queue))
    plt.ylim((0, max_queue * 1.2))
    plt.legend()
    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_peri_queue.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)
    else:
        file_name = f"e{e + 1}_peri_queue.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        plt.savefig(plot_path)
    plt.close()


################# save #######################
def save_data_train_upper(agent_upper, agent_lower):
    ''' save objective, reward, penalty, throughput along the training process of upper agents
    '''
    data_train = {}

    ## upper
    # data_train['obj_epis'] = agent_upper.cul_obj #obj_epis
    # data_train['reward_epis'] = agent_upper.cul_reward #reward_epis
    # data_train['penalty_epis'] = agent_upper.cul_penalty #

    # data_train['reward_epis_all'] = agent_upper.reward_epis_all #
    # data_train['penalty_epis_all'] = agent_upper.penalty_epis_all #

    # # data_train['throughput_epis'] = agent_upper.throughput_episode# throughput_epis
    # data_train['accu_epis'] = agent_upper.accu_episode# accu_epis
    # data_train['flow_epis'] = agent_upper.flow_episode#
    # data_train['speed_epis'] = agent_upper.speed_episode#
    # data_train['TTD_epis'] = agent_upper.TTD_episode#
    # data_train['PN_waiting_episode'] = agent_upper.PN_waiting_episode
    # data_train['entered_vehs_episode'] = agent_upper.entered_vehs_episode

    ''' upper-level  '''
    if agent_upper.accu_crit_list:
        agent_upper.record_epis['accu_crit_list'] = agent_upper.accu_crit_list  # throughput_epis
    if agent_upper.mfdpara_list:
        agent_upper.record_epis['mfdpara_list'] = agent_upper.mfdpara_list  # throughput_epis

    if agent_upper.peri_mode in ['DQN', 'DDPG', 'C_DQN']:
        agent_upper.record_epis['critic_loss'] = agent_upper.critic.qloss_list  # critic_loss
        agent_upper.record_epis['last_critic_loss'] = agent_upper.critic.last_qloss_list  # critic_loss

    ## save upper level
    data_train['record_epis_upper'] = agent_upper.record_epis
    data_train['best_epis_upper'] = agent_upper.best_epis

    ''' lower level '''
    if agent_lower.mode == 'OAM':
        agent_lower.record_epis['critic_loss'] = agent_lower.critic.qloss_list
        agent_lower.record_epis['last_critic_loss'] = agent_lower.critic.last_qloss_list

    ## save lower level
    data_train['record_epis_lower'] = agent_lower.record_epis

    ## save data
    save_dir = config['models_path_name'] + 'data_' + config[
        'mode'] + '_' + agent_upper.peri_mode + '_' + agent_lower.lower_mode + '.pkl'
    with open(save_dir, 'wb') as f:
        pickle.dump([data_train], f)

    print('###### Data save: Success ######')


def save_stats(config, stats, e, n_jobs, stats_name):
    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_{stats_name}.pkl"
    else:
        file_name = f"e{e + 1}_{stats_name}.pkl"
    stats_path = os.path.join(config['stats_path_name'], file_name)
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

# def save_tsc_config(config):
#
#     tsc_config = {'peri_distribution': config['peri_distribution_mode'],
#                   'peri_phase_mode': config['peri_signal_phase_mode']}

