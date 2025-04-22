import matplotlib.pyplot as plt
from utils.utilize import config
import pickle
import numpy as np
import os
import matplotlib
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from utils.networkdata import NetworkData
matplotlib.use('Agg')
import pandas as pd
from scipy.optimize import curve_fit

########## MFD Curve ##########
def mfd_func(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

################# PLOT WITHIN ONE EPISODE #######################
def plot_MFD(config, accu, flow, step, e, reward, n_jobs, reward_lower):
    ''' Plot MFD of each simulation
    '''
    plt.xlabel('acc(veh)')
    plt.ylabel('outflow(veh/h)')
    plt.title(f'MFD (episode{e + 1})')
    # give color to each point according to the index (equivalent to time step)
    color_group_size = 20       # size * 5 = time of each group
    group_num = (len(accu) + color_group_size - 1) // color_group_size
    color_values = np.arange(len(accu)) // color_group_size
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0, vmax=group_num - 1)
    # plot the scatter
    scatter = plt.scatter(accu, flow, c=color_values, cmap=cmap, norm=norm, edgecolor='gray')
    # cbar = plt.colorbar(scatter)
    # tick_locs = np.arange(group_num)
    # tick_labels = (tick_locs + 1) * color_group_size * config['lower_agent_step']
    # cbar.set_ticks(tick_locs)
    # cbar.set_ticklabels(tick_labels)
    # cbar.set_label('Time interval')

    # plot the fitting curve
    if config['demand_mode'] == 'MFD':
        mfd_x = np.linspace(0, max(accu), int(max(accu)))
        coeffs, var = curve_fit(mfd_func, accu, flow, p0=[1, -1, 1, 0], maxfev=10000)
        mfd_y = mfd_func(mfd_x, *coeffs)
        # mfd_coeffs = np.polyfit(accu, flow, 3)
        # mfd_func = np.poly1d(mfd_coeffs)
        # mfd_x = np.linspace(0, max(accu), int(max(accu)))
        # mfd_y = mfd_func(mfd_x)
        plt.plot(mfd_x, mfd_y, color='red', linewidth=2, label='MFD')
    plt.xlim(0., max(accu) * 1.1)
    plt.ylim(0., max(flow) * 1.2)
    plt.grid()
    # save the figure
    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_MFD_{np.around(reward, 2)}_{np.around(reward_lower, 2)}.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)
    else:
        file_name = f"e{e + 1}_MFD_{int(np.around(reward, 2))}_{np.around(reward_lower, 2)}.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        plt.savefig(plot_path)
    plt.close()

    # print the key parameters in the console
    if config['demand_mode'] == 'MFD':
        critical_accu = mfd_x[np.argmax(mfd_y)]
        print(f"The critical accumulation of the PN is {critical_accu} vehicles. ")
        reduced_max_flow = np.max(mfd_y) * config['max_flow_ratio']
        accu_idx = np.argmax(mfd_y)
        while True:
            if mfd_y[accu_idx] < reduced_max_flow:
                break
            accu_idx += 1
        upper_bound_accu = mfd_x[accu_idx]
        print(f"The upper bound accumulation of the PN is {upper_bound_accu} vehicles. ")


def plot_trip_MFD(config, accu: np.ndarray, trip: np.ndarray, e: int, n_jobs: int):
    '''
    plot the MFD with mean-density and trip completion rate
    '''
    # process the data
    density_epis: np.ndarray = accu / config['PN_total_length']
    agg_step_num = int(config['trip_complete_interval'] / config['lower_agent_step'])
    assert len(density_epis) % agg_step_num == 0
    average_density_epis = density_epis.reshape(-1, agg_step_num).mean(axis=1)      # veh/ln-km
    assert len(average_density_epis) == len(trip)
    trip_completion_epis = trip / config['trip_complete_interval'] * 3600       # veh/h
    # remove points with zero completion rate
    zero_mask = average_density_epis != 0
    average_density_epis = average_density_epis[zero_mask]
    trip_completion_epis = trip_completion_epis[zero_mask]
    # plot
    plt.xlabel('Density (veh/km)')
    plt.ylabel('Trip completion rate (veh/h)')
    plt.title(f'MFD (episode{e + 1})')
    plt.scatter(average_density_epis, trip_completion_epis)
    plt.xlim(0., max(average_density_epis * 1.2))
    plt.ylim(0., max(trip_completion_epis) * 1.2)
    plt.grid()
    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_MFD.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)
    else:
        file_name = f"e{e + 1}_MFD.png"
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


def plot_link_status(config, net_data: dict, link_density: pd.DataFrame, e, n_jobs):
    """
    plot the congestion status of each link
    """

    # generate the road map
    link_width = 25
    link_coords = []
    for edge_id, edge_data in net_data['edge'].items():
        edge_id = int(edge_id)
        if edge_id not in config['Edge_PN']:
            continue
        from_node, to_node = edge_data['outnode'], edge_data['incnode']
        from_x, from_y = net_data['node'][from_node]['x'], net_data['node'][from_node]['y']
        to_x, to_y = net_data['node'][to_node]['x'], net_data['node'][to_node]['y']
        # modify the coordinates to fit the plot
        if from_x == to_x:      # vertical link
            if from_y < to_y:       # from south to north
                from_y += link_width
                to_y -= link_width
                from_x += link_width / 2
                to_x += link_width / 2
            else:       # from north to south
                from_y -= link_width
                to_y += link_width
                from_x -= link_width / 2
                to_x -= link_width / 2
        else:    # horizontal link
            if from_x < to_x:       # from west to east
                from_x += link_width
                to_x -= link_width
                from_y -= link_width / 2
                to_y -= link_width / 2
            else:       # from east to west
                from_x -= link_width
                to_x += link_width
                from_y += link_width / 2
                to_y += link_width / 2
        link_coords.append([[from_x, from_y], [to_x, to_y]])
    links = []
    for link_coord in link_coords:
        links.append(np.array(link_coord))

    # plot the congestion status at each sampled time step (average the link density within the last cycle)
    sample_interval = 200
    for i in range(int(config['max_steps'] / sample_interval)):
        interval_begin = (i + 1) * sample_interval
        link_density_value = []
        for edge_id, edge_data in net_data['edge'].items():
            if int(edge_id) not in config['Edge_PN']:
                continue
            step, density_evolution = 5, []
            while step <= config['cycle_time']:
                density_evolution.append(link_density.loc[interval_begin - step, edge_id][0])
                step += 5
            link_density_value.append(float(np.mean(density_evolution)))
        # figure settings
        norm = mcolors.Normalize(vmin=0, vmax=300)
        colors = ['green', 'yellow', 'red']
        cmap = mcolors.LinearSegmentedColormap.from_list("green_yellow_red", colors, N=256)
        lc = LineCollection(links, cmap=cmap, norm=norm, linewidths=link_width / 8)
        lc.set_array(np.array(link_density_value))
        # plot the figure
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.set_xlim(-100, 2100)
        ax.set_ylim(-100, 2100)
        fig.colorbar(lc, ax=ax, label='Link density')
        # save the figure
        if e % n_jobs == 0:
            file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_link_density_time_{interval_begin}.png"
            plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
            fig.savefig(plot_path)
        else:
            file_name = f"e{e + 1}_link_density_time_{interval_begin}.png"
            plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
            fig.savefig(plot_path)
        plt.close(fig)


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
                 DemandType['Demand_mean'],
                 label=f"{DemandType['DemandType']}")

    plt.legend()
    # plt.show()
    plt.savefig(f"{config['plots_path_name']}Demand.png")
    plt.close()


def plot_demand_from_turn(config):
    Demand_Intervals = [sum(config['Demand_interval'][:i]) for i in range(len(config['Demand_interval']) + 1)]
    Demand_Intervals.append(config['max_steps'])
    Demand_Intervals = np.array(Demand_Intervals)

    for DemandType in config['DemandConfig']:
        Demand_Volume = [demand * DemandType['multiplier'] * len(DemandType['FromEdges']) for demand in DemandType['VolumeProfile']] + [0, 0]
        Demand_Volume = np.array(Demand_Volume)
        plt.step(Demand_Intervals, Demand_Volume, where='post', label=f"{DemandType['DemandType']}")

    plt.legend()
    plt.xlabel('Time(sec)')
    plt.ylabel('Demand(veh/h)')
    plt.title('Demand profile')

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
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_accu.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)
    else:
        file_name = f"e{e + 1}_accu.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        plt.savefig(plot_path)
        # plt.savefig(f"{plot_path}e{e+1}_tls_delay.png")
    plt.close()


# def plot_accu(config, accu, throughput, buffer_queue, e):
#     ''' Plot progression of accumulation and throughput of each simulation
#     '''
#     plt.subplot(3, 1, 1)
#     # plt.xlabel('cycle')
#     plt.ylabel('accumulation (veh)')
#     plt.title(f'progression of (episode{e + 1})')
#     plt.plot(range(len(accu)), accu, 'o-', label=f"accumulation")
#     plt.legend()
#
#     plt.subplot(3, 1, 2)
#     plt.xlabel('cycle')
#     plt.ylabel('throughput (veh/cycle)')
#     plt.plot(range(len(throughput)), throughput, 'g>-', label=f"throughput")
#     plt.legend()
#
#     plt.subplot(3, 1, 3)
#     plt.xlabel('cycle')
#     plt.ylabel('vehicle number (veh/cycle)')
#     plt.plot(range(len(buffer_queue)), buffer_queue,
#              'k>-', label=f"buffer queue")
#     plt.legend()
#
#     plt.savefig(f"{config['plots_path_name']}e{e + 1}_accu_throuput.png")
#     plt.close()


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


def plot_ordered_real_action(config, ordered_inflow, actual_inflow, e, n_jobs):
    plt.xlabel('time')
    plt.ylabel('inflow (veh)')
    # plt.title('Estimated and actual inflow')
    plt.plot(np.arange(0, len(ordered_inflow) * 100, 100), ordered_inflow, '-', c='b', label='Ordered inflow')
    plt.plot(np.arange(0, len(actual_inflow) * 100, 100), actual_inflow, '-', c='g', label='Actual inflow')
    plt.ylim((0, max(max(ordered_inflow), max(actual_inflow)) * 1.2))  # 统一scale
    plt.legend()
    plt.grid()
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


def plot_feature_progression(config, feature_epis, e, n_jobs, feature_title, ylabel_name, has_title=True):
    """
    plot the progression of a specific feature
    """
    plt.xlabel('time')
    plt.ylabel(ylabel_name)
    plt.grid()
    if has_title:
        plt.title(f'{feature_title} progression of episode {e + 1}')
    if isinstance(feature_epis, list) or isinstance(feature_epis, np.ndarray):      # 一条直线
        step = config['max_steps'] / len(feature_epis)
        plt.plot(np.arange(0, config['max_steps'], step), feature_epis, '-')
        plt.ylim((0, max(feature_epis) * 1.2))
    elif isinstance(feature_epis, dict):    # 同一指标的不同类型的直线，绘制于同一张图
        y_max = 0
        for feature_type, feature_data in feature_epis.items():
            step = config['max_steps'] / len(feature_data)
            plt.plot(np.arange(0, config['max_steps'], step), feature_data, '-', label=feature_type)
            y_max = max(y_max, max(feature_data))
        plt.legend()
        plt.ylim((0, y_max * 1.2 + 10))

    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_{feature_title}.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
        plt.savefig(plot_path)
    else:
        file_name = f"e{e + 1}_{feature_title}.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
        plt.savefig(plot_path)
    plt.close()


def plot_peri_queue_progression(config, queue_epis: dict, max_length, e, n_jobs):
    """
    plot the progression of queue length on each perimeter inflows in a heatmap
    """

    inflow_label = list(queue_epis.keys())
    queue_evolve = np.array(list(queue_epis.values()))

    pixel_width = 0.5
    fig_width = pixel_width * queue_evolve.shape[1]
    fig_height = pixel_width * queue_evolve.shape[0]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))    # type:figure.Figure, axes.Axes
    cax = ax.imshow(queue_evolve, cmap='coolwarm', aspect='equal', vmin=0, vmax=max_length*0.9)
    cbar = fig.colorbar(cax, ax=ax)     # 设置颜色条
    cbar.set_label('Queue length (m)')
    ax.set_xticks(np.arange(queue_evolve.shape[1]))
    ax.set_xticklabels(np.arange(1, queue_evolve.shape[1] + 1))
    ax.set_yticks(np.arange(queue_evolve.shape[0]))
    ax.set_yticklabels(inflow_label)
    ax.set_xlabel('Interval')
    ax.set_ylabel('Gated links')
    ax.set_aspect(aspect='equal', adjustable='box')
    plt.tight_layout()

    if e % n_jobs == 0:
        file_name = f"e{int(np.around((e) / n_jobs + 1, 0))}_peri_queue.png"
        plot_path = os.path.join(config['plots_path_name'], 'test', file_name)
    else:
        file_name = f"e{e + 1}_peri_queue.png"
        plot_path = os.path.join(config['plots_path_name'], 'explore', file_name)
    plt.savefig(plot_path)
    plt.close()


################# SAVE #######################
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


