import itertools
import math
from typing import List, Dict, Tuple, Union
import copy

import numpy as np

LANE_LENGTHS = {
    'E0': 250,
    'E1': 150,
    'E2': 96,
    'E3': 200,
    'E4': 4,
}

def analyze_traffic(state, lane_ids):
    """
    输入：当前所有车的state  &&   需要计算feature的lane id
    输出：
    1. lane_statistics: 每一个 lane 的统计信息 - Dict, key: lane_id, value:
        - vehicle_count: 当前车道的车辆数 1
        - lane_density: 当前车道的车辆密度 1
        - lane_length: 这个车道的长度 1
        - speeds: 在这个车道上车的速度 3 (mean, max, min)
        - waiting_times: 一旦车辆开始行驶，等待时间清零 3  (mean, max, min)
        - accumulated_waiting_times: 车的累积等待时间 3 (mean, max, min)

    2. ego_statistics: ego vehicle 的统计信息
        - speed: vehicle当前车速
        - position: vehicle当前位置
        - heading: vehicle当前朝向
        - road_id: vehicle当前所在道路的 ID
        - lane_index: vehicle当前所在车道的 index
        - surroundings: vehicle周围车辆的相对位置

    3. reward_statistics: Dict, key: vehicle_id, value:
        - 这个车行驶道路的 ID. eg: 'E0'
        - 累积行驶距离
        - 当前车速
        - position[0]: 1
        - position[1]: 1
        - waiting_time: 1

    """

    # 初始化每一个 lane 的统计信息
    lane_statistics = {
        lane_id: {
            'vehicle_count': 0,  # 当前车道的车辆数
            'lane_density': 0,  # 当前车道的车辆密度
            'speeds': [],  # 在这个车道上车的速度
            'waiting_times': [],  # 一旦车辆开始行驶，等待时间清零
            'accumulated_waiting_times': [],  # 车的累积等待时间
            # 'crash_assessment': 0,  # 这个车道碰撞了多少次
            'lane_length': 0,  # 这个车道的长度
            'lane_num_CAV': 0,  # 这个车道上的CAV数量
            'lane_CAV_penetration': 0,  # 这个车道上的CAV占比
        } for lane_id in lane_ids
    }
    # 初始化 ego vehicle 的统计信息
    ego_statistics = {}
    # 初始化 hdv vehicle 的统计信息
    hdv_statistics = {}
    # 初始化 reward_statistics, 用于记录每一辆车所在的 edge, 后面用于统计车辆的 travel time
    reward_statistics = {}

    # 先收录所有车的信息 - （HDV + CAV）
    for vehicle_id, vehicle in state.items():
        lane_id = vehicle['lane_id']  # 这个车所在车道的 ID. eg: 'E0_2'
        lane_index = vehicle['lane_index']
        if lane_id[:3] in [':J3']:
            lane_id = f'E4_{int(lane_index)}'
        elif lane_id[:3] in [':J1']:
            lane_id = f'E1_{int(lane_index)}'
        elif lane_id[:3] in [':J2']:
            lane_id = f'E2_{int(lane_index)}'

        road_id = vehicle['road_id']  # 这个车行驶道路的 ID. eg: 'E0'
        if road_id[:3] in [':J3']:
            road_id = 'E4'
        elif road_id[:3] in [':J1']:
            road_id = 'E1'
        elif road_id[:3] in [':J2']:
            road_id = 'E2'

        lane_index = vehicle['lane_index']  # 这个车所在车道的 index eg: 2
        speed = vehicle['speed']  # vehicle当前车速
        position = vehicle['position']  # vehicle当前位置
        heading = vehicle['heading']  # vehicle当前朝向

        leader_info = vehicle['leader']  # vehicle前车的信息
        distance = vehicle['distance']  # 这个车的累积行驶距离
        waiting_time = vehicle['waiting_time']  # vehicle的等待时间
        accumulated_waiting_time = vehicle['accumulated_waiting_time']  # vehicle的累积等待时间
        lane_position = vehicle['lane_position']  # vehicle 前保险杠到车道起点的距离

        # 感兴趣的lane数据统计
        if lane_id in lane_ids:
            # reward计算需要的信息
            # 记录每一个 vehicle 的 (edge id, distance, speed), 用于统计 travel time 从而计算平均速度
            reward_statistics[vehicle_id] = [road_id, distance, speed,
                                             position[0], position[1],
                                             waiting_time, accumulated_waiting_time]

        # ego车需要的信息
        if vehicle['vehicle_type'] == 'ego':
            # TODO: when will the state['leader'] be not None?
            surroundings = vehicle['surround']
            surroundings_expand = vehicle['surround_expand']

            ego_statistics[vehicle_id] = [position, speed,
                                          # speed, position,
                                          heading, road_id, lane_index,
                                          surroundings,
                                          surroundings_expand
                                          ]
        # HDV车需要的信息
        else:
            hdv_statistics[vehicle_id] = [speed, position,
                                          heading, road_id, lane_index]

        # 根据lane_id把以下信息写进lane_statistics
        if lane_id in lane_statistics:
            stats = lane_statistics[lane_id]
            stats['lane_length'] = LANE_LENGTHS[road_id],  # 这个车道的长度
            stats['vehicle_count'] += 1
            if vehicle['vehicle_type'] == 'ego':
                stats['lane_num_CAV'] += 1
            stats['speeds'].append(vehicle['speed'])
            stats['accumulated_waiting_times'].append(vehicle['accumulated_waiting_time'])
            stats['waiting_times'].append(vehicle['waiting_time'])
            stats['lane_density'] = stats['vehicle_count'] / LANE_LENGTHS[road_id]

    # lane_statistics计算
    for lane_id, stats in lane_statistics.items():
        speeds = np.array(stats['speeds'])
        waiting_times = np.array(stats['waiting_times'])
        accumulated_waiting_times = np.array(stats['accumulated_waiting_times'])
        vehicle_count = stats['vehicle_count']
        lane_length = stats['lane_length']
        lane_density = stats['lane_density']
        lane_num_CAV = stats['lane_num_CAV']
        lane_CAV_penetration = lane_num_CAV / vehicle_count if vehicle_count > 0 else 0

        if vehicle_count > 0:  # 当这个车道上有车的时候
            lane_statistics[lane_id] = [
                vehicle_count,
                lane_density,
                lane_length[0],
                np.mean(speeds), np.max(speeds), np.min(speeds),
                np.mean(waiting_times), np.max(waiting_times), np.min(waiting_times),
                np.mean(accumulated_waiting_times), np.max(accumulated_waiting_times), np.min(accumulated_waiting_times),
                lane_num_CAV, lane_CAV_penetration
            ]
        else:
            # lane_statistics[lane_id] = [0] * 12
            lane_statistics[lane_id] = [0] * 14  # add lane_num_CAV, lane_CAV_penetration
    # lane_statistics转换成dict
    lane_statistics = {lane_id: stats for lane_id, stats in lane_statistics.items()}

    return lane_statistics, ego_statistics, reward_statistics, hdv_statistics

def check_collisions_based_pos(vehicles, gap_threshold: float):
    """输出距离过小的车辆的 ID, 直接根据 pos 来进行计算是否碰撞 (比较简单)

    Args:
        vehicles: 包含车辆部分的位置信息
        gap_threshold (float): 最小的距离限制
    """
    collisions = []
    collisions_info = []

    _distance = {}  # 记录每辆车之间的距离
    for (id1, v1), (id2, v2) in itertools.combinations(vehicles.items(), 2):
        dist = math.sqrt(
            (v1['position'][0] - v2['position'][0]) ** 2 \
            + (v1['position'][1] - v2['position'][1]) ** 2
        )
        _distance[f'{id1}-{id2}'] = dist
        if dist < gap_threshold:
            collisions.append((id1, id2))
            collisions_info.append({'collision': True,
                                    'CAV_key': id1,
                                    'surround_key': id2,
                                    'distance': dist,
                                    })

    return collisions, collisions_info

def check_collisions(vehicles, ego_ids, gap_threshold: float, gap_warn_collision: float):
    ego_collision = []
    ego_warn = []

    info = []

    # 把ego的state专门拿出来
    def filter_vehicles(vehicles, ego_ids):
        # Using dictionary comprehension to create a new dictionary
        # by iterating over all key-value pairs in the original dictionary
        # and including only those whose keys are in ego_ids
        filtered_vehicles = {key: value for key, value in vehicles.items() if key in ego_ids}
        return filtered_vehicles

    filtered_ego_vehicles = filter_vehicles(vehicles, ego_ids)

    for ego_key, ego_value in filtered_ego_vehicles.items():
        for surround_direction, content in filtered_ego_vehicles[ego_key]['surround'].items():
            c_info = None
            w_info = None

            # print(ego_key, 'is surrounded by:', content[0], 'with direction', surround_direction,
            # 'at distance', content[1])
            # TODO: 同一个车道和不同车道的车辆的warn gap应该是不一样！！！！11
            distance = math.sqrt(content[1] ** 2 + content[2] ** 2)
            # print('distance:', distance)
            if distance < gap_threshold:
                # print(ego_key, 'collision with', content[0])
                ego_collision.append((ego_key, content[0]))
                c_info = {'collision': True,
                          'CAV_key': ego_key,
                          'surround_key': content[0],
                          'distance': distance,
                          'relative_speed': content[3],
                          }

            elif gap_threshold <= distance < gap_warn_collision:
                ego_warn.append((ego_key, content[0]))
                w_info = {'warn': True,
                          'CAV_key': ego_key,
                          'surround_key': content[0],
                          'distance': distance,
                          'relative_speed': content[3],
                          }
            if c_info:
                info.append(c_info)
            elif w_info:
                info.append(w_info)

    return ego_collision, ego_warn, info

def check_prefix(a: str, B: List[str]) -> bool:
    """检查 B 中元素是否有以 a 开头的

    Args:
        a (str): lane_id
        B (List[str]): bottle_neck ids

    Returns:
        bool: 返回 lane_id 是否是 bottleneck
    """
    return any(a.startswith(prefix) for prefix in B)

def count_bottleneck_vehicles(lane_statistics, bottle_necks) -> int:
    """
    统计 bottleneck 处车辆的个数
    """
    veh_num = 0
    for lane_id, lane_info in lane_statistics.items():
        if check_prefix(lane_id, bottle_necks):
            veh_num += lane_info[0]  # lane_statistics的第一个是vehicle_count
    return veh_num

def calculate_congestion(vehicles: int, length: float, num_lane: int, ratio: float = 1) -> float:
    """计算 bottle neck 的占有率, 我们假设一辆车算上车间距是 10m, 那么一段路的。占有率是
        占有率 = 车辆数/(车道长度*车道数/10)
    于是可以根据占有率计算拥堵系数为:
        拥堵程度 = min(占有率, 1)

    Args:
        vehicles (int): 在 bottle neck 处车辆的数量
        length (float): bottle neck 的长度, 单位是 m
        num_lane (int): bottle neck 的车道数

    Returns:
        float: 拥堵系数 in (0,1)
    """
    capacity_used = ratio * vehicles / (length * num_lane / 10)  # 占有率
    congestion_level = min(capacity_used, 1)  # Ensuring congestion level does not exceed 100%
    return congestion_level

def calculate_speed(congestion_level: float, speed: int) -> float:
    """根据拥堵程度来计算车辆的速度

    Args:
        congestion_level (float): 拥堵的程度, 通过 calculate_congestion 计算得到
        speed (int): 车辆当前的速度

    Returns:
        float: 车辆新的速度
    """
    if congestion_level > 0.2:
        speed = speed * (1 - congestion_level)
        speed = max(speed, 1)
    else:
        speed = -1  # 不控制速度
    return speed

def one_hot_encode(value, unique_values):
    """Create an array with zeros and set the corresponding index to 1
    """
    one_hot = np.zeros(len(unique_values))
    index = unique_values.index(value)
    one_hot[index] = 1
    return one_hot.tolist()

def euclidean_distance(point1, point2):
    # Convert points to numpy arrays
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate the Euclidean distance
    distance = np.linalg.norm(point1 - point2)
    return distance

def compute_ego_vehicle_features(
        ego_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        lane_statistics: Dict[str, List[float]],
        unique_edges: List[str],
        edge_lane_num: Dict[str, int],
        bottle_neck_positions: Tuple[float]
) -> Dict[str, List[float]]:
    """计算每一个 ego vehicle 的特征

    Args:
        ego_statistics (Dict[str, List[Union[float, str, Tuple[int]]]]): ego vehicle 的信息
        lane_statistics (Dict[str, List[float]]): 路网的信息
        unique_edges (List[str]): 所有考虑的 edge
        edge_lane_num (Dict[str, int]): 每一个 edge 对应的车道
        bottle_neck_positions (Tuple[float]): bottle neck 的坐标
    """
    feature_vectors = {}

    for ego_id, ego_info in ego_statistics.items():
        # ############################## ego_statistics 的信息 ##############################
        speed, position, heading, road_id, lane_index, surroundings = ego_info

        # 速度归一化  - 1
        normalized_speed = speed / 15.0

        # 位置归一化  - 2
        position_x, position_y = position
        normalized_position_x = position_x / 700
        normalized_position_y = position_y

        # 转向归一化 - 1
        normalized_heading = heading / 360

        # One-hot encode road_id - 5
        road_id_one_hot = one_hot_encode(road_id, unique_edges)

        # One-hot encode lane_index - 4
        lane_index_one_hot = one_hot_encode(lane_index, list(range(edge_lane_num.get(road_id, 0))))
        # 如果车道数不足4个, 补0 对齐到最多数量的lane num
        if len(lane_index_one_hot) < 4:
            lane_index_one_hot += [0] * (4 - len(lane_index_one_hot))

        # ############################## 周车信息 ##############################
        # 提取surrounding的信息 -18
        surround = []
        for index, (_, statistics) in enumerate(surroundings.items()):
            relat_x, relat_y, relat_v = statistics[1:4]
            surround.append([relat_x, relat_y, relat_v])  # relat_x, relat_y, relat_v
        flat_surround = [item for sublist in surround for item in sublist]
        # 如果周围车辆的信息不足6*3个, 补0 对齐到最多数量的周车信息
        if len(flat_surround) < 18:
            flat_surround += [0] * (18 - len(flat_surround))

        # ############################## 当前车道信息 ##############################
        # ego_id当前所在的lane
        ego_lane_id = f'{road_id}_{lane_index}'
        # 每个obs只需要一个lane的信息，不需要所有lane的信息， shared obs可以拿到所有lane的信息
        ego_lane_stats = lane_statistics[ego_lane_id]

        # - vehicle_count: 当前车道的车辆数 1
        # - lane_density: 当前车道的车辆密度 1
        # - lane_length: 这个车道的长度 1
        # - speeds: 在这个车道上车的速度 1 (mean)
        # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
        # - accumulated_waiting_times: 车的累积等待时间 1 (mean, max)
        ego_lane_stats = ego_lane_stats[:4] + ego_lane_stats[6:7] + ego_lane_stats[9:10]

        # ############################## bottle_neck 的信息 ##############################
        # 车辆距离bottle_neck
        bottle_neck_position_x = bottle_neck_positions[0] / 700
        bottle_neck_position_y = bottle_neck_positions[1]

        distance = euclidean_distance(position, bottle_neck_positions)
        normalized_distance = distance / 700

        # ############################## 合并所有 ##############################
        # feature_vector = [normalized_speed, normalized_position_x, normalized_position_y, normalized_heading,
        #                   bottle_neck_position_x, bottle_neck_position_y, normalized_distance] + \
        #                   road_id_one_hot + lane_index_one_hot + flat_surround + all_lane_stats

        feature_vector = [normalized_speed, normalized_position_x, normalized_position_y, normalized_heading,
                          bottle_neck_position_x, bottle_neck_position_y, normalized_distance] + \
                         road_id_one_hot + lane_index_one_hot + flat_surround + ego_lane_stats

        # Assign the feature vector to the corresponding ego vehicle
        feature_vectors[ego_id] = [float(item) for item in feature_vector]

    # 保证每一个 ego vehicle 的特征长度一致
    assert all(len(feature_vector) == 40 for feature_vector in feature_vectors.values())

    # # Create a new dictionary to hold the updated feature lists
    # updated_feature_vectors = {}
    #
    # # Iterate over each key-value pair in the dictionary
    # for cav, features in feature_vectors.items():
    #     # Start with the original list of features
    #     new_features = features.copy()
    #     # Iterate over all other CAVs
    #     for other_cav, other_features in feature_vectors.items():
    #         if other_cav != cav:
    #             # Append the first four elements of the other CAV's features
    #             new_features.extend(other_features[:4])
    #     # Update the dictionary with the new list
    #     updated_feature_vectors[cav] = new_features
    #
    # for updated_feature_vector in updated_feature_vectors.values():
    #     if not len(updated_feature_vector) == 56:
    #         updated_feature_vector += [0] * (56 - len(updated_feature_vector))
    #
    # # assert all(len(updated_feature_vector) == 56 for updated_feature_vector in updated_feature_vectors.values())

    return feature_vectors

def get_target(
        ego_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        lane_statistics: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """计算每一个 ego vehicle 的临近目标点（x_target, y_target）

        Args:
            ego_statistics (Dict[str, List[Union[float, str, Tuple[int]]]]): ego vehicle 的信息
            lane_statistics (Dict[str, List[float]]): 路网的信息
    """
    target_points = {}
    for ego_id, ego_info in ego_statistics.items():
        if ego_info[0][0] < 400:
            target_points[ego_id] = [400/700, ego_info[0][1]/3.2]
        elif ego_info[0][0] < 496:
            target_points[ego_id] = [496/700, 1.6/3.2] if ego_info[0][1] > 0 else [496/700, -1.6/3.2]
        else:
            target_points[ego_id] = [1, 1.6/3.2] if ego_info[0][1] > 0 else [1, -1.6/3.2]
    return target_points

def compute_base_ego_vehicle_features(
        self,
        hdv_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        ego_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        lane_statistics: Dict[str, List[float]],
        unique_edges: List[str],
        edge_lane_num: Dict[str, int],
        bottle_neck_positions: Tuple[float],
        ego_ids: List[str]
) -> Dict[str, List[float]]:
    """计算每一个 ego vehicle 的特征

    Args:
        ego_statistics (Dict[str, List[Union[float, str, Tuple[int]]]]): ego vehicle 的信息
        lane_statistics (Dict[str, List[float]]): 路网的信息
        unique_edges (List[str]): 所有考虑的 edge
        edge_lane_num (Dict[str, int]): 每一个 edge 对应的车道
        bottle_neck_positions (Tuple[float]): bottleneck 的坐标
    output:
        surround_vehs_stats: 12 * 3-relative x,y,v
        ego_lane_stats: 6--vehicle count, lane density, lane length, speed-mean, waiting time-mean, lane CAV penetration
        left_lane_stats: 6--vehicle count, lane density, lane length, speed-mean, waiting time-mean, lane CAV penetration
        right_lane_stats: 6--vehicle count, lane density, lane length, speed-mean, waiting time-mean, lane CAV penetration
        bottleneck_position & distance: 1 * 2
        self_stats: 1 * 13
    """
    # ############################## 所有HDV的信息 ############################## 13
    hdv_stats = {}
    for hdv_id, hdv_info in hdv_statistics.items():
        speed, position, heading, road_id, lane_index = hdv_info
        # 速度归一化  - 1
        normalized_speed = speed / 15.0
        # 位置归一化  - 2
        position_x, position_y = position
        normalized_position_x = position_x / 700
        normalized_position_y = position_y
        # 转向归一化 - 1
        normalized_heading = heading / 360
        # One-hot encode road_id - 5
        road_id_one_hot = one_hot_encode(road_id, unique_edges)
        # One-hot encode lane_index - 4
        lane_index_one_hot = one_hot_encode(lane_index, list(range(edge_lane_num.get(road_id, 0))))
        # 如果车道数不足4个, 补0 对齐到最多数量的lane num
        if len(lane_index_one_hot) < 4:
            lane_index_one_hot += [0] * (4 - len(lane_index_one_hot))
        hdv_stats[hdv_id] = [normalized_speed, normalized_position_x, normalized_position_y,
                             normalized_heading] + road_id_one_hot + lane_index_one_hot
    # convert to 2D array (12 * 13)  - 12 is max number of HDVs
    hdv_stats = np.array(list(hdv_stats.values()))
    if 0 < hdv_stats.shape[0] <= 12:
        # add 0 to make sure the shape is (12, 13)
        hdv_stats = np.vstack([hdv_stats, np.zeros((12 - hdv_stats.shape[0], 13))])
    elif hdv_stats.shape[0] == 0:
        hdv_stats = np.zeros((12, 13))
    # ############################## 所有CAV的信息 ############################## 13
    cav_stats = {}
    ego_stats = {}
    global_cav = {}
    surround_vehs_stats = {key:[] for key in ego_ids}
    for ego_id, ego_info in ego_statistics.items():
        # ############################## 自己车的信息 ############################## 13
        position, speed, heading, road_id, lane_index, surroundings = ego_info
        # 速度归一化  - 1
        normalized_speed = speed / 15.0
        # 位置归一化  - 2
        position_x, position_y = position
        normalized_position_x = position_x / 700
        normalized_position_y = position_y
        # 转向归一化 - 1
        normalized_heading = heading / 360
        # One-hot encode road_id - 5
        road_id_one_hot = one_hot_encode(road_id, unique_edges)
        # One-hot encode lane_index - 4
        lane_index_one_hot = one_hot_encode(lane_index, list(range(edge_lane_num.get(road_id, 0))))
        # 如果车道数不足4个, 补0 对齐到最多数量的lane num
        if len(lane_index_one_hot) < 4:
            lane_index_one_hot += [0] * (4 - len(lane_index_one_hot))
        # ############################## 周车信息 ############################## 18
        # 提取surrounding的信息 -18
        surround = []

        for index, (_, statistics) in enumerate(surroundings.items()):
            relat_x, relat_y, relat_v = statistics[1:4]
            surround.append([relat_x, relat_y, relat_v])  # relat_x, relat_y, relat_v
            surround_vehs_stats[ego_id].append([relat_x/700, relat_y, relat_v/15])
        flat_surround = [item for sublist in surround for item in sublist]
        # 如果周围车辆的信息不足6*3个, 补0 对齐到最多数量的周车信息
        if len(flat_surround) < 18:
            flat_surround += [0] * (18 - len(flat_surround))

        # cav_stats[ego_id] = [normalized_speed, normalized_position_x, normalized_position_y,
        #                      normalized_heading] + road_id_one_hot + lane_index_one_hot
        # ego_stats[ego_id] = [normalized_speed, normalized_position_x, normalized_position_y,
        #                      normalized_heading] + road_id_one_hot + lane_index_one_hot
        cav_stats[ego_id] = [normalized_position_x, normalized_position_y, normalized_speed,
                             normalized_heading] + road_id_one_hot + lane_index_one_hot
        ego_stats[ego_id] = [normalized_position_x, normalized_position_y, normalized_speed,
                             normalized_heading] + road_id_one_hot + lane_index_one_hot
        global_cav[ego_id] = [normalized_position_x, normalized_position_y, normalized_speed]
    # convert to 2D array (5 * 5)
    cav_stats = np.array(list(cav_stats.values()))
    if 0 < cav_stats.shape[0] <= 12:
        # add 0 to make sure the shape is (12, 13)
        cav_stats = np.vstack([cav_stats, np.zeros((12 - cav_stats.shape[0], 13))])
    elif cav_stats.shape[0] == 0:
        cav_stats = np.zeros((12, 13))

    if len(ego_stats) != len(ego_ids):
        for ego_id in ego_ids:
            if ego_id not in ego_stats:
                ego_stats[ego_id] = [0.0] * 13
                global_cav[ego_id] = [0.0] * 3
    global_cav = np.array(list(global_cav.values()))
    if global_cav.shape[0] < self.max_num_CAVs:
        global_cav = np.vstack([global_cav, np.zeros((self.max_num_CAVs - global_cav.shape[0], 3))])

    # ############################## lane_statistics 的信息 ############################## 18
    # Initialize a list to hold all lane statistics
    all_lane_stats = {}
    all_lane_state_simple = {}

    # Iterate over all possible lanes to get their statistics
    for lane_id, lane_info in lane_statistics.items():
        # - vehicle_count: 当前车道的车辆数 1
        # - lane_density: 当前车道的车辆密度 1
        # - lane_length: 这个车道的长度 1
        # - speeds: 在这个车道上车的速度 1 (mean)
        # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
        # - accumulated_waiting_times: 车的累积等待时间 1 (mean)--delete
        # - lane_CAV_penetration: 这个车道上的CAV占比 1

        # all_lane_stats[lane_id] = lane_info[:4] + lane_info[6:7] + lane_info[9:10]
        lane_info[0] = lane_info[0] / 10
        lane_info[2] = lane_info[2] / 700
        lane_info[3] = lane_info[3] / 15
        all_lane_stats[lane_id] = lane_info[:4] + lane_info[6:7] + lane_info[13:14]
        all_lane_state_simple[lane_id] = lane_info[1:2] + lane_info[3:4] + lane_info[6:7]

    ego_lane_stats = {}
    left_lane_stats = {}
    right_lane_stats = {}
    for ego_id in ego_statistics.keys():
        ego_lane = f'{ego_statistics[ego_id][3]}_{ego_statistics[ego_id][4]}'
        ego_lane_stats[ego_id] = all_lane_stats[ego_lane]
        ego_lane_index = ego_statistics[ego_id][4]
        if ego_lane_index > 0:
            right_lane = f'{ego_statistics[ego_id][3]}_{ego_lane_index - 1}'
            right_lane_stats[ego_id] = all_lane_stats[right_lane]
        else:
            right_lane_stats[ego_id] = np.zeros(6)
        if ego_lane_index < edge_lane_num[ego_statistics[ego_id][3]]-1:
            left_lane = f'{ego_statistics[ego_id][3]}_{ego_lane_index + 1}'
            left_lane_stats[ego_id] = all_lane_stats[left_lane]
        else:
            left_lane_stats[ego_id]= np.zeros(6)

    # convert to 2D array (18 * 6)
    all_lane_stats = np.array(list(all_lane_stats.values()))
    all_lane_state_simple = np.array(list(all_lane_state_simple.values()))

    # ############################## bottle_neck 的信息 ##############################
    # 车辆距离bottle_neck
    bottle_neck_position_x = bottle_neck_positions[0] / 700
    bottle_neck_position_y = bottle_neck_positions[1]

    feature_vector = {}
    # feature_vector['road_structure'] = np.array([0, 0, bottle_neck_position_x, bottle_neck_position_y, 4,
    #                                              bottle_neck_position_x, bottle_neck_position_y, 1, 0, 2])
    feature_vector['road_end'] = np.array([1, 0])
    feature_vector['bottleneck'] = np.array([bottle_neck_position_x, bottle_neck_position_y])
    # A function to flatten a dictionary structure into 1D array
    def flatten_to_1d(data_dict):
        flat_list = []
        for key, item in data_dict.items():
            if isinstance(item, list):
                flat_list.extend(item)
            elif isinstance(item, np.ndarray):
                flat_list.extend(item.flatten())
        size_obs = np.size(np.array(flat_list))
        return np.array(flat_list)

    veh_target = get_target(ego_statistics, lane_statistics)
    feature_vectors_current = {}
    shared_feature_vectors_current = {}
    flat_surround_vehs = {key: [] for key in veh_target}
    for ego_id in ego_statistics.keys():
        feature_vectors_current[ego_id] = feature_vector.copy()
        feature_vectors_current[ego_id]['self_stats'] = ego_stats[ego_id][:3]
        feature_vectors_current[ego_id]['target'] = veh_target[ego_id]
        feature_vectors_current[ego_id]['distance_bott'] = np.array([bottle_neck_position_x - ego_stats[ego_id][0], 0])
        feature_vectors_current[ego_id]['distance_end'] = np.array([1 - ego_stats[ego_id][0], 0])
        flat_surround_vehs[ego_id] = [item for sublist in surround_vehs_stats[ego_id] for item in sublist]
        if len(flat_surround_vehs[ego_id]) < 18:
            flat_surround_vehs[ego_id] += [0] * (18 - len(flat_surround_vehs[ego_id]))
        feature_vectors_current[ego_id]['surround_vehs_stats'] = flat_surround_vehs[ego_id]
        # feature_vectors_current[ego_id]['ego_lane_stats'] = ego_lane_stats[ego_id]
        # feature_vectors_current[ego_id]['left_lane_stats'] = left_lane_stats[ego_id]
        # feature_vectors_current[ego_id]['right_lane_stats'] = right_lane_stats[ego_id]

        shared_feature_vectors_current[ego_id] = feature_vector.copy()
        shared_feature_vectors_current[ego_id]['cav_stats'] = global_cav
        shared_feature_vectors_current[ego_id]['lane_stats'] = all_lane_state_simple

    # Flatten the dictionary structure
    feature_vectors_current_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                               feature_vectors_current.items()}
    feature_vectors = {key: {} for key in ego_statistics.keys()}
    if self.use_hist_info:
        # for ego_id, feature_vector_current in feature_vectors_current_flatten.items():
        for ego_id, feature_vector_current in feature_vectors_current_flatten.items():
            if ego_id not in feature_vectors:
                print('ego_id not in feature_vectors')
            if ego_id not in self.hist_info['hist_4']:
                print('ego_id not in hist_info')
            feature_vectors[ego_id]['1hist_4'] = self.hist_info['hist_4'][ego_id]
            feature_vectors[ego_id]['2hist_3'] = self.hist_info['hist_3'][ego_id]
            feature_vectors[ego_id]['3hist_2'] = self.hist_info['hist_2'][ego_id]
            feature_vectors[ego_id]['4hist_1'] = self.hist_info['hist_1'][ego_id]
            feature_vectors[ego_id]['5current'] = feature_vector_current

            self.hist_info['hist_4'][ego_id] = self.hist_info['hist_3'][ego_id]
            self.hist_info['hist_3'][ego_id] = self.hist_info['hist_2'][ego_id]
            self.hist_info['hist_2'][ego_id] = self.hist_info['hist_1'][ego_id]
            self.hist_info['hist_1'][ego_id] = feature_vector_current
        feature_vectors_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                                   feature_vectors.items()}
    else:
        for ego_id, feature_vector_current in feature_vectors_current.items():
            feature_vectors[ego_id] = feature_vector_current
        feature_vectors_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                                   feature_vectors.items()}

    shared_feature_flatten = {ego_id: flatten_to_1d(shared_feature_vector) for ego_id, shared_feature_vector in
                                shared_feature_vectors_current.items()}
    return shared_feature_vectors_current, shared_feature_flatten, feature_vectors, feature_vectors_flatten

def compute_hierarchical_ego_vehicle_features(
        self,
        hdv_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        ego_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        lane_statistics: Dict[str, List[float]],
        unique_edges: List[str],
        edge_lane_num: Dict[str, int],
        bottle_neck_positions: Tuple[float],
        ego_ids: List[str]
) -> Dict[str, List[float]]:
    """计算每一个 ego vehicle 的特征

    Args:
        ego_statistics (Dict[str, List[Union[float, str, Tuple[int]]]]): ego vehicle 的信息
        lane_statistics (Dict[str, List[float]]): 路网的信息
        unique_edges (List[str]): 所有考虑的 edge
        edge_lane_num (Dict[str, int]): 每一个 edge 对应的车道
        bottle_neck_positions (Tuple[float]): bottleneck 的坐标
    output:
        surround_hdv_stats: 12 * 3-relative x,y,v
        surround_cav_stats: 12 * 3-relative x,y,v
        ego_lane_stats: 4--vehicle count, lane density, speed-mean, waiting time-mean
        left_lane_stats: 4--vehicle count, lane density, speed-mean, waiting time-mean
        right_lane_stats: 4--vehicle count, lane density, speed-mean, waiting time-mean
        bottleneck_position & distance: 1 * 2 + 1
        self_stats: 1 * 13
    """
    # 建立全局车辆图关系
    vehicle_order = {'CAV_0': 0, 'CAV_1': 1, 'CAV_2': 2, 'CAV_3': 3, 'CAV_4': 4, 'CAV_5': 5, 'CAV_6': 6, 'CAV_7': 7,
                     'CAV_8': 8,
                     'CAV_9': 9, 'CAV_10': 10, 'CAV_11': 11, 'CAV_12': 12, 'CAV_13': 13, 'CAV_14': 14, 'CAV_15': 15,
                     'HDV_0': 16, 'HDV_1': 17, 'HDV_2': 18, 'HDV_3': 19, 'HDV_4': 20, 'HDV_5': 21, 'HDV_6': 22,
                     'HDV_7': 23,
                     'HDV_8': 24, 'HDV_9': 25, 'HDV_10': 26, 'HDV_11': 27, 'HDV_12': 28, 'HDV_13': 29, 'HDV_14': 30,
                     'HDV_15': 31,
                     'HDV_16': 32, 'HDV_17': 33, 'HDV_18': 34, 'HDV_19': 35, 'HDV_20': 36, 'HDV_21': 37, 'HDV_22': 38,
                     'HDV_23': 39}

    vehicle_relation_graph = np.zeros((self.max_num_CAVs + self.max_num_HDVs, self.max_num_CAVs + self.max_num_HDVs))
    # ############################## 所有HDV的信息 ############################## 13
    hdv_stats = {}
    surround_modes = {
        'left_followers': 0b000,  # Left and followers
        'right_followers': 0b001,  # Left and leaders
        'left_leaders': 0b010,  # Right and followers
        'right_leaders': 0b011  # Right and leaders
    }

    # A function to flatten a dictionary structure into 1D array
    def flatten_to_1d(data_dict):
        flat_list = []
        for key, item in data_dict.items():
            if isinstance(item, list):
                flat_list.extend(item)
            elif isinstance(item, np.ndarray):
                flat_list.extend(item.flatten())
        size_obs = np.size(np.array(flat_list))
        return np.array(flat_list)

    for hdv_id, hdv_info in hdv_statistics.items():
        speed, position, heading, road_id, lane_index = hdv_info
        # 速度归一化  - 1
        normalized_speed = speed / 15.0
        # 位置归一化  - 2
        position_x, position_y = position
        normalized_position_x = position_x / 700
        normalized_position_y = position_y / 3.2
        # 转向归一化 - 1
        normalized_heading = heading / 360
        # One-hot encode road_id - 5
        road_id_one_hot = one_hot_encode(road_id, unique_edges)
        # One-hot encode lane_index - 4
        lane_index_one_hot = one_hot_encode(lane_index, list(range(edge_lane_num.get(road_id, 0))))
        # 如果车道数不足4个, 补0 对齐到最多数量的lane num
        if len(lane_index_one_hot) < 4:
            lane_index_one_hot += [0] * (4 - len(lane_index_one_hot))
        hdv_stats[hdv_id] = [normalized_position_x, normalized_position_y, normalized_speed, normalized_heading, -1, -1, -1]
        if self.use_hist_info:
            for i in range(self.hist_length - 1):
                self.vehicles_hist[f'hist_{self.hist_length - i}'][hdv_id] = copy.deepcopy(self.vehicles_hist[f'hist_{self.hist_length - i - 1}'][hdv_id])
            self.vehicles_hist['hist_1'][hdv_id] = [position_x, position_y, speed, heading, -1, -1, -1]
        # hdv_stats[hdv_id] = [normalized_speed, normalized_position_x, normalized_position_y,
        #                      normalized_heading] + road_id_one_hot + lane_index_one_hot
        if self.use_gui:
            import traci as traci
        else:
            import libsumo as traci
        for key, mode in surround_modes.items():
            neighbors = traci.vehicle.getNeighbors(hdv_id, mode)
            if neighbors and neighbors[0][0] in vehicle_order:
                vehicle_relation_graph[vehicle_order[hdv_id], vehicle_order[neighbors[0][0]]] = 1
                vehicle_relation_graph[vehicle_order[neighbors[0][0]], vehicle_order[hdv_id]] = 1
        front_vehicle = traci.vehicle.getLeader(hdv_id)
        if front_vehicle and front_vehicle[0] in vehicle_order:
            vehicle_relation_graph[vehicle_order[hdv_id], vehicle_order[front_vehicle[0]]] = 1
        following_vehicle = traci.vehicle.getFollower(hdv_id)
        if following_vehicle and following_vehicle[0] in vehicle_order:
            vehicle_relation_graph[vehicle_order[following_vehicle[0]], vehicle_order[hdv_id]] = 1

    # convert to 2D array (24 * 3)  - 24 is max number of HDVs
    for i in range(self.max_num_HDVs):
        if 'HDV_'+str(i) not in hdv_stats:
            hdv_stats['HDV_'+str(i)] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    hdv_stats = np.array(list(hdv_stats.values()))
    if 0 < hdv_stats.shape[0] <= self.max_num_HDVs:
        # add 0 to make sure the shape is (max_num_HDVs, 13)
        hdv_stats = np.vstack([hdv_stats, np.zeros((self.max_num_HDVs - hdv_stats.shape[0], 7))])
    elif hdv_stats.shape[0] == 0:
        hdv_stats = np.zeros((self.max_num_HDVs, 7))
    # ############################## 所有CAV的信息 ############################## 13
    cav_stats = {}
    ego_stats = {}
    surround_vehs_stats = {key:[] for key in ego_ids}
    ego_cav_motion = {key:[] for key in ego_ids}
    ego_hdv_motion = {key:[] for key in ego_ids}
    surround_stats = {key:[] for key in ego_ids}
    surround_relation_graph = {key: np.ones((1 + 9, 1 + 9)) for key in ego_ids}
    surround_order = {'ego': 0, 'front': 1, 'front_expand': 2, 'back': 3,
                      'left_followers': 4, 'right_followers': 5,
                      'left_leaders': 6, 'left_leaders_expand': 7,
                      'right_leaders': 8, 'right_leaders_expand': 9}
    expand_surround_stats = {key: {surround_key: np.zeros(20) for surround_key in surround_order.keys()} for key in ego_ids}
    surround_IDs = {key: {surround_key: 0 for surround_key in surround_order.keys()} for key in ego_ids}
    for ego_id, ego_info in ego_statistics.items():
        # ############################## 自己车的信息 ############################## 13
        position, speed, heading, road_id, lane_index, surroundings, expand_surroundings = ego_info
        # 速度归一化  - 1
        normalized_speed = speed / 15.0
        # 位置归一化  - 2
        position_x, position_y = position
        normalized_position_x = position_x / 700
        normalized_position_y = position_y / 3.2
        # 转向归一化 - 1
        normalized_heading = heading / 360
        # One-hot encode road_id - 5
        road_id_one_hot = one_hot_encode(road_id, unique_edges)
        # One-hot encode lane_index - 4
        lane_index_one_hot = one_hot_encode(lane_index, list(range(edge_lane_num.get(road_id, 0))))
        # 如果车道数不足4个, 补0 对齐到最多数量的lane num
        if len(lane_index_one_hot) < 4:
            lane_index_one_hot += [0] * (4 - len(lane_index_one_hot))
        # ############################## 自己车的运动信息 ############################## 20
        ego_hist_4 = self.vehicles_hist['hist_5'][ego_id]
        ego_hist_3 = self.vehicles_hist['hist_4'][ego_id]
        ego_hist_2 = self.vehicles_hist['hist_3'][ego_id]
        ego_hist_1 = self.vehicles_hist['hist_2'][ego_id]
        relative_hist_4 = [(ego_hist_4[0] - position_x)/700, (ego_hist_4[1] - position_y)/3.2, (speed - ego_hist_4[2])/15] if ego_hist_4[0] != 0 else [0, 0, 0]
        relative_hist_3 = [(ego_hist_3[0] - position_x)/700, (ego_hist_3[1] - position_y)/3.2, (speed - ego_hist_3[2])/15] if ego_hist_3[0] != 0 else [0, 0, 0]
        relative_hist_2 = [(ego_hist_2[0] - position_x)/700, (ego_hist_2[1] - position_y)/3.2, (speed - ego_hist_2[2])/15] if ego_hist_2[0] != 0 else [0, 0, 0]
        relative_hist_1 = [(ego_hist_1[0] - position_x)/700, (ego_hist_1[1] - position_y)/3.2, (speed - ego_hist_1[2])/15] if ego_hist_1[0] != 0 else [0, 0, 0]
        relative_current = [0, 0, 0]
        ego_hdv_motion[ego_id] = relative_hist_4 + relative_hist_3 + relative_hist_2 + relative_hist_1 + relative_current
        last_action = copy.deepcopy(self.lowercontroller_action[ego_id][-1]) if self.lowercontroller_action[ego_id] else [0, 0, 0]
        last_action = [last_action[0]/4, last_action[1]/4, last_action[2]/15]
        relative_motion = relative_current + last_action
        relative_motion += [0] * (15 - len(relative_motion))
        ego_cav_motion[ego_id] = relative_motion
        # ############################## 周车信息 ############################## 18
        # 提取surrounding的信息 -18
        for index, (_, statistics) in enumerate(surroundings.items()):
            relat_x, relat_y, relat_v = statistics[1:4]
            surround_ID = statistics[0]
            surround_vehs_stats[ego_id].append([relat_x/700, relat_y, relat_v/15])  # relat_x, relat_y, relat_v
            if surround_ID[:3] == 'HDV':
                target_hist_4 = self.vehicles_hist['hist_5'][statistics[0]]
                target_hist_3 = self.vehicles_hist['hist_4'][statistics[0]]
                target_hist_2 = self.vehicles_hist['hist_3'][statistics[0]]
                target_hist_1 = self.vehicles_hist['hist_2'][statistics[0]]
                relative_hist_4 = [(target_hist_4[0] - position_x)/700, (target_hist_4[1] - position_y)/3.2, (speed - target_hist_4[2])/15] if target_hist_4[0] != 0 else [0, 0, 0]
                relative_hist_3 = [(target_hist_3[0] - position_x)/700, (target_hist_3[1] - position_y)/3.2, (speed - target_hist_3[2])/15] if target_hist_3[0] != 0 else [0, 0, 0]
                relative_hist_2 = [(target_hist_2[0] - position_x)/700, (target_hist_2[1] - position_y)/3.2, (speed - target_hist_2[2])/15] if target_hist_2[0] != 0 else [0, 0, 0]
                relative_hist_1 = [(target_hist_1[0] - position_x)/700, (target_hist_1[1] - position_y)/3.2, (speed - target_hist_1[2])/15] if target_hist_1[0] != 0 else [0, 0, 0]
                relative_current = [relat_x/700, relat_y/3.2, relat_v/15]
                relative_motion = [0] + relative_hist_4 + relative_hist_3 + relative_hist_2 + relative_hist_1 + relative_current
                surround_stats[ego_id].append(relative_motion)
            elif surround_ID[:3] == 'CAV':
                relative_current = [relat_x/700, relat_y/3.2, relat_v/15]
                last_action = copy.deepcopy(self.lowercontroller_action[ego_id][-1]) if self.lowercontroller_action[statistics[0]] else [0, 0, 0]
                last_action = [last_action[0]/4, last_action[1]/4, last_action[2]/15]
                relative_motion = [1] + relative_current + last_action
                if len(relative_motion) < 16:
                    relative_motion += [0] * (16 - len(relative_motion))
                surround_stats[ego_id].append(relative_motion)
            vehicle_relation_graph[vehicle_order[ego_id], vehicle_order[statistics[0]]] = 1
            vehicle_relation_graph[vehicle_order[statistics[0]], vehicle_order[ego_id]] = 1
        cav_last_action = copy.deepcopy(self.lowercontroller_action[ego_id][-1]) if self.lowercontroller_action[ego_id] else [0, 0, 0]
        cav_last_action = [cav_last_action[0] / 4, cav_last_action[1] / 4, cav_last_action[2] / 15]
        cav_stats[ego_id] = [normalized_position_x, normalized_position_y, normalized_speed, normalized_heading] + cav_last_action
        ego_stats[ego_id] = [normalized_position_x, normalized_position_y, normalized_speed,
                             normalized_heading] + road_id_one_hot + lane_index_one_hot
        if self.use_hist_info:
            for i in range(self.hist_length - 1):
                self.vehicles_hist[f'hist_{self.hist_length - i}'][ego_id] = copy.deepcopy(
                    self.vehicles_hist[f'hist_{self.hist_length - i - 1}'][ego_id])
            last_action = copy.deepcopy(self.lowercontroller_action[ego_id][-1]) if self.lowercontroller_action[ego_id] else [0, 0, 0]
            self.vehicles_hist['hist_1'][ego_id] = [position_x, position_y, speed, heading] + last_action
    # convert to 2D array
    for i in range(self.max_num_CAVs):
        if 'CAV_'+str(i) not in cav_stats:
            cav_stats['CAV_'+str(i)] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cav_stats = np.array(list(cav_stats.values()))
    if 0 < cav_stats.shape[0] <= self.max_num_CAVs:
        # add 0 to make sure the shape is (12, 13)
        cav_stats = np.vstack([cav_stats, np.zeros((self.max_num_CAVs - cav_stats.shape[0], 7))])
    elif cav_stats.shape[0] == 0:
        cav_stats = np.zeros((self.max_num_CAVs, 7))

    if len(ego_stats) != len(ego_ids):
        for ego_id in ego_ids:
            if ego_id not in ego_stats:
                ego_stats[ego_id] = [0.0] * 13

    # ############################## lane_statistics 的信息 ############################## 18
    # Initialize a list to hold all lane statistics
    all_lane_stats = {}

    # Iterate over all possible lanes to get their statistics
    for lane_id, lane_info in lane_statistics.items():
        # - vehicle_count: 当前车道的车辆数 1
        # - lane_density: 当前车道的车辆密度 1
        # - lane_length: 这个车道的长度 1
        # - speeds: 在这个车道上车的速度 1 (mean)
        # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
        # - accumulated_waiting_times: 车的累积等待时间 1 (mean)--delete
        # - lane_CAV_penetration: 这个车道上的CAV占比 1

        # all_lane_stats[lane_id] = lane_info[:4] + lane_info[6:7] + lane_info[9:10]
        lane_info[0] = lane_info[0] / 10
        lane_info[2] = lane_info[2] / 700
        lane_info[3] = lane_info[3] / 15
        all_lane_stats[lane_id] = lane_info[:4] + lane_info[6:7] + lane_info[13:14]

    ego_lane_stats = {}
    left_lane_stats = {}
    right_lane_stats = {}
    for ego_id in ego_statistics.keys():
        ego_lane = f'{ego_statistics[ego_id][3]}_{ego_statistics[ego_id][4]}'
        ego_lane_stats[ego_id] = all_lane_stats[ego_lane]
        ego_lane_index = ego_statistics[ego_id][4]
        if ego_lane_index > 0:
            right_lane = f'{ego_statistics[ego_id][3]}_{ego_lane_index - 1}'
            right_lane_stats[ego_id] = all_lane_stats[right_lane]
        else:
            right_lane_stats[ego_id] = np.zeros(6)
        if ego_lane_index < edge_lane_num[ego_statistics[ego_id][3]]-1:
            left_lane = f'{ego_statistics[ego_id][3]}_{ego_lane_index + 1}'
            left_lane_stats[ego_id] = all_lane_stats[left_lane]
        else:
            left_lane_stats[ego_id]= np.zeros(6)

    # convert to 2D array (18 * 6)
    all_lane_stats = np.array(list(all_lane_stats.values()))

    # ############################## bottle_neck 的信息 ##############################
    # 车辆距离bottle_neck
    bottle_neck_position_x = bottle_neck_positions[0] / 700
    bottle_neck_position_y = bottle_neck_positions[1]

    feature_vector = {}
    feature_vector['road_structure'] = np.array([0, 0, bottle_neck_position_x, bottle_neck_position_y, 4,
                                                 bottle_neck_position_x, bottle_neck_position_y, 1, 0, 2])
    feature_vector['bottle_neck_position'] = np.array([bottle_neck_position_x, bottle_neck_position_y])
    feature_vector['road_end'] = np.array([1, 0])

    veh_target = get_target(ego_statistics, lane_statistics)
    feature_vectors_current = {}
    shared_feature_vectors = {}
    flat_surround_vehs = {key: [] for key in ego_ids}
    flat_expand_surround = {key: [] for key in ego_ids}
    flat_surround_IDs = {key: [] for key in ego_ids}
    flat_surround_relation = {key: [] for key in ego_ids}
    for ego_id, ego_info in ego_statistics.items():
        position, speed, heading, road_id, lane_index, surroundings, expand_surroundings = ego_info
        position_x, position_y = position
        surround_IDs[ego_id]['ego'] = int(ego_id[4:]) + 200
        surround_relation_graph[ego_id][1:, 1:] = 0
        ego_hist_4 = self.vehicles_hist['hist_5'][ego_id]
        ego_hist_3 = self.vehicles_hist['hist_4'][ego_id]
        ego_hist_2 = self.vehicles_hist['hist_3'][ego_id]
        ego_hist_1 = self.vehicles_hist['hist_2'][ego_id]
        relative_hist_4 = [(ego_hist_4[0] - position_x)/700, (ego_hist_4[1] - position_y)/3.2,
                           (speed - ego_hist_4[2])/15, (ego_hist_4[3] - heading)/360] if ego_hist_4[0] != 0 else [0, 0, 0, 0]
        relative_hist_3 = [(ego_hist_3[0] - position_x)/700, (ego_hist_3[1] - position_y)/3.2,
                           (speed - ego_hist_3[2])/15, (ego_hist_3[3] - heading)/360] if ego_hist_3[0] != 0 else [0, 0, 0, 0]
        relative_hist_2 = [(ego_hist_2[0] - position_x)/700, (ego_hist_2[1] - position_y)/3.2,
                           (speed - ego_hist_2[2])/15, (ego_hist_2[3] - heading)/360] if ego_hist_2[0] != 0 else [0, 0, 0, 0]
        relative_hist_1 = [(ego_hist_1[0] - position_x)/700, (ego_hist_1[1] - position_y)/3.2,
                           (speed - ego_hist_1[2])/15, (ego_hist_1[3] - heading)/360] if ego_hist_1[0] != 0 else [0, 0, 0, 0]
        reltaive_current = [0, 0, 0, 0]
        ego_motion = relative_hist_4 + relative_hist_3 + relative_hist_2 + relative_hist_1 + reltaive_current
        expand_surround_stats[ego_id]['ego'] = ego_motion
        for index, (_, statistics) in enumerate(expand_surroundings.items()):
            relat_x, relat_y, relat_v = statistics[1:4]
            target_hist_4 = self.vehicles_hist['hist_5'][statistics[0]]
            target_hist_3 = self.vehicles_hist['hist_4'][statistics[0]]
            target_hist_2 = self.vehicles_hist['hist_3'][statistics[0]]
            target_hist_1 = self.vehicles_hist['hist_2'][statistics[0]]
            relative_hist_4 = [(target_hist_4[0] - position_x)/700, (target_hist_4[1] - position_y)/3.2,
                               (speed - target_hist_4[2])/15, (target_hist_4[3] - heading)/360] if target_hist_4[0] != 0 else [0, 0, 0, 0]
            relative_hist_3 = [(target_hist_3[0] - position_x)/700, (target_hist_3[1] - position_y)/3.2,
                               (speed - target_hist_3[2])/15, (target_hist_3[3] - heading)/360] if target_hist_3[0] != 0 else [0, 0, 0, 0]
            relative_hist_2 = [(target_hist_2[0] - position_x)/700, (target_hist_2[1] - position_y)/3.2,
                               (speed - target_hist_2[2])/15, (target_hist_2[3] - heading)/360] if target_hist_2[0] != 0 else [0, 0, 0, 0]
            relative_hist_1 = [(target_hist_1[0] - position_x)/700, (target_hist_1[1] - position_y)/3.2,
                               (speed - target_hist_1[2])/15, (target_hist_1[3] - heading)/360] if target_hist_1[0] != 0 else [0, 0, 0, 0]
            reltaive_current = [relat_x/700, relat_y/3.2, relat_v/15, 0]
            relative_motion = relative_hist_4 + relative_hist_3 + relative_hist_2 + relative_hist_1 + reltaive_current
            expand_surround_stats[ego_id][_] = relative_motion
            if statistics[0][:3] == 'HDV':
                surround_IDs[ego_id][_] = int(statistics[0][4:]) + 100
            else:
                surround_IDs[ego_id][_] = int(statistics[0][4:]) + 200
            for key, num in surround_order.items():
                if key in expand_surroundings.keys():
                    dist_x = expand_surroundings[key][1] - relat_x
                    dist_y = expand_surroundings[key][2] - relat_y
                    dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
                    if dist < 30:
                        surround_relation_graph[ego_id][num, surround_order[_]] = 1
                        surround_relation_graph[ego_id][surround_order[_], num] = 1


        feature_vectors_current[ego_id] = feature_vector.copy()
        feature_vectors_current[ego_id]['target'] = veh_target[ego_id]
        feature_vectors_current[ego_id]['self_stats'] = ego_stats[ego_id]
        feature_vectors_current[ego_id]['distance_bott'] = np.array([bottle_neck_position_x - ego_stats[ego_id][0], 0])
        feature_vectors_current[ego_id]['distance_end'] = np.array([1 - ego_stats[ego_id][0], 0])
        last_actor_action = copy.deepcopy(self.actor_action[ego_id][-1]) if self.actor_action[ego_id] else [0, 0, 0]
        last_actor_action = [last_actor_action[0] / 4, last_actor_action[1] / 4, last_actor_action[2] / 15]
        feature_vectors_current[ego_id]['actor_action'] = np.array(last_actor_action)
        last_actual_action = copy.deepcopy(self.lowercontroller_action[ego_id][-1]) if self.lowercontroller_action[ego_id] else [0, 0, 0]
        last_actual_action = [last_actual_action[0] / 4, last_actual_action[1] / 4, last_actual_action[2] / 15]
        feature_vectors_current[ego_id]['actual_action'] = np.array(last_actual_action)
        feature_vectors_current[ego_id]['ego_cav_motion'] = np.array(ego_cav_motion[ego_id])
        feature_vectors_current[ego_id]['ego_hdv_motion'] = np.array(ego_hdv_motion[ego_id])

        flat_surround_vehs[ego_id] = [item for sublist in surround_stats[ego_id] for item in sublist]
        if len(flat_surround_vehs[ego_id]) < 6*16:
            flat_surround_vehs[ego_id] += [0] * (6*16 - len(flat_surround_vehs[ego_id]))
        feature_vectors_current[ego_id]['surround_stats'] = flat_surround_vehs[ego_id]

        flat_expand_surround[ego_id] = flatten_to_1d(expand_surround_stats[ego_id])
        feature_vectors_current[ego_id]['expand_surround_stats'] = flat_expand_surround[ego_id]

        flat_surround_relation[ego_id] = [item for sublist in surround_relation_graph[ego_id] for item in sublist]
        feature_vectors_current[ego_id]['surround_relation_graph'] = flat_surround_relation[ego_id]

        for key in surround_IDs[ego_id]:
            flat_surround_IDs[ego_id].append(surround_IDs[ego_id][key])
        feature_vectors_current[ego_id]['surround_IDs'] = flat_surround_IDs[ego_id]

        feature_vectors_current[ego_id]['ego_lane_stats'] = ego_lane_stats[ego_id]
        feature_vectors_current[ego_id]['left_lane_stats'] = left_lane_stats[ego_id]
        feature_vectors_current[ego_id]['right_lane_stats'] = right_lane_stats[ego_id]

        feature_vectors_current[ego_id]['hdv_stats'] = hdv_stats
        feature_vectors_current[ego_id]['cav_stats'] = cav_stats

        # shared_feature_vectors[ego_id] = feature_vector.copy()  # EP mode
        shared_feature_vectors[ego_id] = copy.deepcopy(feature_vectors_current[ego_id])  # FP mode
        shared_feature_vectors[ego_id]['veh_relation'] = vehicle_relation_graph
        shared_feature_vectors[ego_id]['all_lane_stats'] = all_lane_stats

    # Flatten the dictionary structure
    feature_vectors_current_flatten = {ego_id: flatten_to_1d(feature_vectors) for ego_id, feature_vectors in
                               feature_vectors_current.items()}

    shared_feature_vectors_flatten = {ego_id: flatten_to_1d(shared_feature_vector) for ego_id, shared_feature_vector in shared_feature_vectors.items()}

    return shared_feature_vectors, shared_feature_vectors_flatten, feature_vectors_current, feature_vectors_current_flatten

def compute_centralized_vehicle_features(lane_statistics, feature_vectors, bottle_neck_positions):
    shared_features = {}

    # ############################## 所有车的速度 位置 转向信息 ##############################
    all_vehicle = []
    for _, ego_feature in feature_vectors.items():
        all_vehicle += ego_feature[:4]

    # ############################## lane_statistics 的信息 ##############################
    # Initialize a list to hold all lane statistics
    all_lane_stats = []

    # Iterate over all possible lanes to get their statistics
    for _, lane_info in lane_statistics.items():
        # - vehicle_count: 当前车道的车辆数 1
        # - lane_density: 当前车道的车辆密度 1
        # - lane_length: 这个车道的长度 1
        # - speeds: 在这个车道上车的速度 1 (mean)
        # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
        # - accumulated_waiting_times: 车的累积等待时间 1 (mean, max)

        all_lane_stats += lane_info[:4] + lane_info[6:7] + lane_info[9:10]

    # ############################## bottleneck 的信息 ##############################
    # 车辆距离bottle_neck
    bottle_neck_position_x = bottle_neck_positions[0] / 700
    bottle_neck_position_y = bottle_neck_positions[1]

    for ego_id in feature_vectors.keys():
        shared_features[ego_id] = [bottle_neck_position_x, bottle_neck_position_y] + all_vehicle + all_lane_stats

    # assert all(len(shared_feature) == 130 for shared_feature in shared_features.values())

    return shared_features

def compute_centralized_vehicle_features_hierarchical_version(
        obs_size, shared_obs_size, lane_statistics,
        feature_vectors_current, feature_vectors_current_flatten,
        feature_vectors, feature_vectors_flatten, ego_ids):
    shared_features = {}
    actor_features = {}

    for ego_id in feature_vectors.keys():
        actor_features[ego_id] = feature_vectors[ego_id].copy() # shared_features--actor / critic , can there output different obs to actor and critic?
    for ego_id in feature_vectors_current.keys():
        shared_features[ego_id] = feature_vectors_current[ego_id].copy()
    def flatten_to_1d(data_dict):
        flat_list = []
        for key, item in data_dict.items():
            if isinstance(item, list):
                flat_list.extend(item)
            elif isinstance(item, np.ndarray):
                flat_list.extend(item.flatten())
        return np.array(flat_list)

    shared_features_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                               shared_features.items()}
    actor_features_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                               actor_features.items()}
    if len(shared_features_flatten) != len(ego_ids):
        for ego_id in feature_vectors_flatten.keys():
            if ego_id not in shared_features_flatten:
                shared_features_flatten[ego_id] = np.zeros(shared_obs_size)
    if len(actor_features_flatten) != len(ego_ids):
        for ego_id in feature_vectors_flatten.keys():
            if ego_id not in actor_features_flatten:
                actor_features_flatten[ego_id] = np.zeros(obs_size)
    if len(shared_features_flatten) != len(ego_ids):
        print("Error: len(shared_features_flatten) != len(ego_ids)")
    if len(feature_vectors_flatten) != len(ego_ids):
        print("Error: len(feature_vectors_flatten) != len(ego_ids)")
    return actor_features, actor_features_flatten, shared_features, shared_features_flatten
