import json
# import carla
import argparse
import xml.etree.ElementTree as ET
# from agents.navigation.global_route_planner import GlobalRoutePlanner
import os
import atexit
import subprocess
import time
import random

scenes = dict(
    highway_merge_slow = [23771, 2283, 23901, ],
    highway_cutin = [2286, ],
    highway_exit_merge = [23658, ],
    static_cutin = [2709, ],
    parking_cutin = [1711, ],
    constuction_two_ways = [1825, ],
    constuction_merge = [2509, ],
    left_turn_merge = [2201, 3086, 28099, 28087],
    left_turn_opposite = [3936, 2086, 2084, ],
    left_turn_ped_crossing = [2144, 2164, ],
    left_turn_naive = [2082, 2416, 14842, 2373],
    left_turn_tjunction = [26458, ],
    left_turn_blocked = [27494, ],
    right_turn_merge = [2050, 26956, 2115, ],
    right_turn_blocked = [27532, 2204, ],
    right_turn_naive = [2390, ],
    hidden_ped_crossing = [17752, 3248, ],
    parked_obstacle_merge = [1773, 2539, 2534, ],
    parked_obstacle_two_ways = [1852, 2664, 2668, ],
    cyclist_merge = [1792, 1790, ],
    cyclist_two_ways = [3436, ],
    sequential_merge = [17563, 23695, ],
    yield_emergency = [3364, ],
    lane_follow = [2273, 2790, 3540, 3561, 28154, 26406, ],
    ped_crossing_nonsig_straight = [14194, ],
    parking_exit = [1956, ],
    fire_truck = [2129, 2127, ],
    vehicle_door_two_ways = [3464, ],
)

scenes["left_turn"] = scenes["left_turn_merge"] + scenes["left_turn_opposite"] + scenes["left_turn_ped_crossing"] + scenes["left_turn_naive"] + scenes["left_turn_tjunction"] + scenes["left_turn_blocked"]
scenes["right_turn"] = scenes["right_turn_merge"] + scenes["right_turn_blocked"] + scenes["right_turn_naive"]
scenes["merge"] = scenes["highway_merge_slow"] + scenes["highway_exit_merge"] + scenes["constuction_merge"] + scenes["left_turn_merge"] + scenes["right_turn_merge"] + scenes["parked_obstacle_merge"] + scenes["cyclist_merge"] + scenes["sequential_merge"]
scenes["cutin"] = scenes["highway_cutin"] + scenes["static_cutin"] + scenes["parking_cutin"]
scenes["two_ways"] = scenes["constuction_two_ways"] + scenes["parked_obstacle_two_ways"] + scenes["cyclist_two_ways"] + scenes["vehicle_door_two_ways"]
scenes["corner"] = scenes["fire_truck"] + scenes["yield_emergency"]
scenes["ped_crossing"] = scenes["left_turn_ped_crossing"] + scenes["hidden_ped_crossing"] + scenes["ped_crossing_nonsig_straight"]
full_route = []
for scene, route in scenes.items():
    full_route = full_route + route
scenes["all"] = list(set(full_route))
scenes["all_220"] = [1711, 1773, 1790, 1792, 1825, 1833, 1852, 1956, 2050, 2082, 2084, 2086, 2091, 2115, 2127, 2129, 2143, 2144, 2164, 2201, 2204, 2215, 2273, 2283, 2286, 2373, 2390, 2397, 2403, 2416, 2509, 2513, 2534, 2539, 2554, 2606, 2643, 2664, 2668, 2709, 2715, 2790, 2802, 2844, 2847, 2881, 2903, 2913, 2943, 2989, 3048, 3072, 3074, 3080, 3086, 3090, 3093, 3099, 3100, 3144, 3178, 3184, 3189, 3248, 3255, 3307, 3364, 3373, 3378, 3380, 3410, 3436, 3457, 3464, 3472, 3476, 3482, 3514, 3520, 3540, 3561, 3564, 3572, 3575, 3666, 3670, 3676, 3697, 3708, 3712, 3717, 3731, 3737, 3749, 3785, 3800, 3813, 3865, 3869, 3876, 3890, 3904, 3905, 3936, 4183, 4468, 4669, 4683, 4937, 10857, 11381, 11715, 11755, 14194, 14842, 14862, 14909, 17563, 17569, 17598, 17635, 17655, 17752, 18252, 18305, 18311, 18356, 20920, 23658, 23659, 23670, 23687, 23695, 23700, 23708, 23771, 23901, 23910, 23918, 23930, 24041, 24071, 24078, 24092, 24098, 24206, 24211, 24224, 24240, 24252, 24258, 24294, 24330, 24333, 24340, 24367, 24757, 24759, 24781, 24784, 24785, 24795, 24816, 24841, 25300, 25318, 25358, 25378, 25381, 25383, 25424, 25439, 25845, 25854, 25857, 25863, 25865, 25896, 25928, 25951, 25955, 25968, 25975, 26393, 26394, 26396, 26401, 26405, 26406, 26408, 26418, 26435, 26456, 26458, 26944, 26950, 26956, 26966, 26990, 27018, 27494, 27506, 27515, 27529, 27532, 27582, 28035, 28048, 28087, 28093, 28099, 28111, 28154, 28198, 28210, 28219, 28229, 28241, 28243, 28330]

def get_infraction_status(record):
    for infraction,  value in record['infractions'].items():
        if infraction == "min_speed_infractions":
            continue
        elif len(value) > 0:
            return True
    return False

def update_Ability(scenario_name, Ability_Statistic, status):
    for ability, scenarios in Ability.items():
        if scenario_name in scenarios:
            Ability_Statistic[ability][1] += 1
            if status:
                Ability_Statistic[ability][0] += 1
    pass

def update_Success(scenario_name, Success_Statistic, status):
    if scenario_name not in Success_Statistic:
        if status:
            Success_Statistic[scenario_name] = [1, 1]
        else:
            Success_Statistic[scenario_name] = [0, 1]
    else:
        Success_Statistic[scenario_name][1] += 1
        if status:
            Success_Statistic[scenario_name][0] += 1
    pass

def get_position(xml_route):
    waypoints_elem = xml_route.find('waypoints')
    keypoints = waypoints_elem.findall('position')
    return [carla.Location(float(pos.get('x')), float(pos.get('y')), float(pos.get('z'))) for pos in keypoints]

def get_route_result(records, route_id):
    for record in records:
        record_route_id = record['route_id'].split('_')[1]
        if route_id == record_route_id:
            return record
    return None

def get_waypoint_route(locs, grp):
    route = []
    for i in range(len(locs) - 1):
        loc = locs[i]
        loc_next = locs[i + 1]
        interpolated_trace = grp.trace_route(loc, loc_next)
        for wp, _ in interpolated_trace:
            route.append(wp)
    return route

def get_metric(result_file, repeat_num=1):
    with open(result_file, 'r') as f:
        data = json.load(f)
    records = data["_checkpoint"]["records"]
    res = dict()
    for scene, route in scenes.items():
        res[scene] = dict(cnt=0, ids=[], ds=0, success=0, col_veh=0, col_ped=0, out_route=0, route_timeout=0, scenario_timeouts=0)
    for record in records:
        route_id = int(record["route_id"].split("_")[1])
        for scene, route in scenes.items():
            if route_id in route:
                res[scene]["cnt"] += 1
                res[scene]["ids"].append(route_id)
                res[scene]["ds"] += record["scores"]["score_composed"]
                if (record["status"] == 'Completed' or record["status"] == "Perfect") and not get_infraction_status(record):
                    res[scene]["success"] += 1
                if len(record["infractions"]["collisions_vehicle"]) > 0:
                    res[scene]["col_veh"] += 1
                if len(record["infractions"]["collisions_pedestrian"]) > 0:
                    res[scene]["col_ped"] += 1
                if len(record["infractions"]["outside_route_lanes"]) > 0:
                    res[scene]["out_route"] += 1
                if len(record["infractions"]["route_timeout"]) > 0:
                    res[scene]["route_timeout"] += 1
                if len(record["infractions"]["scenario_timeouts"]) > 0:
                    res[scene]["scenario_timeouts"] += 1

    for scene in res.keys():
        res[scene]["ds"] /= max(res[scene]["cnt"], 1)
        res[scene]["success"] /= max(res[scene]["cnt"], 1)
        res[scene]["col_veh"] /= max(res[scene]["cnt"], 1)
        res[scene]["col_ped"] /= max(res[scene]["cnt"], 1)
        res[scene]["out_route"] /= max(res[scene]["cnt"], 1)
        res[scene]["route_timeout"] /= max(res[scene]["cnt"], 1)
        res[scene]["scenario_timeouts"] /= max(res[scene]["cnt"], 1)

    if repeat_num > 1:
        repeat_stat = dict()
        for record in records:
            record_id = int(record["route_id"].split("_")[1])
            if record_id not in repeat_stat:
                repeat_stat[record_id] = [0, 1] ## succ / cnt
            else:
                repeat_stat[record_id][1] += 1
            if (record["status"] == 'Completed' or record["status"] == "Perfect") and not get_infraction_status(record):
                repeat_stat[record_id][0] += 1
        
        repeat_succ = dict()
        for i in range(repeat_num + 1):
            repeat_succ[f"sucess_{i}"] = 0
            # repeat_succ[f"sucess_{i}_ids"] = []
        for key, value in repeat_stat.items():
            assert value[1] == repeat_num
            success_num = value[0]
            repeat_succ[f"sucess_{success_num}"] += 1
            # repeat_succ[f"sucess_{success_num}_ids"].append(key)

        for i in range(repeat_num + 1):
            repeat_succ[f"sucess_{i}_rate"] = repeat_succ[f"sucess_{i}"] / (len(records) / repeat_num)
        res.update(repeat_succ)
    return res

if __name__=='__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-r', '--result_file', nargs=None, default="close_loop_log/result/merge.json", help='result json file')
    argparser.add_argument('-o', '--out_dir', nargs=None, default="close_loop_log/result")
    args = argparser.parse_args()
    res = get_metric(args.result_file)
    with open(f"{args.out_dir}/custom_metric.json", 'w') as file:
        json.dump(res, file, indent=4)
        
    