from envs.env import BasicEnv, BasicMultiEnv
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict, MultiDiscrete, Tuple
from ray.rllib.utils.framework import try_import_tf

_, tf, _ = try_import_tf()


class DramaEnv(BasicEnv):

    def __init__(self, scenario, sumo_config, control_config, train_config):
        super().__init__(scenario, sumo_config, control_config, train_config)
        
        self.num_observed = control_config.getint('num_observed', fallback=1)
        self.speed_limit = control_config.getfloat("target_velocity", fallback=15)
        self.max_accel = control_config.getfloat('max_accel', fallback=3)
        self.max_decel = control_config.getfloat("max_decel", fallback=-3)
        self.max_decel_norm = 15  # emergence stop
        
        # process net file to get network information
        self.mapping_inc, self.num_int_lane_max, self.mapping_out, self.num_out_lane_max = self.scenario.node_mapping()
        self.max_speed = None  # get from self.sumo
        self.max_length = None  # get from self.sumo
        self.all_lane = self.scenario.specify_lanes()
        self.states_tl = self.scenario.get_phases_all_tls()

        # store immediate data
        self.veh_list_lane = {i: {} for i in self.all_lane}
        # {lane_id: {veh_id: [speed, accel, dist to junc, lane_index]}}
        self.arrived_vehs = []
        self.observed_ids = []
        self.observation_info = None

        self._action_space = Box(low=0.0, high=1.0, shape=(len(self.mapping_inc),))
        # self._action_space = MultiDiscrete([3] * len(self.mapping_inc))

        self._observation_space = Box(low=0., high=1, shape=(len(self.mapping_inc) * (self.num_int_lane_max + self.num_out_lane_max + 1),))
        # self._observation_space = Tuple([Box(low=0., high=1, shape=(self.num_int_lane_max + self.num_out_lane_max + 1,)) for _ in range(len(self.mapping_inc))])
        # print(len(self._observation_space.shape))
        # self._observation_space = Dict({
        #     tl_id: Box(
        #         low=0., 
        #         high=1, 
        #         shape=(4 * self.num_int_lane_max * self.num_observed
        #                                   + self.num_int_lane_max + self.num_out_lane_max + 1,)
        #     ) 
        #     for tl_id in self.mapping_inc.keys()
        #     })

    @property
    def action_space(self,):
        return self._action_space

    @property
    def observation_space(self,):
        return self._observation_space

    @property
    def action_space_tl(self):
        return Discrete(2)

    @property
    def action_space_cav(self):
        return Box(low=-abs(self.max_decel),
                   high=self.max_accel, shape=(1,))

    @property
    def observation_space_tl(self):
        return Box(low=0., high=1, shape=(4 * self.num_int_lane_max * self.num_observed
                                          + self.num_int_lane_max + self.num_out_lane_max + 1,))

    @property
    def observation_space_cav(self):
        return Box(low=-5, high=5, shape=(7,))

    def _get_state(self):
        obs = []

        if not self.max_speed:
            self.max_speed = self.scenario.max_speed()
            # vehicles will have a random speedFactor with a deviation of 0.1 and mean of 1.0
            # which means there will be different desired speeds in the vehicle population by default
            self.max_length = self.scenario.max_length()
        accel_norm = self.max_accel + self.max_decel_norm

        veh_ids_lane = {each: [] for each in self.veh_list_lane.keys()}
        for each_veh in self.sumo.vehicle.getIDList():
            lane_id = self.sumo.vehicle.getLaneID(each_veh)
            if lane_id in self.veh_list_lane.keys():  # ignore internal links of road network
                veh_ids_lane[self.sumo.vehicle.getLaneID(each_veh)].append(each_veh)

        max_num_veh_c = self.max_length / 7.5  # 7.5 = vehicle len + min gap
        veh_num_per_lane = {}  # {lane_id: num_veh}
        for each in self.veh_list_lane.keys():
            now_veh_id_list = veh_ids_lane[each]
            pre_veh_id_list = list(self.veh_list_lane[each].keys())
            for each_veh in pre_veh_id_list:
                if each_veh not in now_veh_id_list:
                    del self.veh_list_lane[each][each_veh]
            # update vehicles (add vehicles newly departed)
            for veh in now_veh_id_list:
                self.veh_list_lane[each].update({veh: self.update_vehicles(veh, each, accel_norm)})
            veh_num_per_lane.update({each: len(now_veh_id_list) / max_num_veh_c})

        # add observation of TL
        for tl_id in self.mapping_inc.keys():
            tl_id_num = list(self.mapping_inc.keys()).index(tl_id)
            local_inc_lanes = self.mapping_inc[tl_id]
            local_out_lanes = self.mapping_out[tl_id]

            veh_num_per_in = [veh_num_per_lane[each] for each in local_inc_lanes]
            veh_num_per_out = [veh_num_per_lane[each] for each in local_out_lanes]
            # not 4-leg intersection
            if len(local_inc_lanes) < self.num_int_lane_max:
                diff = self.num_int_lane_max - len(local_inc_lanes)
                veh_num_per_in.extend([0] * diff)
            if len(local_out_lanes) < self.num_out_lane_max:
                diff = self.num_out_lane_max - len(local_out_lanes)
                veh_num_per_out.extend([0] * diff)

            states = self.states_tl[tl_id]
            now_state = self.sumo.trafficlight.getRedYellowGreenState(tl_id)
            state_index = states.index(now_state)

            observation = np.array([round(i, 8) for i in np.concatenate(
                [veh_num_per_in, veh_num_per_out, [state_index / len(states)]]
            )])

            # EMRAN changed from dict {tl_id:observation}
            obs.extend(observation)            


        if self.cast_obs:
            obs = tf.cast(obs, tf.float32)
        self.observation_info = obs

        return obs

    def _compute_reward(self):
        reward = 0.0
        for each_tl in range(len(self.states_tl)):
            obs = list(self.observation_info)
            num_vehicle_start_index = each_tl * (self.num_int_lane_max + self.num_int_lane_max + 1)
            in_traffic_sum = np.sum(obs[num_vehicle_start_index:
                                        num_vehicle_start_index + self.num_int_lane_max])
            out_traffic_sum = np.sum(obs[num_vehicle_start_index + self.num_int_lane_max:
                                         num_vehicle_start_index + self.num_int_lane_max + self.num_out_lane_max])
            reward -= in_traffic_sum - out_traffic_sum # EMRAN not sure if this will work but sure look 
                                                       # EMRAN not divided by capacity but cotv code must do it somewhere, I guess?
       
        return reward

    def _compute_dones(self): # EMRAN changed to just output bool
        # termination conditions for the environment
        done = {}
        if self.step_count_in_episode >= self.sim_step * (self.num_steps + self.warmup_steps):
            # done['__all__'] = True
            done = True
        else:
            # done['__all__'] = False
            done = False

        # arrived_vehs_this_timestep = []
        # for each in self.sumo.simulation.getArrivedIDList():
        #     if each not in self.arrived_vehs:
        #         arrived_vehs_this_timestep.append(each)
        #         done.update({each: True})
        # self.arrived_vehs.extend(arrived_vehs_this_timestep)

        return done

    def _get_info(self):
        return {}

    def _apply_actions(self, actions):
        for (agent_id, states), action in zip(self.states_tl.items(), actions):
            # perform signal switching for traffic light controller
            switch = action > 0 # EMRAN revise how actions are used
            now_state = self.sumo.trafficlight.getRedYellowGreenState(agent_id)
            state_index = states.index(now_state)
            if switch and 'G' in now_state:
                self.sumo.trafficlight.setPhase(agent_id, state_index + 1)

    # ---- Specific functions used in this env for traffic control ----
    def update_vehicles(self, veh_id, lane_id, accel_norm):
        """
        The status of one vehicle is modified at the current timestep.

        Returns
        -------
        list include speed, acceleration, distance to the intersection
        """
        speed = (self.sumo.vehicle.getSpeed(veh_id) / self.max_speed) \
            if (self.sumo.vehicle.getSpeed(veh_id) <= self.max_speed) else 1
        accel = self.sumo.vehicle.getAcceleration(veh_id)
        dist_to_junc = (self.sumo.lane.getLength(lane_id) - self.sumo.vehicle.getLanePosition(veh_id)) / self.max_length
        lane_index = self.convert_lane_into_num(lane_id) / len(self.all_lane)

        # accel normalization
        accel = accel if accel >= -15 else -15
        accel = (accel + 15) / accel_norm

        return [speed, accel, dist_to_junc, lane_index]

    def convert_lane_into_num(self, lane_id):
        return self.all_lane.index(lane_id) + 1

    def avg_speed_diff(self, veh_ids, lane_id):
        speeds = [self.sumo.vehicle.getSpeed(each_veh) for each_veh in veh_ids]
        speed_limit = self.sumo.lane.getMaxSpeed(lane_id)

        speed_diff = sum(abs(speed_limit - speed) for speed in speeds)

        return speed_diff / (speed_limit * len(speeds))

    def stable_acceleration_positive(self, veh_ids):
        accels = []  # after normalization by 9, not max_accel for CAV due to involving all vehicles
        for veh in veh_ids:
            accel = self.sumo.vehicle.getAcceleration(veh)
            if accel >= 0:  # only positive
                accels.append(accel / 9 if accel <= 9 else 1)
            else:
                pass
        accels = np.linalg.norm(np.array(accels))
        base = np.linalg.norm(np.array([1] * len(veh_ids)))

        return accels / base

    # ----

    def reset(self, **kwargs):
        """ Append some specific variables in this env to reset between episodes

        Returns
        -------
        observation: defined in BasicEnv
        """
        obs = super().reset()
        self.observation_info = {}
        self.veh_list_lane = {i: {} for i in self.all_lane}
        self.observed_ids = []
        return obs, {}

    def _additional_command(self):
        for veh_id in self.observed_ids:
            self.sumo.vehicle.highlight(veh_id, alphaMax=255, duration=1)
