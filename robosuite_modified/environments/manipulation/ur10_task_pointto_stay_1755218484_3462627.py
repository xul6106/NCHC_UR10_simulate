import numpy as np
import random
from collections import OrderedDict

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import MultiTableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

# 確保你已經導入了所有需要的物件類別
from robosuite.models.objects.xml_objects import (
    MugObject, GlassObject, WineGlassObject, WineBottleObject, 
    CoffeePotObject, MilkPackObject, WaterJugObject, BeerCanObject, BeerGlassObject
)


class DualTableTask(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types=None,
        initialization_noise="default",
        table_full_size=(1.5, 0.4, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        robot_init_qpos="left_ready_up",
        success_threshold=0.25,  # 從 0.22 到 0.25，讓成功條件更容易達成
        success_hold_time=200,    # 新增：需要在目標附近停留的時間步數
        **kwargs
    ):
        # 任務特定的設置
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.success_threshold = success_threshold
        self.success_hold_time = success_hold_time  # 新增成功停留時間

        # 機器人初始化設置
        self.robot_init_qpos = robot_init_qpos

        # 右桌可選物件
        self.right_table_objects = ["mug", "beer_glass", "wine_glass"]
        self.current_right_object = None

        # 左桌固定物件
        self.left_table_objects = ["water_jug", "packed_tea", "wine_bottle", "beer_can"]
        self.target_drink_name = ""
        
        # 建立杯子與飲料的對應關係
        self.object_pairs = {
            "mug": ["water_jug", "packed_tea"],
            "beer_glass": ["beer_can"],
            "wine_glass": ["wine_bottle"],
        }

        # **新增**：成功狀態追蹤
        self.success_counter = 0  # 記錄連續在目標附近的時間步數
        self.is_success = False   # 是否已經達成成功條件
        self.last_distance = float('inf')  # 記錄上一步的距離

        # **關鍵修改**：創建所有可能的右桌物件，但只啟用一個
        self.objects = {}
        self.object_body_ids = {}

        # 固定的物件順序（包含所有可能的右桌物件）
        self.all_possible_objects = ["mug", "beer_glass", "wine_glass", "packed_tea", "wine_bottle", "water_jug", "beer_can"]
        self.object_state_dim = len(self.all_possible_objects) * 7  # 每個物件 7 維

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            **kwargs
        )

    def reward(self, action=None):
        """
        Modified reward function that works for both live training and data conversion
        """
        reward = 0.0
        
        target_drink_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
        eef_pos = self._get_eef_position()
        
        if eef_pos is None:
            return 0.0
        
        dist_to_drink = np.linalg.norm(eef_pos - target_drink_pos)
        
        # 1. Basic distance reward
        distance_reward = 2.0 * np.exp(-3.0 * dist_to_drink)
        reward += distance_reward
        
        # 2. Staged rewards (these work during conversion)
        if dist_to_drink < 0.5:
            reward += 2.0
        if dist_to_drink < 0.4:
            reward += 3.0
        if dist_to_drink < 0.3:
            reward += 4.0
        if dist_to_drink < self.success_threshold:
            reward += 8.0  # Base success reward
            
            # 3. **NEW**: Check if we're in a "successful trajectory context"
            # This works during data conversion by checking if this is a successful demo
            if hasattr(self, '_is_successful_demo') and self._is_successful_demo:
                reward += 10.0  # Additional success bonus for successful demos
            
            # 4. **NEW**: Time-based success bonus (for conversion)
            # Estimate how long we've been near target based on trajectory position
            if hasattr(self, '_trajectory_timestep') and hasattr(self, '_trajectory_length'):
                # If we're near the end of a successful trajectory and close to target
                progress = self._trajectory_timestep / self._trajectory_length
                if progress > 0.7:  # In the last 30% of trajectory
                    stay_bonus = min((progress - 0.7) * 33.0, 10.0)  # Scale to 10 max
                    reward += stay_bonus
        
        # 5. For live training, use the original counter-based logic
        if hasattr(self, 'success_counter') and self.success_counter > 0:
            stay_bonus = min(self.success_counter * 0.05, 10.0)
            reward += stay_bonus
        
        # 6. Final success reward for completed trajectories
        if self._check_success():
            reward += 20.0
        
        return reward * self.reward_scale

    def _load_model(self):
        super()._load_model()
        
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0] - 1.0)
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = MultiTableArena(
            table_offsets=[
                [0, -0.7, 0.8],
                [0, 0.7, 0.8]
            ],
            table_full_sizes=self.table_full_size,
            table_frictions=self.table_friction,
            table_rots=[0, 0],
            has_legs=True
        )
        
        # 隨機選擇右桌物件
        self.current_right_object = random.choice(self.right_table_objects)
        
        # **關鍵修改**：創建所有物件，確保 geom 存在
        self._create_all_objects()

        # 選擇目標飲料
        possible_drinks = self.object_pairs[self.current_right_object]
        self.target_drink_name = random.choice(possible_drinks)

        self._display_task_info()
        
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.objects.values()),
        )
        
        self.arena = mujoco_arena

    def _display_task_info(self):
        cup_names = {
            "mug": "馬克杯",
            "beer_glass": "玻璃杯", 
            "wine_glass": "紅酒杯"
        }
        
        drink_names = {
            "water_jug": "水壺",
            "packed_tea": "綠茶",
            "wine_bottle": "紅酒瓶",
            "beer_can": "啤酒罐"
        }
        
        cup_display_name = cup_names.get(self.current_right_object, self.current_right_object)
        drink_display_name = drink_names.get(self.target_drink_name, self.target_drink_name)
        
        print(f"\n{'='*50}")
        print(f"任務開始!")
        print(f"檢測到杯子類型: {cup_display_name}")
        print(f"目標飲料: {drink_display_name}")
        print(f"任務目標: 將機械手臂移動到 {drink_display_name} 前方並停留")
        print(f"成功距離閾值: {self.success_threshold:.2f} 米")
        print(f"需要停留時間: {self.success_hold_time} 步")
        print(f"{'='*50}\n")
        
        if self.current_right_object == "mug":
            possible_drinks = ["水壺", "綠茶"]
            target_drink_label = "水壺" if self.target_drink_name == "water_jug" else "綠茶"
            print(f"機械手臂: 檢測到馬克杯，目標飲料是 {target_drink_label}")
        elif self.current_right_object == "beer_glass":
            print("機械手臂: 檢測到玻璃杯，目標飲料是啤酒")
        elif self.current_right_object == "wine_glass":
            print("機械手臂: 檢測到紅酒杯，目標飲料是紅酒")

    def _create_all_objects(self):
        """
        **關鍵修改**：創建所有可能的物件，確保所有 geom 都存在於模型中
        但只有當前選中的右桌物件會被正確放置，其他的會被隱藏
        """
        # 創建所有可能的右桌物件
        self.objects["mug"] = MugObject(name="mug")
        self.objects["beer_glass"] = BeerGlassObject(name="beer_glass")  
        self.objects["wine_glass"] = WineGlassObject(name="wine_glass")
        
        # 創建左桌物件
        self.objects["packed_tea"] = MilkPackObject(name="packed_tea")
        self.objects["wine_bottle"] = WineBottleObject(name="wine_bottle")
        self.objects["water_jug"] = WaterJugObject(name="water_jug")
        self.objects["beer_can"] = BeerCanObject(name="beer_can")

    def _setup_references(self):
        super()._setup_references()
        
        # 為所有物件建立 body_id 映射
        for obj_name, obj in self.objects.items():
            self.object_body_ids[obj_name] = self.sim.model.body_name2id(obj.root_body)

    def _setup_observables(self):
        """
        設置觀察值 - 包含所有可能物件的狀態和成功狀態信息
        """
        observables = super()._setup_observables()
        
        if self.use_object_obs:
            # 為所有可能的物件建立觀察值
            for obj_name in self.all_possible_objects:
                def create_pos_sensor(name):
                    @sensor(modality=f"{name}_pos")
                    def obj_pos(obs_cache):
                        if name in self.object_body_ids:
                            return self.sim.data.body_xpos[self.object_body_ids[name]]
                        else:
                            return np.zeros(3)
                    return obj_pos
                
                def create_quat_sensor(name):
                    @sensor(modality=f"{name}_quat") 
                    def obj_quat(obs_cache):
                        if name in self.object_body_ids:
                            return self.sim.data.body_xquat[self.object_body_ids[name]]
                        else:
                            return np.zeros(4)
                    return obj_quat
                
                pos_sensor = create_pos_sensor(obj_name)
                quat_sensor = create_quat_sensor(obj_name)
                
                observables[f"{obj_name}_pos"] = Observable(
                    name=f"{obj_name}_pos",
                    sensor=pos_sensor,
                    sampling_rate=self.control_freq,
                )
                
                observables[f"{obj_name}_quat"] = Observable(
                    name=f"{obj_name}_quat",
                    sensor=quat_sensor,
                    sampling_rate=self.control_freq,
                )

            # 統一的 object-state 觀察值
            @sensor(modality="object")
            def object_state(obs_cache):
                """
                包含所有物件的狀態，維度固定
                """
                state = []
                
                for obj_name in self.all_possible_objects:
                    if obj_name in self.object_body_ids:
                        pos = self.sim.data.body_xpos[self.object_body_ids[obj_name]]
                        quat = self.sim.data.body_xquat[self.object_body_ids[obj_name]]
                        state.extend(pos)
                        state.extend(quat)
                    else:
                        # 物件不存在，添加零值
                        state.extend([0.0] * 7)
                
                return np.array(state)

            observables["object-state"] = Observable(
                name="object-state",
                sensor=object_state,
                sampling_rate=self.control_freq,
            )
            
            # **新增**：活躍物件指示器，告訴 agent 哪個右桌物件是活躍的
            @sensor(modality="object")
            def active_right_object(obs_cache):
                """
                one-hot 編碼，指示當前活躍的右桌物件
                """
                indicator = np.zeros(len(self.right_table_objects))
                if self.current_right_object:
                    idx = self.right_table_objects.index(self.current_right_object)
                    indicator[idx] = 1.0
                return indicator

            observables["active_right_object"] = Observable(
                name="active_right_object",
                sensor=active_right_object,
                sampling_rate=self.control_freq,
            )
            
            # **新增**：目標飲料指示器
            @sensor(modality="object")
            def target_drink_indicator(obs_cache):
                """
                one-hot 編碼，指示當前目標飲料
                """
                indicator = np.zeros(len(self.left_table_objects))
                if self.target_drink_name:
                    idx = self.left_table_objects.index(self.target_drink_name)
                    indicator[idx] = 1.0
                return indicator

            observables["target_drink_indicator"] = Observable(
                name="target_drink_indicator",
                sensor=target_drink_indicator,
                sampling_rate=self.control_freq,
            )
            
            # **新增**：任務嵌入，結合杯子和目標飲料信息
            @sensor(modality="object")  
            def task_embedding(obs_cache):
                """
                任務嵌入，編碼當前的杯子-飲料組合
                """
                # 為每種可能的組合創建唯一編碼
                task_combinations = {
                    ("mug", "water_jug"): [1, 0, 0, 0],
                    ("mug", "packed_tea"): [0, 1, 0, 0], 
                    ("beer_glass", "beer_can"): [0, 0, 1, 0],
                    ("wine_glass", "wine_bottle"): [0, 0, 0, 1],
                }
                
                current_combo = (self.current_right_object, self.target_drink_name)
                embedding = task_combinations.get(current_combo, [0, 0, 0, 0])
                return np.array(embedding, dtype=np.float32)

            observables["task_embedding"] = Observable(
                name="task_embedding", 
                sensor=task_embedding,
                sampling_rate=self.control_freq,
            )
            
            # **新增**：成功狀態觀察值
            @sensor(modality="object")
            def success_state(obs_cache):
                """
                返回成功相關的狀態信息
                """
                eef_pos = self._get_eef_position()
                if eef_pos is None or self.target_drink_name not in self.object_body_ids:
                    return np.array([0.0, 0.0, 0.0, 0.0])
                
                target_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
                distance = np.linalg.norm(eef_pos - target_pos)
                
                return np.array([
                    distance,                                    # 當前距離
                    float(distance < self.success_threshold),    # 是否在成功範圍內
                    float(self.success_counter),                 # 已停留時間
                    float(self.success_counter / self.success_hold_time)  # 停留進度
                ])

            observables["success_state"] = Observable(
                name="success_state",
                sensor=success_state,
                sampling_rate=self.control_freq,
            )
        
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        
        # **重置成功狀態追蹤**
        self.success_counter = 0
        self.is_success = False
        self.last_distance = float('inf')
       
        self._place_objects()
        
        self._set_robot_initial_pose()

    def _place_objects(self):
        """
        **關鍵修改**：只放置當前選中的右桌物件，其他右桌物件隱藏到遠處
        """
        left_table_pos = self.arena.table_offsets[0]
        right_table_pos = self.arena.table_offsets[1]
        
        table_height = self.arena.table_half_sizes[0][2]
        
        # 右桌物件位置
        right_obj_pos = np.array([
            right_table_pos[0], 
            right_table_pos[1], 
            right_table_pos[2] + table_height + 0.1
        ])
        
        # 隱藏位置（遠離桌子）
        hidden_pos = np.array([10.0, 10.0, -1.0])  # 遠離場景的位置
        
        # 放置右桌物件
        for obj_name in self.right_table_objects:
            obj = self.objects.get(obj_name)
            if obj and hasattr(obj, 'joints') and obj.joints:
                joint_name = obj.joints[0]
                if joint_name in self.sim.model.joint_names:
                    if obj_name == self.current_right_object:
                        # 當前選中的物件放在右桌上
                        quat = np.array([0.7071, 0.7071, 0, 0])
                        pos = right_obj_pos
                    else:
                        # 其他物件隱藏
                        quat = np.array([1, 0, 0, 0])
                        pos = hidden_pos
                    
                    self.sim.data.set_joint_qpos(
                        joint_name,
                        np.concatenate([pos, quat])
                    )

        # 放置左桌物件
        left_base_z = left_table_pos[2] + table_height + 0.1
        left_positions = [
            np.array([left_table_pos[0] - 0.45, left_table_pos[1], left_base_z]),
            np.array([left_table_pos[0] - 0.15, left_table_pos[1], left_base_z]),
            np.array([left_table_pos[0] + 0.15, left_table_pos[1], left_base_z]),
            np.array([left_table_pos[0] + 0.45, left_table_pos[1], left_base_z]),
        ]
        
        left_objects = ["packed_tea", "wine_bottle", "water_jug", "beer_can"]
        for i, obj_name in enumerate(left_objects):
            obj = self.objects.get(obj_name)
            if obj and hasattr(obj, 'joints') and obj.joints:
                joint_name = obj.joints[0]
                if joint_name in self.sim.model.joint_names:
                    quat = np.array([1, 0, 0, 0])
                    if obj_name in ["packed_tea", "beer_can"]:
                        quat = np.array([0.7071, 0.7071, 0, 0])
                    
                    self.sim.data.set_joint_qpos(
                        joint_name,
                        np.concatenate([left_positions[i], quat])
                    )

    def visualize(self, vis_settings):
        super().visualize(vis_settings)

    def _check_success(self):
        """
        改進的成功檢查：需要在目標附近停留一段時間
        """
        try:
            target_drink_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
            eef_pos = self._get_eef_position()
            
            if eef_pos is None:
                return False
            
            dist_to_drink = np.linalg.norm(eef_pos - target_drink_pos)
            
            # 檢查是否在成功範圍內
            if dist_to_drink < self.success_threshold:
                self.success_counter += 1
                
                # 每10步打印一次進度
                if self.success_counter % 10 == 0:
                    progress = (self.success_counter / self.success_hold_time) * 100
                    drink_names = {
                        "water_jug": "水壺", "packed_tea": "綠茶", 
                        "wine_bottle": "紅酒瓶", "beer_can": "啤酒罐"
                    }
                    drink_name = drink_names.get(self.target_drink_name, self.target_drink_name)
                    print(f"停留進度: {progress:.1f}% ({self.success_counter}/{self.success_hold_time}) - 距離: {dist_to_drink:.3f}m")
                
                # 檢查是否停留足夠長時間
                if self.success_counter >= self.success_hold_time and not self.is_success:
                    self.is_success = True
                    drink_names = {
                        "water_jug": "水壺", "packed_tea": "綠茶", 
                        "wine_bottle": "紅酒瓶", "beer_can": "啤酒罐"
                    }
                    drink_name = drink_names.get(self.target_drink_name, self.target_drink_name)
                    print(f"\n🎉 SUCCESS! 機械手臂已成功在 {drink_name} 前方停留 {self.success_hold_time} 步!")
                    print(f"最終距離: {dist_to_drink:.3f} 米 (閾值: {self.success_threshold:.3f} 米)")
                    return True
            else:
                # 離開成功範圍，重置計數器
                if self.success_counter > 0:
                    print(f"離開目標範圍，重置停留計數器 (之前: {self.success_counter} 步)")
                self.success_counter = 0
                self.is_success = False
            
            return self.is_success
            
        except Exception as e:
            print(f"Error in _check_success: {e}")
            return False
    
    def _get_eef_position(self):
        try:
            if hasattr(self.robots[0], 'controller') and hasattr(self.robots[0].controller, 'ee_pos'):
                return self.robots[0].controller.ee_pos
            
            if hasattr(self.robots[0], 'controller') and hasattr(self.robots[0].controller, 'controllers'):
                for controller in self.robots[0].controller.controllers.values():
                    if hasattr(controller, 'ee_pos'):
                        return controller.ee_pos
            
            robot_prefix = self.robots[0].robot_model.naming_prefix
            possible_site_names = [
                f"{robot_prefix}gripper0_eef", f"{robot_prefix}eef", 
                f"{robot_prefix}right_eef", f"{robot_prefix}ee_site", 
                "gripper0_eef", "eef", "right_eef", "ee_site"
            ]
            
            for site_name in possible_site_names:
                try:
                    site_id = self.sim.model.site_name2id(site_name)
                    return self.sim.data.site_xpos[site_id].copy()
                except:
                    continue
            
            if hasattr(self.robots[0], 'gripper') and hasattr(self.robots[0].gripper, 'worldbody'):
                gripper_body_name = f"{robot_prefix}gripper0_base"
                try:
                    gripper_body_id = self.sim.model.body_name2id(gripper_body_name)
                    return self.sim.data.body_xpos[gripper_body_id].copy()
                except:
                    pass
            
            if hasattr(self.robots[0], 'robot_joints'):
                last_joint = self.robots[0].robot_joints[-1]
                try:
                    body_name = f"{robot_prefix}ee_link"
                    body_id = self.sim.model.body_name2id(body_name)
                    return self.sim.data.body_xpos[body_id].copy()
                except:
                    pass
            
            possible_body_names = [
                f"{robot_prefix}wrist_3_link", f"{robot_prefix}tool0", 
                f"{robot_prefix}ee_link", "wrist_3_link", "tool0", "ee_link"
            ]
            
            for body_name in possible_body_names:
                try:
                    body_id = self.sim.model.body_name2id(body_name)
                    return self.sim.data.body_xpos[body_id].copy()
                except:
                    continue
            
            print("Warning: Could not find end effector position using any known method")
            return None
            
        except Exception as e:
            print(f"Error in _get_eef_position: {e}")
            return None
    
    def _set_robot_initial_pose(self):
        robot = self.robots[0]
        
        if self.robot_init_qpos == "default":
            return
        elif self.robot_init_qpos == "left_ready":
            init_qpos = np.array([-1.57, -1.2, 2.7, -1.57, 1.57, -1.57])
        elif self.robot_init_qpos == "left_ready_up":
            init_qpos = np.array([-1.57, -1.8, 1.7, 0, 1.57, -1.57])
        elif isinstance(self.robot_init_qpos, (list, np.ndarray)):
            init_qpos = np.array(self.robot_init_qpos)
        else:
            print(f"Unknown robot_init_qpos: {self.robot_init_qpos}, using default")
            return
        
        try:
            robot_joints = robot.robot_joints
            
            if len(init_qpos) != len(robot_joints):
                print(f"Warning: init_qpos length ({len(init_qpos)}) doesn't match robot joints ({len(robot_joints)})")
                if len(init_qpos) > len(robot_joints):
                    init_qpos = init_qpos[:len(robot_joints)]
                else:
                    padded_qpos = np.zeros(len(robot_joints))
                    padded_qpos[:len(init_qpos)] = init_qpos
                    init_qpos = padded_qpos
            
            for i, joint_name in enumerate(robot_joints):
                joint_qpos_addr = self.sim.model.get_joint_qpos_addr(joint_name)
                if isinstance(joint_qpos_addr, tuple):
                    self.sim.data.qpos[joint_qpos_addr[0]] = init_qpos[i]
                else:
                    self.sim.data.qpos[joint_qpos_addr] = init_qpos[i]
            
            if hasattr(robot, 'gripper') and hasattr(robot.gripper, 'format_action'):
                gripper_action = robot.gripper.format_action([1])
                for gripper_joint in robot.gripper.joints:
                    joint_qpos_addr = self.sim.model.get_joint_qpos_addr(gripper_joint)
                    if isinstance(joint_qpos_addr, tuple):
                        for j, addr in enumerate(range(joint_qpos_addr[0], joint_qpos_addr[1])):
                            if j < len(gripper_action):
                                self.sim.data.qpos[addr] = gripper_action[j]
                    else:
                        self.sim.data.qpos[joint_qpos_addr] = gripper_action[0]
            
            for _ in range(10):
                self.sim.forward()
                
            print(f"Set robot initial pose: {self.robot_init_qpos}")
            
        except Exception as e:
            print(f"Error setting robot initial pose: {e}")
            print("Available joints:", robot.robot_joints if hasattr(robot, 'robot_joints') else "None")

    def get_current_task_info(self):
        """
        獲取當前任務信息，包括成功狀態
        """
        cup_names = { "mug": "馬克杯", "beer_glass": "玻璃杯", "wine_glass": "紅酒杯" }
        drink_names = { "water_jug": "水壺", "packed_tea": "綠茶", "wine_bottle": "紅酒瓶", "beer_can": "啤酒罐" }
        
        info = {
            "current_cup": cup_names.get(self.current_right_object, self.current_right_object),
            "target_drink": drink_names.get(self.target_drink_name, self.target_drink_name),
            "success_threshold": self.success_threshold,
            "success_hold_time": self.success_hold_time,
            "current_success_counter": self.success_counter,
            "is_success": self.is_success
        }
        
        if hasattr(self, 'sim') and self.sim is not None:
            try:
                eef_pos = self._get_eef_position()
                target_drink_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
                if eef_pos is not None:
                    current_distance = np.linalg.norm(eef_pos - target_drink_pos)
                    info["current_distance"] = current_distance
                    info["distance_to_success"] = max(0, current_distance - self.success_threshold)
                    info["in_success_range"] = current_distance < self.success_threshold
                    info["success_progress"] = min(self.success_counter / self.success_hold_time, 1.0)
            except:
                pass
        return info

    def get_state(self):
        """
        獲取完整的環境狀態，包括成功追蹤狀態
        """
        return {
            "model_xml": self.model.get_xml(),
            "initial_qpos": self.sim.data.qpos.copy(),
            "initial_qvel": self.sim.data.qvel.copy(),
            "objects": {
                obj_name: {
                    "pos": self.sim.data.body_xpos[self.object_body_ids[obj_name]].copy(),
                    "quat": self.sim.data.body_xquat[self.object_body_ids[obj_name]].copy()
                } for obj_name in self.object_body_ids.keys()
            },
            "current_right_object": self.current_right_object,
            "target_drink_name": self.target_drink_name,
            "success_counter": self.success_counter,
            "is_success": self.is_success,
            "last_distance": self.last_distance
        }

    def reset_to(self, state):
        """
        恢復到指定狀態，包括成功追蹤狀態
        """
        ret = self.reset()
        if state is not None:
            try:
                if "current_right_object" in state:
                    self.current_right_object = state["current_right_object"]
                if "target_drink_name" in state:
                    self.target_drink_name = state["target_drink_name"]
                if "success_counter" in state:
                    self.success_counter = state["success_counter"]
                if "is_success" in state:
                    self.is_success = state["is_success"]
                if "last_distance" in state:
                    self.last_distance = state["last_distance"]
                if "initial_qpos" in state:
                    self.sim.data.qpos[:] = state["initial_qpos"]
                if "initial_qvel" in state:
                    self.sim.data.qvel[:] = state["initial_qvel"]
                
                # 重新放置物件以反映正確的配置
                self._place_objects()
                
                self.sim.forward()
            except Exception as e:
                print(f"Warning: Could not fully restore state: {e}")
        return ret

    def step(self, action):
        """
        重寫 step 方法以添加額外的成功檢查和調試信息
        """
        obs, reward, done, info = super().step(action)
        
        # 添加成功相關信息到 info
        task_info = self.get_current_task_info()
        info.update(task_info)
        
        # 每100步打印一次狀態（用於調試）
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
            
        if self._step_count % 100 == 0 and 'current_distance' in task_info:
            print(f"Step {self._step_count}: 距離={task_info['current_distance']:.3f}m, "
                  f"停留={self.success_counter}步, 成功={self.is_success}")
        
        return obs, reward, done, info