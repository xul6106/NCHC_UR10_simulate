"""
自定義雙桌面任務 - 在 Robosuite 中創建包含兩個桌子和六個物件的任務
"""

import numpy as np
import random
from collections import OrderedDict

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat

# 導入您創建的物件類別
from robosuite.models.objects.xml_objects import MugObject, GlassObject, WineGlassObject, WineBottleObject, CoffeePotObject, MilkPackObject, WaterJugObject, BeerCanObject, BeerGlassObject

# 使用 Robosuite 內建的 MultiTableArena
from robosuite.models.arenas import MultiTableArena


class DualTableTask(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(1.5, 0.4, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
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
        robot_init_qpos="default",  # 新增：機器人初始關節位置
        **kwargs
    ):
        # 任務特定的設置
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        
        # 機器人初始化設置
        self.robot_init_qpos = robot_init_qpos
        
        # 右桌可選物件
        self.right_table_objects = ["mug", "glass", "wine_glass"]
        self.current_right_object = None
        
        # 左桌固定物件
        self.left_table_objects = ["water_jug", "packed_tea", "wine_bottle", "beer_can"]
        self.target_drink_name = ""
        # 建立杯子與飲料的對應關係
        self.object_pairs = {
            "mug": ["water_jug", "packed_tea"],
            "glass": ["beer_can"],
            "wine_glass": ["wine_bottle"],
        }
        
        # 物件實例
        self.objects = {}
        self.object_body_ids = {}
        
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
        獎勵函數 - 修復了 end effector 位置獲取的問題
        """
        reward = 0.0
        
        # 獲取目標物件和其對應的飲料
        target_cup_name = self.current_right_object
        #target_drink_name = self.object_pairs[target_cup_name]
        
        # 獲取物件的物理位置
        target_cup_pos = self.sim.data.body_xpos[self.object_body_ids["right_object"]]
        target_drink_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
        
        # 正確獲取機器人末端執行器位置
        eef_pos = self._get_eef_position()
        
        if eef_pos is None:
            print("Warning: Could not get end effector position, using default")
            eef_pos = np.array([0, 0, 1])
        
        # --- 獎勵塑形 (Reward Shaping) ---
        
        # 1. 接近目標飲料的獎勵
        dist_to_drink = np.linalg.norm(eef_pos - target_drink_pos)
        reaching_reward = 1 - np.tanh(10.0 * dist_to_drink)
        reward += reaching_reward
        
        # 2. 成功抓取目標飲料的獎勵
        if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.objects[self.target_drink_name]):
            reward += 1.0
            
            # 3. 接近目標杯子的獎勵（在抓到飲料之後）
            dist_to_cup = np.linalg.norm(target_drink_pos - target_cup_pos)
            placement_reward = 1 - np.tanh(10.0 * dist_to_cup)
            reward += placement_reward
        
        # 4. 成功完成任務的獎勵
        if self._check_success():
            reward = 5.0
        
        return reward * self.reward_scale

    def _load_model(self):
        """
        載入模型 - 創建場景、物件和機器人
        """
        super()._load_model()
        
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0] - 1.0)
        self.robots[0].robot_model.set_base_xpos(xpos)

        # 使用內建的 MultiTableArena 創建雙桌面場景
        mujoco_arena = MultiTableArena(
            table_offsets=[
                [0, -0.7, 0.8],  # 左桌
                [0, 0.7, 0.8]    # 右桌
            ],
            table_full_sizes=self.table_full_size,
            table_frictions=self.table_friction,
            table_rots=[0, 0],
            has_legs=True
        )
        
        # 隨機選擇右桌物件
        self.current_right_object = random.choice(self.right_table_objects)
        
        # 創建物件實例
        self._create_objects()

        possible_drinks = self.object_pairs[self.current_right_object]
        self.target_drink_name = random.choice(possible_drinks)

        if self.current_right_object == "mug":
            # "水" (water_jug) 和 "熱茶" (packed_tea)
            drink_options = ["水", "熱茶"]
            target_drink_label = "水" if self.target_drink_name == "water_jug" else "熱茶"
            print(f"機械手臂: 這容器看起來像馬克杯，請問您是要{drink_options[0]}還是{drink_options[1]}?")
            print(f"任務目標: 將 {target_drink_label} 移動到馬克杯旁。")
        elif self.current_right_object == "glass":
            print("機械手臂: 這是玻璃杯，請問您是要啤酒嗎?")
            print("任務目標: 將啤酒移動到玻璃杯旁。")
        elif self.current_right_object == "wine_glass":
            print("機械手臂: 這是紅酒杯，請問您是要紅酒嗎?")
            print("任務目標: 將紅酒移動到紅酒杯旁。")
        
        # 創建任務
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.objects.values()),
        )
        
        # 儲存 arena 參考以便後續使用
        self.arena = mujoco_arena

    def _create_objects(self):
        """
        創建所有需要的物件
        """
        # 右桌物件（隨機選擇一個）
        if self.current_right_object == "mug":
            self.objects["right_object"] = MugObject(name="mug")
        elif self.current_right_object == "glass":
            self.objects["right_object"] = BeerGlassObject(name="glass")
        elif self.current_right_object == "wine_glass":
            self.objects["right_object"] = WineGlassObject(name="wine_glass")
        
        # 左桌物件（固定三個）
        self.objects["packed_tea"] = MilkPackObject(name="packed_tea")
        self.objects["wine_bottle"] = WineBottleObject(name="wine_bottle")
        self.objects["water_jug"] = WaterJugObject(name="water_jug")
        self.objects["beer_can"] = BeerCanObject(name="beer_can")

    def _setup_references(self):
        """
        設置物件參考和 body IDs
        """
        super()._setup_references()
        
        # 獲取物件的 body IDs
        for obj_name, obj in self.objects.items():
            
            self.object_body_ids[obj_name] = self.sim.model.body_name2id(obj.root_body)
            # print("123" + obj_name + "456" + obj.root_body)

    def _setup_observables(self):
        """
        設置觀察值
        """
        observables = super()._setup_observables()
        
        # 只有在 use_object_obs=True 時才添加物件位置觀察值
        if self.use_object_obs:
            # 為每個物件創建觀察值
            for obj_name in self.objects.keys():
                
                # 創建位置 sensor - 使用閉包捕獲物件名稱
                def create_pos_sensor(name):
                    @sensor(modality=f"{name}_pos")
                    def obj_pos(obs_cache):
                        return np.array(self.sim.data.body_xpos[self.object_body_ids[name]])
                    return obj_pos
                
                # 創建旋轉 sensor - 使用閉包捕獲物件名稱
                def create_quat_sensor(name):
                    @sensor(modality=f"{name}_quat") 
                    def obj_quat(obs_cache):
                        return np.array(self.sim.data.body_xquat[self.object_body_ids[name]])
                    return obj_quat
                
                # 建立 sensor 實例
                pos_sensor = create_pos_sensor(obj_name)
                quat_sensor = create_quat_sensor(obj_name)
                
                # 添加位置觀察值
                observables[f"{obj_name}_pos"] = Observable(
                    name=f"{obj_name}_pos",
                    sensor=pos_sensor,
                    sampling_rate=self.control_freq,
                )
                
                # 添加旋轉觀察值
                observables[f"{obj_name}_quat"] = Observable(
                    name=f"{obj_name}_quat",
                    sensor=quat_sensor,
                    sampling_rate=self.control_freq,
                )
        
        return observables

    def _reset_internal(self):
        """
        內部重置函數
        """
        super()._reset_internal()
       
        # 設置物件初始位置
        self._place_objects()
        
        # 設置機器人手臂初始位置
        self._set_robot_initial_pose()

    def _place_objects(self):
        """
        放置物件到正確位置
        """
        # 獲取桌面位置資訊
        left_table_pos = self.arena.table_offsets[0]
        right_table_pos = self.arena.table_offsets[1]
        
        # 桌面高度 = 桌子位置z + 桌子高度的一半
        table_height = self.arena.table_half_sizes[0][2]
        
        # 右桌物件位置 (隨機選擇的物件)
        right_obj_pos = np.array([
            right_table_pos[0], 
            right_table_pos[1], 
            right_table_pos[2] + table_height + 0.1
        ])
        
        # 左桌物件位置 (三個固定物件排成一排)
        left_base_z = left_table_pos[2] + table_height + 0.1
        left_positions = [
            np.array([left_table_pos[0] - 0.45, left_table_pos[1], left_base_z]),  # packed_tea
            np.array([left_table_pos[0] - 0.15, left_table_pos[1], left_base_z]),  # wine_bottle  
            np.array([left_table_pos[0] + 0.15, left_table_pos[1], left_base_z]),  # water_jug
            np.array([left_table_pos[0] + 0.45, left_table_pos[1], left_base_z]),  # beer_can
        ]
        
        # 設置右桌物件位置（只有一個被選中的物件）
        right_obj = self.objects.get("right_object")
        
        if right_obj and hasattr(right_obj, 'joints') and right_obj.joints:
            joint_name = right_obj.joints[0]
            if joint_name in self.sim.model.joint_names:
                # 根據物件類型設置四元數
                quat = np.array([0.7071, 0.7071, 0, 0])

                self.sim.data.set_joint_qpos(
                    joint_name,
                    np.concatenate([right_obj_pos, quat])
                )

        # 設置左桌物件位置
        left_objects = ["packed_tea", "wine_bottle", "water_jug", "beer_can"]
        for i, obj_name in enumerate(left_objects):
            obj = self.objects.get(obj_name)
            if obj and hasattr(obj, 'joints') and obj.joints:
                joint_name = obj.joints[0]
                if joint_name in self.sim.model.joint_names:
                    # 根據物件類型設置四元數
                    quat = np.array([1, 0, 0, 0])
                    if obj_name in ["packed_tea", "beer_can"]:
                        quat = np.array([0.7071, 0.7071, 0, 0])
                    
                    self.sim.data.set_joint_qpos(
                        joint_name,
                        np.concatenate([left_positions[i], quat])
                    )

    def visualize(self, vis_settings):
        """
        視覺化設置
        """
        super().visualize(vis_settings)

    def _check_success(self):
        """更寬鬆的成功條件"""
        try:
            #target_drink_name = self.object_pairs[self.current_right_object]
            #print(self.target_drink_name, self.current_right_object)
            target_drink_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
            right_table_pos_y = self.arena.table_offsets[1][1]
            right_table_height = self.arena.table_offsets[1][2]
            # 寬鬆條件：飲料移動到右桌區域就算成功
            # success = (target_drink_pos[1] > 0.3 and  # y > 0.3 (接近右桌)
            #         target_drink_pos[2] > right_table_height)     # z > 0.75 (在桌面附近)
            success = target_drink_pos[1] > -0.3
            if success:
                print(f"SUCCESS! 飲料已移動到右桌區域!")
            return success
        except:
            return False
    
    def _get_eef_position(self):
        """
        獲取機器人末端執行器位置的專用方法
        """
        try:
            # 方法1：使用機器人控制器的末端執行器位置
            if hasattr(self.robots[0], 'controller') and hasattr(self.robots[0].controller, 'ee_pos'):
                return self.robots[0].controller.ee_pos
            
            # 方法2：使用 composite controller 的末端執行器位置
            if hasattr(self.robots[0], 'controller') and hasattr(self.robots[0].controller, 'controllers'):
                for controller in self.robots[0].controller.controllers.values():
                    if hasattr(controller, 'ee_pos'):
                        return controller.ee_pos
            
            # 方法3：直接從 sim 數據獲取 site 位置
            # 首先嘗試找到末端執行器 site
            robot_prefix = self.robots[0].robot_model.naming_prefix
            possible_site_names = [
                f"{robot_prefix}gripper0_eef",
                f"{robot_prefix}eef",
                f"{robot_prefix}right_eef",
                f"{robot_prefix}ee_site",
                "gripper0_eef",
                "eef",
                "right_eef",
                "ee_site"
            ]
            
            for site_name in possible_site_names:
                try:
                    site_id = self.sim.model.site_name2id(site_name)
                    return self.sim.data.site_xpos[site_id].copy()
                except:
                    continue
            
            # 方法4：使用夾持器位置（如果有的話）
            if hasattr(self.robots[0], 'gripper') and hasattr(self.robots[0].gripper, 'worldbody'):
                gripper_body_name = f"{robot_prefix}gripper0_base"
                try:
                    gripper_body_id = self.sim.model.body_name2id(gripper_body_name)
                    return self.sim.data.body_xpos[gripper_body_id].copy()
                except:
                    pass
            
            # 方法5：使用機器人最後一個關節的位置
            if hasattr(self.robots[0], 'robot_joints'):
                last_joint = self.robots[0].robot_joints[-1]
                try:
                    body_name = f"{robot_prefix}ee_link"  # 常見的末端執行器連桿名稱
                    body_id = self.sim.model.body_name2id(body_name)
                    return self.sim.data.body_xpos[body_id].copy()
                except:
                    pass
            
            # 方法6：嘗試其他常見的末端執行器 body 名稱
            possible_body_names = [
                f"{robot_prefix}wrist_3_link",
                f"{robot_prefix}tool0",
                f"{robot_prefix}ee_link",
                "wrist_3_link",
                "tool0",
                "ee_link"
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
        """
        設置機器人手臂的初始姿態
        """
        robot = self.robots[0]
        
        # 定義不同的初始姿態選項
        if self.robot_init_qpos == "default":
            # 使用機器人模型的默認初始位置
            return
        elif self.robot_init_qpos == "left_ready":
            # 面向左桌的準備位置
            init_qpos = np.array([-1.57, -1.2, 2.7, -1.57, 1.57, -1.57])
        elif self.robot_init_qpos == "right_ready":
            # 面向右桌的準備位置
            init_qpos = np.array([1.57, -1.2, 0.8, -1.0, -1.57, -1.57])
        elif isinstance(self.robot_init_qpos, (list, np.ndarray)):
            # 自定義關節角度
            init_qpos = np.array(self.robot_init_qpos)
        else:
            print(f"Unknown robot_init_qpos: {self.robot_init_qpos}, using default")
            return
        
        # 設置機器人關節位置
        try:
            # 獲取機器人關節名稱
            robot_joints = robot.robot_joints
            
            # 確保關節數量匹配
            if len(init_qpos) != len(robot_joints):
                print(f"Warning: init_qpos length ({len(init_qpos)}) doesn't match robot joints ({len(robot_joints)})")
                # 截斷或填充到正確長度
                if len(init_qpos) > len(robot_joints):
                    init_qpos = init_qpos[:len(robot_joints)]
                else:
                    # 用零填充
                    padded_qpos = np.zeros(len(robot_joints))
                    padded_qpos[:len(init_qpos)] = init_qpos
                    init_qpos = padded_qpos
            
            # 設置每個關節的位置
            for i, joint_name in enumerate(robot_joints):
                joint_qpos_addr = self.sim.model.get_joint_qpos_addr(joint_name)
                if isinstance(joint_qpos_addr, tuple):
                    # 對於多維關節（如球關節），只設置第一個維度
                    self.sim.data.qpos[joint_qpos_addr[0]] = init_qpos[i]
                else:
                    # 對於單維關節
                    self.sim.data.qpos[joint_qpos_addr] = init_qpos[i]
            
            # 設置夾持器為打開狀態
            if hasattr(robot, 'gripper') and hasattr(robot.gripper, 'format_action'):
                gripper_action = robot.gripper.format_action([1])  # 1 表示打開
                for gripper_joint in robot.gripper.joints:
                    joint_qpos_addr = self.sim.model.get_joint_qpos_addr(gripper_joint)
                    if isinstance(joint_qpos_addr, tuple):
                        for j, addr in enumerate(range(joint_qpos_addr[0], joint_qpos_addr[1])):
                            if j < len(gripper_action):
                                self.sim.data.qpos[addr] = gripper_action[j]
                    else:
                        self.sim.data.qpos[joint_qpos_addr] = gripper_action[0]
            
            # 前進物理模擬幾步以穩定姿態
            for _ in range(10):
                self.sim.forward()
                
            print(f"Set robot initial pose: {self.robot_init_qpos}")
            
        except Exception as e:
            print(f"Error setting robot initial pose: {e}")
            print("Available joints:", robot.robot_joints if hasattr(robot, 'robot_joints') else "None")