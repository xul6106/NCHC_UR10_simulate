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
        success_threshold=0.3,         # 增加到 0.35，避免撞倒
        success_hold_time=150,         # 減少到 150 步，更合理
        stability_threshold=0.02,      # 新增：飲料穩定性閾值（位置變化）
        stability_check_time=20,       # 新增：檢查穩定性的時間窗口
        approach_penalty_distance=0.2, # 新增：太靠近的懲罰距離
        **kwargs
    ):
        # 任務特定的設置
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.success_threshold = success_threshold
        self.success_hold_time = success_hold_time
        
        # **新增穩定性參數**
        self.stability_threshold = stability_threshold
        self.stability_check_time = stability_check_time
        self.approach_penalty_distance = approach_penalty_distance

        # 機器人初始化設置
        self.robot_init_qpos = robot_init_qpos

        # 右桌可選物件
        self.right_table_objects = ["mug", "beer_glass", "wine_glass"]
        self.current_right_object = None

        # 左桌固定物件
        self.left_table_objects = ["water_jug", "packed_tea", "wine_bottle", "beer_can"]
        self.target_drink_name = ""
        
        # **改進的對應關係**：明確指定每次的目標
        self.object_pairs = {
            "mug": ["water_jug", "packed_tea"],
            "beer_glass": ["beer_can"],
            "wine_glass": ["wine_bottle"],
        }
        
        # **新增**：明確的任務組合，避免混淆
        self.predefined_tasks = [
            ("mug", "water_jug", "馬克杯配水壺"),
            ("mug", "packed_tea", "馬克杯配綠茶"),
            ("beer_glass", "beer_can", "玻璃杯配啤酒"),
            ("wine_glass", "wine_bottle", "紅酒杯配紅酒")
        ]

        # **新增**：成功狀態追蹤
        self.success_counter = 0
        self.is_success = False
        self.last_distance = float('inf')
        
        # **新增**：穩定性追蹤
        self.drink_position_history = []  # 記錄飲料位置歷史
        self.drink_stable = True         # 飲料是否穩定
        self.collision_penalty = 0.0     # 碰撞懲罰累積
        self.initialization_steps = 0    # 初始化步數計數器

        # **改進的穩定性參數**
        self.tilt_angle_threshold = 15.0     # 新增：傾倒角度閾值（度）
        self.height_drop_threshold = 0.08    # 新增：高度下降閾值
        self.velocity_threshold = 0.1        # 新增：速度閾值
        
        # **新增穩定性追蹤變數**
        self.drink_rotation_history = []     # 記錄飲料旋轉歷史
        self.drink_height_history = []       # 記錄飲料高度歷史
        self.initial_drink_height = None     # 初始飲料高度
        self.initial_drink_rotation = None   # 初始飲料旋轉
        self.consecutive_unstable_steps = 0  # 連續不穩定步數
        self.stability_confirmation_needed = 3  # 需要連續確認的步數

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
        改進的獎勵函數，加入穩定性檢查和碰撞懲罰
        """
        reward = 0.0
        
        # 檢查是否所有必需組件都存在
        if (self.target_drink_name not in self.object_body_ids or 
            not hasattr(self, 'sim') or self.sim is None):
            return 0.0
        
        target_drink_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
        eef_pos = self._get_eef_position()
        
        if eef_pos is None:
            return 0.0
        
        dist_to_drink = np.linalg.norm(eef_pos - target_drink_pos)
        
        # **1. 穩定性檢查和懲罰**（修正：初始化階段不懲罰）
        is_drink_stable = self._check_drink_stability()
        if not is_drink_stable and self.initialization_steps >= 30:  # 只在非初始化階段懲罰
            reward -= 50.0  # 嚴重懲罰倒飲料
            print(f"⚠️  警告：{self._get_drink_display_name()}不穩定！獲得 -50 懲罰")
        
        # **2. 碰撞懲罰**（太靠近）
        if dist_to_drink < self.approach_penalty_distance:
            collision_penalty = (self.approach_penalty_distance - dist_to_drink) * 20.0
            reward -= collision_penalty
            self.collision_penalty += collision_penalty
            
            if dist_to_drink < 0.1:  # 非常接近
                reward -= 30.0
                print(f"⚠️  警告：機械手臂太靠近{self._get_drink_display_name()}！距離: {dist_to_drink:.3f}m")
        
        # **3. 安全距離內的基礎獎勵**
        if dist_to_drink >= self.approach_penalty_distance:
            # 距離獎勵 - 鼓勵接近但不要太近
            target_distance = (self.success_threshold + self.approach_penalty_distance) / 2
            if dist_to_drink > self.success_threshold:
                # 還沒到成功距離，鼓勵接近目標距離
                distance_reward = 3.0 * np.exp(-2.0 * abs(dist_to_drink - target_distance))
            else:
                # 在成功距離內，給予穩定獎勵
                distance_reward = 5.0 * np.exp(-1.0 * (dist_to_drink - self.success_threshold))
            reward += distance_reward
        
        # **4. 階段性獎勵**（更保守的距離）
        if dist_to_drink < 0.8:
            reward += 1.0
        if dist_to_drink < 0.6:
            reward += 2.0
        if dist_to_drink < 0.5:
            reward += 3.0
        if dist_to_drink < 0.4:
            reward += 4.0
        if dist_to_drink >= self.approach_penalty_distance and dist_to_drink < self.success_threshold:
            reward += 8.0  # 在安全成功範圍內
            
        # **5. 停留獎勵**（只有在穩定時才給）
        if (dist_to_drink >= self.approach_penalty_distance and 
            dist_to_drink < self.success_threshold and is_drink_stable):
            
            # 基礎停留獎勵
            stay_bonus = min(self.success_counter * 0.1, 15.0)
            reward += stay_bonus
            
            # 成功檢查的額外獎勵
            if hasattr(self, '_is_successful_demo') and self._is_successful_demo:
                reward += 10.0
            
            # 軌跡進度獎勵
            if hasattr(self, '_trajectory_timestep') and hasattr(self, '_trajectory_length'):
                progress = self._trajectory_timestep / self._trajectory_length
                if progress > 0.7:
                    progress_bonus = min((progress - 0.7) * 33.0, 10.0)
                    reward += progress_bonus
        
        # **6. 最終成功獎勵**
        if self._check_success():
            reward += 30.0
            
        # **7. 任務清晰度獎勵**（給予明確指向正確目標的獎勵）
        if hasattr(self, '_current_task_clarity_bonus'):
            reward += self._current_task_clarity_bonus
        
        return reward * self.reward_scale

    def _get_drink_display_name(self):
        """獲取飲料的顯示名稱"""
        drink_names = {
            "water_jug": "水壺", "packed_tea": "綠茶", 
            "wine_bottle": "紅酒瓶", "beer_can": "啤酒罐"
        }
        return drink_names.get(self.target_drink_name, self.target_drink_name)

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
        
        # **改進的任務選擇**：從預定義任務中隨機選擇一個
        selected_task = random.choice(self.predefined_tasks)
        self.current_right_object, self.target_drink_name, self._task_description = selected_task
        
        # **設置任務清晰度獎勵**
        self._current_task_clarity_bonus = 2.0  # 給予明確任務的小獎勵

        # **關鍵修改**：創建所有物件，確保 geom 存在
        self._create_all_objects()

        self._display_task_info()
        
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.objects.values()),
        )
        
        self.arena = mujoco_arena

    def _display_task_info(self):
        print(f"\n{'='*60}")
        print(f"🎯 任務開始！")
        print(f"📋 當前任務：{self._task_description}")
        print(f"🥤 目標飲料：{self._get_drink_display_name()}")
        print(f"📏 成功距離閾值：{self.success_threshold:.2f} 米")
        print(f"📏 安全距離閾值：{self.approach_penalty_distance:.2f} 米")
        print(f"⏱️  需要停留時間：{self.success_hold_time} 步")
        print(f"🔧 穩定性閾值：{self.stability_threshold:.3f} 米")
        print(f"{'='*60}\n")
        
        cup_names = {
            "mug": "馬克杯", "beer_glass": "玻璃杯", "wine_glass": "紅酒杯"
        }
        cup_name = cup_names.get(self.current_right_object, self.current_right_object)
        drink_name = self._get_drink_display_name()
        
        print(f"🤖 機械手臂：檢測到{cup_name}，目標飲料是{drink_name}")
        print(f"📖 任務說明：將機械手臂移動到{drink_name}附近的安全距離，並保持穩定")
        print(f"⚠️  注意：不要太靠近飲料，避免撞倒！\n")

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
        設置觀察值 - 包含所有可能物件的狀態、成功狀態信息和穩定性信息
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
                """包含所有物件的狀態，維度固定"""
                state = []
                for obj_name in self.all_possible_objects:
                    if obj_name in self.object_body_ids:
                        pos = self.sim.data.body_xpos[self.object_body_ids[obj_name]]
                        quat = self.sim.data.body_xquat[self.object_body_ids[obj_name]]
                        state.extend(pos)
                        state.extend(quat)
                    else:
                        state.extend([0.0] * 7)
                return np.array(state)

            observables["object-state"] = Observable(
                name="object-state",
                sensor=object_state,
                sampling_rate=self.control_freq,
            )
            
            # **改進的任務嵌入**：更清晰的任務指示
            @sensor(modality="object")  
            def task_embedding(obs_cache):
                """任務嵌入，清晰編碼當前的任務組合"""
                # 為每個預定義任務創建唯一編碼
                task_encoding = {
                    ("mug", "water_jug"): [1, 0, 0, 0],
                    ("mug", "packed_tea"): [0, 1, 0, 0], 
                    ("beer_glass", "beer_can"): [0, 0, 1, 0],
                    ("wine_glass", "wine_bottle"): [0, 0, 0, 1],
                }
                
                current_combo = (self.current_right_object, self.target_drink_name)
                embedding = task_encoding.get(current_combo, [0, 0, 0, 0])
                return np.array(embedding, dtype=np.float32)

            observables["task_embedding"] = Observable(
                name="task_embedding", 
                sensor=task_embedding,
                sampling_rate=self.control_freq,
            )
            
            # **新增**：穩定性和安全狀態觀察值
            @sensor(modality="object")
            def stability_safety_state(obs_cache):
                """返回穩定性和安全相關的狀態信息"""
                eef_pos = self._get_eef_position()
                if eef_pos is None or self.target_drink_name not in self.object_body_ids:
                    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                
                target_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
                distance = np.linalg.norm(eef_pos - target_pos)
                
                return np.array([
                    distance,                                           # 當前距離
                    float(distance < self.success_threshold),           # 是否在成功範圍內
                    float(distance < self.approach_penalty_distance),   # 是否太靠近（危險）
                    float(self.success_counter),                        # 已停留時間
                    float(self.success_counter / self.success_hold_time), # 停留進度
                    float(self._check_drink_stability()),              # 飲料是否穩定
                    self.collision_penalty                             # 累積碰撞懲罰
                ])

            observables["stability_safety_state"] = Observable(
                name="stability_safety_state",
                sensor=stability_safety_state,
                sampling_rate=self.control_freq,
            )
            
            # **保留原有的成功狀態觀察值以向後兼容**
            @sensor(modality="object")
            def success_state(obs_cache):
                """返回成功相關的狀態信息"""
                eef_pos = self._get_eef_position()
                if eef_pos is None or self.target_drink_name not in self.object_body_ids:
                    return np.array([0.0, 0.0, 0.0, 0.0])
                
                target_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
                distance = np.linalg.norm(eef_pos - target_pos)
                
                return np.array([
                    distance,
                    float(distance < self.success_threshold),
                    float(self.success_counter),
                    float(self.success_counter / self.success_hold_time)
                ])

            observables["success_state"] = Observable(
                name="success_state",
                sensor=success_state,
                sampling_rate=self.control_freq,
            )
        
        return observables

    def _place_objects(self):
        """
        **改進的物件放置**：確保飲料放置更穩定，不容易被撞倒
        """
        left_table_pos = self.arena.table_offsets[0]
        right_table_pos = self.arena.table_offsets[1]
        
        table_height = self.arena.table_half_sizes[0][2]
        
        # 右桌物件位置（稍微靠後，避免機械手臂意外撞到）
        right_obj_pos = np.array([
            right_table_pos[0] + 0.1,  # 稍微向後
            right_table_pos[1], 
            right_table_pos[2] + table_height + 0.1
        ])
        
        # 隱藏位置（遠離桌子）
        hidden_pos = np.array([10.0, 10.0, -1.0])
        
        # 放置右桌物件
        for obj_name in self.right_table_objects:
            obj = self.objects.get(obj_name)
            if obj and hasattr(obj, 'joints') and obj.joints:
                joint_name = obj.joints[0]
                if joint_name in self.sim.model.joint_names:
                    if obj_name == self.current_right_object:
                        quat = np.array([0.7071, 0.7071, 0, 0])
                        pos = right_obj_pos
                    else:
                        quat = np.array([1, 0, 0, 0])
                        pos = hidden_pos
                    
                    self.sim.data.set_joint_qpos(
                        joint_name,
                        np.concatenate([pos, quat])
                    )

        # **改進的左桌物件放置**：更穩定的位置，增加間距
        left_base_z = left_table_pos[2] + table_height + 0.12  # 稍微提高
        left_positions = [
            np.array([left_table_pos[0] - 0.45, left_table_pos[1], left_base_z]),  # packed_tea
            np.array([left_table_pos[0] - 0.15, left_table_pos[1], left_base_z]),  # wine_bottle  
            np.array([left_table_pos[0] + 0.15, left_table_pos[1], left_base_z]),  # water_jug
            np.array([left_table_pos[0] + 0.45, left_table_pos[1], left_base_z]),  # beer_can
        ]
        
        left_objects = ["packed_tea", "wine_bottle", "water_jug", "beer_can"]
        for i, obj_name in enumerate(left_objects):
            obj = self.objects.get(obj_name)
            if obj and hasattr(obj, 'joints') and obj.joints:
                joint_name = obj.joints[0]
                if joint_name in self.sim.model.joint_names:
                    # 確保物件直立放置
                    if obj_name in ["packed_tea", "beer_can"]:
                        quat = np.array([0.7071, 0.7071, 0, 0])
                    else:
                        quat = np.array([1, 0, 0, 0])  # 直立
                    
                    self.sim.data.set_joint_qpos(
                        joint_name,
                        np.concatenate([left_positions[i], quat])
                    )

    def _check_success(self):
        """
        改進的成功檢查：需要在安全距離內停留，且飲料必須穩定
        """
        try:
            target_drink_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
            eef_pos = self._get_eef_position()
            
            if eef_pos is None:
                return False
            
            dist_to_drink = np.linalg.norm(eef_pos - target_drink_pos)
            is_drink_stable = self._check_drink_stability()
            
            # **檢查是否在安全成功範圍內且飲料穩定**
            is_in_success_zone = (dist_to_drink >= self.approach_penalty_distance and 
                                 dist_to_drink < self.success_threshold)
            
            if is_in_success_zone and is_drink_stable:
                self.success_counter += 1
                
                # 每20步打印一次進度
                if self.success_counter % 20 == 0:
                    progress = (self.success_counter / self.success_hold_time) * 100
                    drink_name = self._get_drink_display_name()
                    print(f"✅ 安全停留進度: {progress:.1f}% ({self.success_counter}/{self.success_hold_time}) - 距離: {dist_to_drink:.3f}m")
                
                # 檢查是否停留足夠長時間
                if self.success_counter >= self.success_hold_time and not self.is_success:
                    self.is_success = True
                    drink_name = self._get_drink_display_name()
                    print(f"\n🎉 SUCCESS! 機械手臂已成功在{drink_name}前方安全停留 {self.success_hold_time} 步!")
                    print(f"✅ 最終距離: {dist_to_drink:.3f} 米 (安全範圍: {self.approach_penalty_distance:.3f} - {self.success_threshold:.3f} 米)")
                    print(f"✅ 飲料狀態: 穩定")
                    print(f"✅ 累積碰撞懲罰: {self.collision_penalty:.2f}")
                    return True
            else:
                # 離開成功範圍或飲料不穩定，重置計數器
                if self.success_counter > 0:
                    reasons = []
                    if not is_in_success_zone:
                        if dist_to_drink < self.approach_penalty_distance:
                            reasons.append(f"太靠近 (距離: {dist_to_drink:.3f}m < {self.approach_penalty_distance:.3f}m)")
                        elif dist_to_drink >= self.success_threshold:
                            reasons.append(f"太遠 (距離: {dist_to_drink:.3f}m >= {self.success_threshold:.3f}m)")
                    if not is_drink_stable:
                        reasons.append("飲料不穩定")
                    
                    reason_text = " & ".join(reasons)
                    print(f"⚠️  離開安全區域，重置停留計數器 (之前: {self.success_counter} 步) - 原因: {reason_text}")
                
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
                
            print(f"✅ 設定機械手臂初始姿勢: {self.robot_init_qpos}")
            
        except Exception as e:
            print(f"Error setting robot initial pose: {e}")
            print("Available joints:", robot.robot_joints if hasattr(robot, 'robot_joints') else "None")

    def get_current_task_info(self):
        """
        獲取當前任務信息，包括成功狀態和穩定性
        """
        cup_names = { "mug": "馬克杯", "beer_glass": "玻璃杯", "wine_glass": "紅酒杯" }
        drink_names = { "water_jug": "水壺", "packed_tea": "綠茶", "wine_bottle": "紅酒瓶", "beer_can": "啤酒罐" }
        
        info = {
            "current_cup": cup_names.get(self.current_right_object, self.current_right_object),
            "target_drink": drink_names.get(self.target_drink_name, self.target_drink_name),
            "task_description": getattr(self, '_task_description', ''),
            "success_threshold": self.success_threshold,
            "success_hold_time": self.success_hold_time,
            "approach_penalty_distance": self.approach_penalty_distance,
            "current_success_counter": self.success_counter,
            "is_success": self.is_success,
            "drink_stable": self.drink_stable,
            "collision_penalty": self.collision_penalty,
            "stability_threshold": self.stability_threshold
        }
        
        if hasattr(self, 'sim') and self.sim is not None:
            try:
                eef_pos = self._get_eef_position()
                target_drink_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
                if eef_pos is not None:
                    current_distance = np.linalg.norm(eef_pos - target_drink_pos)
                    info["current_distance"] = current_distance
                    info["distance_to_success"] = max(0, current_distance - self.success_threshold)
                    info["in_success_range"] = (current_distance >= self.approach_penalty_distance and 
                                              current_distance < self.success_threshold)
                    info["too_close"] = current_distance < self.approach_penalty_distance
                    info["success_progress"] = min(self.success_counter / self.success_hold_time, 1.0)
            except:
                pass
        return info

    def get_state(self):
        """
        獲取完整的環境狀態，包括穩定性追蹤狀態
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
            "task_description": getattr(self, '_task_description', ''),
            "success_counter": self.success_counter,
            "is_success": self.is_success,
            "last_distance": self.last_distance,
            "drink_position_history": self.drink_position_history.copy(),
            "drink_stable": self.drink_stable,
            "collision_penalty": self.collision_penalty
        }

    def reset_to(self, state):
        """
        恢復到指定狀態，包括穩定性追蹤狀態
        """
        ret = self.reset()
        if state is not None:
            try:
                if "current_right_object" in state:
                    self.current_right_object = state["current_right_object"]
                if "target_drink_name" in state:
                    self.target_drink_name = state["target_drink_name"]
                if "task_description" in state:
                    self._task_description = state["task_description"]
                if "success_counter" in state:
                    self.success_counter = state["success_counter"]
                if "is_success" in state:
                    self.is_success = state["is_success"]
                if "last_distance" in state:
                    self.last_distance = state["last_distance"]
                if "drink_position_history" in state:
                    self.drink_position_history = state["drink_position_history"]
                if "drink_stable" in state:
                    self.drink_stable = state["drink_stable"]
                if "collision_penalty" in state:
                    self.collision_penalty = state["collision_penalty"]
                if "initialization_steps" in state:
                    self.initialization_steps = state["initialization_steps"]
                else:
                    self.initialization_steps = 0
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
        # **增加初始化步數計數器**
        if hasattr(self, 'initialization_steps'):
            self.initialization_steps += 1
        else:
            self.initialization_steps = 1
            
        obs, reward, done, info = super().step(action)
        
        # 添加成功相關信息到 info
        task_info = self.get_current_task_info()
        info.update(task_info)
        
        # 每50步打印一次狀態（用於調試）
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
            
        if self._step_count % 50 == 0 and 'current_distance' in task_info:
            status_icons = []
            if task_info.get('in_success_range', False):
                status_icons.append("✅安全區")
            if task_info.get('too_close', False):
                status_icons.append("⚠️太近")
            if not task_info.get('drink_stable', True):
                status_icons.append("🚨不穩定")
            
            status = " ".join(status_icons) if status_icons else "🔄尋找中"
            
            print(f"步驟 {self._step_count}: 距離={task_info['current_distance']:.3f}m, "
                  f"停留={self.success_counter}步, {status}")
        
        return obs, reward, done, info

    def visualize(self, vis_settings):
        """
        可視化設定
        """
        super().visualize(vis_settings)

    def get_safety_analysis(self):
        """
        **新增**：獲取安全分析報告
        """
        if not hasattr(self, 'sim') or self.sim is None:
            return {"error": "Simulation not initialized"}
        
        try:
            eef_pos = self._get_eef_position()
            target_drink_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
            
            if eef_pos is None:
                return {"error": "Could not get end effector position"}
            
            current_distance = np.linalg.norm(eef_pos - target_drink_pos)
            
            analysis = {
                "current_distance": current_distance,
                "safety_status": {
                    "too_close": current_distance < self.approach_penalty_distance,
                    "safe_zone": (current_distance >= self.approach_penalty_distance and 
                                 current_distance < self.success_threshold),
                    "too_far": current_distance >= self.success_threshold
                },
                "drink_stability": {
                    "is_stable": self._check_drink_stability(),
                    "position_history_length": len(self.drink_position_history),
                    "stability_threshold": self.stability_threshold
                },
                "success_status": {
                    "counter": self.success_counter,
                    "required_time": self.success_hold_time,
                    "progress": self.success_counter / self.success_hold_time,
                    "completed": self.is_success
                },
                "penalties": {
                    "collision_penalty": self.collision_penalty
                }
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def get_task_variants(self):
        """
        **新增**：獲取所有可能的任務變體
        """
        return {
            "predefined_tasks": self.predefined_tasks,
            "current_task": (self.current_right_object, self.target_drink_name, 
                           getattr(self, '_task_description', '')),
            "object_pairs": self.object_pairs,
            "safety_parameters": {
                "success_threshold": self.success_threshold,
                "approach_penalty_distance": self.approach_penalty_distance,
                "stability_threshold": self.stability_threshold,
                "success_hold_time": self.success_hold_time,
                "stability_check_time": self.stability_check_time
            }
        }
    



# =====================================================




    def _check_drink_stability(self):
        """
        **全面改進的穩定性檢測**：結合位置、角度、高度、速度多重指標
        """
        try:
            if self.target_drink_name not in self.object_body_ids:
                return True
                
            # 獲取當前狀態
            current_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]].copy()
            current_quat = self.sim.data.body_xquat[self.object_body_ids[self.target_drink_name]].copy()
            current_height = current_pos[2]
            
            # **初始化階段：記錄初始狀態**
            if self.initialization_steps < 30:
                self.drink_position_history.append(current_pos)
                self.drink_rotation_history.append(current_quat)
                self.drink_height_history.append(current_height)
                
                if self.initialization_steps == 25:  # 在第25步設定初始參考值
                    self.initial_drink_height = np.mean([h for h in self.drink_height_history[-10:]])
                    self.initial_drink_rotation = current_quat.copy()
                    print(f"設定初始參考值 - 高度: {self.initial_drink_height:.3f}, 旋轉: {self.initial_drink_rotation}")
                
                # 限制歷史記錄長度
                self._limit_history_length()
                return True
            
            # **記錄當前狀態到歷史**
            self.drink_position_history.append(current_pos)
            self.drink_rotation_history.append(current_quat)
            self.drink_height_history.append(current_height)
            self._limit_history_length()
            
            # **如果歷史記錄不足，假設穩定**
            if len(self.drink_position_history) < 5:
                return True
            
            # **多重穩定性檢查指標**
            instability_reasons = []
            
            # 1. **位置變化檢查**（改進版）
            recent_positions = np.array(self.drink_position_history[-5:])
            position_changes = np.linalg.norm(np.diff(recent_positions, axis=0), axis=1)
            max_position_change = np.max(position_changes)
            avg_position_change = np.mean(position_changes)
            
            if max_position_change > self.stability_threshold or avg_position_change > self.stability_threshold * 0.6:
                instability_reasons.append(f"位置劇烈變化 (最大: {max_position_change:.4f}, 平均: {avg_position_change:.4f})")
            
            # 2. **高度變化檢查**（新增）
            if self.initial_drink_height is not None:
                height_drop = self.initial_drink_height - current_height
                if height_drop > self.height_drop_threshold:
                    instability_reasons.append(f"高度大幅下降 ({height_drop:.4f}m)")
            
            # 3. **角度傾斜檢查**（新增 - 關鍵改進）
            if self.initial_drink_rotation is not None:
                tilt_angle = self._calculate_tilt_angle(current_quat, self.initial_drink_rotation)
                if tilt_angle > self.tilt_angle_threshold:
                    instability_reasons.append(f"傾斜角度過大 ({tilt_angle:.1f}度)")
            
            # 4. **速度檢查**（新增）
            if len(self.drink_position_history) >= 3:
                recent_velocities = []
                for i in range(len(recent_positions) - 1):
                    dt = 1.0 / self.control_freq  # 時間間隔
                    velocity = np.linalg.norm(recent_positions[i+1] - recent_positions[i]) / dt
                    recent_velocities.append(velocity)
                
                max_velocity = np.max(recent_velocities) if recent_velocities else 0.0
                if max_velocity > self.velocity_threshold:
                    instability_reasons.append(f"移動速度過快 ({max_velocity:.4f}m/s)")
            
            # **穩定性判斷邏輯**
            is_currently_unstable = len(instability_reasons) > 0
            
            if is_currently_unstable:
                self.consecutive_unstable_steps += 1
                
                # **需要連續確認才報告不穩定**（減少誤報）
                if self.consecutive_unstable_steps >= self.stability_confirmation_needed:
                    if self.drink_stable:  # 第一次檢測到不穩定
                        drink_name = self._get_drink_display_name()
                        reasons_text = "; ".join(instability_reasons)
                        print(f"檢測到{drink_name}不穩定！原因: {reasons_text}")
                        self.drink_stable = False
                    return False
                else:
                    # 還在確認階段，暫時不報告不穩定
                    return True
            else:
                # **當前步驟穩定**
                if self.consecutive_unstable_steps > 0:
                    self.consecutive_unstable_steps = 0
                    if not self.drink_stable:
                        drink_name = self._get_drink_display_name()
                        print(f"{drink_name}重新穩定")
                        self.drink_stable = True
                return True
                
        except Exception as e:
            print(f"Error in _check_drink_stability: {e}")
            return True

    def _calculate_tilt_angle(self, current_quat, initial_quat):
        """
        **新增方法**：計算飲料相對於初始狀態的傾斜角度
        """
        try:
            # 將四元數轉換為旋轉矩陣
            from scipy.spatial.transform import Rotation
            
            current_rot = Rotation.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
            initial_rot = Rotation.from_quat([initial_quat[1], initial_quat[2], initial_quat[3], initial_quat[0]])
            
            # 計算相對旋轉
            relative_rot = current_rot * initial_rot.inv()
            
            # 獲取相對於垂直軸的傾斜角度
            euler_angles = relative_rot.as_euler('xyz', degrees=True)
            
            # 計算 X 和 Y 軸的傾斜程度（Z 軸旋轉通常不影響穩定性）
            tilt_angle = np.sqrt(euler_angles[0]**2 + euler_angles[1]**2)
            
            return tilt_angle
            
        except ImportError:
            # 如果沒有 scipy，使用簡化的四元數計算
            return self._calculate_tilt_angle_simple(current_quat, initial_quat)
        except Exception as e:
            print(f"Error calculating tilt angle: {e}")
            return 0.0

    def _calculate_tilt_angle_simple(self, current_quat, initial_quat):
        """
        **備用方法**：不依賴 scipy 的簡化傾斜角度計算
        """
        try:
            # 計算四元數差異
            # q_diff = q_current * q_initial^(-1)
            def quat_multiply(q1, q2):
                w1, x1, y1, z1 = q1
                w2, x2, y2, z2 = q2
                return np.array([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2
                ])
            
            def quat_conjugate(q):
                return np.array([q[0], -q[1], -q[2], -q[3]])
            
            # 計算相對旋轉四元數
            initial_conj = quat_conjugate(initial_quat)
            relative_quat = quat_multiply(current_quat, initial_conj)
            
            # 從四元數計算傾斜角度（簡化版）
            w, x, y, z = relative_quat
            
            # 計算相對於 Z 軸的傾斜角度
            tilt_angle = 2.0 * np.arccos(min(abs(w), 1.0)) * 180.0 / np.pi
            
            # 只考慮 X 和 Y 軸的傾斜（忽略 Z 軸旋轉）
            xy_tilt = np.sqrt(x**2 + y**2) * 2.0 * 180.0 / np.pi
            
            return min(tilt_angle, xy_tilt)
            
        except Exception as e:
            print(f"Error in simple tilt calculation: {e}")
            return 0.0

    def _limit_history_length(self):
        """
        **新增輔助方法**：限制歷史記錄的長度以節省記憶體
        """
        max_length = self.stability_check_time
        
        if len(self.drink_position_history) > max_length:
            self.drink_position_history = self.drink_position_history[-max_length:]
        
        if len(self.drink_rotation_history) > max_length:
            self.drink_rotation_history = self.drink_rotation_history[-max_length:]
            
        if len(self.drink_height_history) > max_length:
            self.drink_height_history = self.drink_height_history[-max_length:]

    def _reset_internal(self):
        """重置時也要重置新的追蹤變數"""
        super()._reset_internal()
        
        # 重置原有狀態
        self.success_counter = 0
        self.is_success = False
        self.last_distance = float('inf')
        self.drink_position_history = []
        self.drink_stable = True
        self.collision_penalty = 0.0
        self.initialization_steps = 0
        
        # **重置新增的穩定性追蹤變數**
        self.drink_rotation_history = []
        self.drink_height_history = []
        self.initial_drink_height = None
        self.initial_drink_rotation = None
        self.consecutive_unstable_steps = 0
    
        self._place_objects()
        self._set_robot_initial_pose()

    def get_stability_debug_info(self):
        """
        **新增調試方法**：獲取詳細的穩定性資訊
        """
        if not hasattr(self, 'sim') or self.sim is None:
            return {"error": "Simulation not initialized"}
        
        try:
            current_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]].copy()
            current_quat = self.sim.data.body_xquat[self.object_body_ids[self.target_drink_name]].copy()
            current_height = current_pos[2]
            
            debug_info = {
                "current_position": current_pos.tolist(),
                "current_height": current_height,
                "initial_height": self.initial_drink_height,
                "height_drop": (self.initial_drink_height - current_height) if self.initial_drink_height else 0.0,
                "position_history_length": len(self.drink_position_history),
                "rotation_history_length": len(self.drink_rotation_history),
                "consecutive_unstable_steps": self.consecutive_unstable_steps,
                "is_stable": self.drink_stable,
                "initialization_steps": getattr(self, 'initialization_steps', 0)
            }
            
            # 計算最近的位置變化
            if len(self.drink_position_history) >= 2:
                recent_changes = []
                for i in range(max(0, len(self.drink_position_history) - 5), len(self.drink_position_history) - 1):
                    change = np.linalg.norm(self.drink_position_history[i+1] - self.drink_position_history[i])
                    recent_changes.append(change)
                
                debug_info["recent_position_changes"] = recent_changes
                debug_info["max_recent_change"] = max(recent_changes) if recent_changes else 0.0
                debug_info["avg_recent_change"] = np.mean(recent_changes) if recent_changes else 0.0
            
            # 計算傾斜角度
            if self.initial_drink_rotation is not None:
                tilt_angle = self._calculate_tilt_angle(current_quat, self.initial_drink_rotation)
                debug_info["tilt_angle"] = tilt_angle
                debug_info["tilt_threshold"] = self.tilt_angle_threshold
                debug_info["is_tilted"] = tilt_angle > self.tilt_angle_threshold
            
            return debug_info
            
        except Exception as e:
            return {"error": f"Debug info failed: {str(e)}"}