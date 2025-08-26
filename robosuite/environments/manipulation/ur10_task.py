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

    def _get_drink_display_name(self):
        """獲取飲料的顯示名稱"""
        drink_names = {
            "water_jug": "水壺", "packed_tea": "綠茶", 
            "wine_bottle": "紅酒瓶", "beer_can": "啤酒罐"
        }
        return drink_names.get(self.target_drink_name, self.target_drink_name)

    def _load_model(self):
        """改進的任務選擇邏輯"""
        super(ManipulationEnv, self)._load_model()
        
        # 設置機器人位置
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0] - 1.0)
        self.robots[0].robot_model.set_base_xpos(xpos)

        # 創建環境
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
        
        # 選擇右桌物件
        self.current_right_object = random.choice(self.right_table_objects)
        
        # 基於右桌物件選擇目標
        if self.current_right_object == "mug":
            # 馬克杯的多目標選擇邏輯
            self.target_drink_name = self._select_mug_target()
            self._task_description = f"馬克杯配{self._get_drink_display_name()}"
        else:
            # 其他物件的單目標邏輯
            possible_targets = self.enhanced_object_pairs[self.current_right_object]
            self.target_drink_name = possible_targets[0]
            cup_names = {"beer_glass": "玻璃杯", "wine_glass": "紅酒杯"}
            cup_name = cup_names.get(self.current_right_object, self.current_right_object)
            self._task_description = f"{cup_name}配{self._get_drink_display_name()}"

        self._create_all_objects()
        self._display_enhanced_task_info()
        
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.objects.values()),
        )
        
        self.arena = mujoco_arena

    def _select_mug_target(self):
        """馬克杯目標選擇邏輯"""
        possible_targets = ["water_jug", "packed_tea"]
        
        if self.mug_target_selection_mode == "random":
            return random.choice(possible_targets)
        elif self.mug_target_selection_mode == "preference":
            # 可以基於某種偏好邏輯，比如訓練進度
            return random.choices(possible_targets, weights=[0.6, 0.4])[0]  # 稍微偏向水壺
        else:
            return random.choice(possible_targets)

    def _display_enhanced_task_info(self):
        """增強的任務信息顯示"""
        print(f"\n{'='*70}")
        print(f"🎯 多目標任務開始！")
        print(f"📋 當前任務：{self._task_description}")
        
        if self.current_right_object == "mug":
            print(f"🤔 決策挑戰：馬克杯可以配水壺或綠茶，本輪選擇：{self._get_drink_display_name()}")
            print(f"📚 學習目標：機器人需要學會在多個合理選項中做出選擇")
        
        print(f"🥤 目標飲料：{self._get_drink_display_name()}")
        
        # 顯示當前物件的特定參數
        params = self._get_object_params(self.target_drink_name)
        print(f"📏 成功距離：{params['success_threshold']:.2f} 米")
        print(f"📏 安全距離：{params['approach_penalty']:.2f} 米")
        print(f"⚖️  碰撞敏感度：{params['collision_sensitivity']:.1f}x")
        print(f"⏱️  停留時間：{self.success_hold_time} 步")
        print(f"{'='*70}\n")

    def reward(self, action=None):
        """針對多目標任務優化的獎勵函數"""
        reward = 0.0
        
        if (self.target_drink_name not in self.object_body_ids or 
            not hasattr(self, 'sim') or self.sim is None):
            return 0.0
        
        # 獲取物件特定參數
        params = self._get_object_params(self.target_drink_name)
        success_thresh = params["success_threshold"]
        penalty_dist = params["approach_penalty"]
        collision_sens = params["collision_sensitivity"]
        
        target_drink_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]]
        eef_pos = self._get_eef_position()
        
        if eef_pos is None:
            return 0.0
        
        dist_to_drink = np.linalg.norm(eef_pos - target_drink_pos)
        
        # 1. 選擇明確性獎勵（針對馬克杯任務）
        if self.current_right_object == "mug":
            choice_clarity_reward = self._calculate_choice_clarity_reward()
            reward += choice_clarity_reward

        # 2. 自適應穩定性檢查
        is_drink_stable = self._check_drink_stability_enhanced(params)
        if not is_drink_stable and self.initialization_steps >= 30:
            # 基於碰撞敏感度調整懲罰
            instability_penalty = 25.0 * collision_sens
            reward -= instability_penalty
            
            if collision_sens > 1.2:  # 對於易碎物件
                print(f"⚠️ {self._get_drink_display_name()}不穩定！敏感度: {collision_sens:.1f}x")

        # 3. 距離基礎獎勵（考慮物件特性）
        if dist_to_drink >= penalty_dist:
            # 安全區域內的接近獎勵
            if dist_to_drink > success_thresh:
                # 鼓勵接近，但考慮物件敏感度
                safe_approach_factor = 2.0 - (collision_sens - 1.0) * 0.5
                approach_reward = 6.0 * safe_approach_factor * np.exp(-0.8 * (dist_to_drink - success_thresh))
                reward += approach_reward
            else:
                # 成功區域內的停留獎勵
                stay_factor = 1.0 + (2.0 - collision_sens) * 0.3  # 穩定物件給更多獎勵
                stay_reward = 12.0 * stay_factor * (1.0 - (dist_to_drink - penalty_dist) / (success_thresh - penalty_dist))
                reward += stay_reward
        else:
            # 危險區域的適應性懲罰
            danger_level = (penalty_dist - dist_to_drink) / penalty_dist
            adaptive_penalty = 8.0 * collision_sens * danger_level
            reward -= adaptive_penalty
            
            # 非常接近時的嚴重警告
            if dist_to_drink < penalty_dist * 0.7:
                severe_penalty = 15.0 * collision_sens
                reward -= severe_penalty

        # 4. 成功停留獎勵
        if (penalty_dist <= dist_to_drink < success_thresh and is_drink_stable):
            base_stay_reward = min(self.success_counter * 0.15, 18.0)
            stability_bonus = (2.0 - collision_sens) * 0.1  # 穩定物件額外獎勵
            reward += base_stay_reward * (1.0 + stability_bonus)
            
            if self._check_success():
                completion_reward = 40.0 + 10.0 * (2.0 - collision_sens)  # 困難物件完成獎勵更高
                reward += completion_reward

        return reward * self.reward_scale

    def _calculate_choice_clarity_reward(self):
        """計算選擇明確性獎勵（針對馬克杯多目標任務）"""
        if not hasattr(self, 'sim') or self.sim is None:
            return 0.0
            
        try:
            eef_pos = self._get_eef_position()
            if eef_pos is None:
                return 0.0
            
            # 計算到兩個可能目標的距離
            water_pos = self.sim.data.body_xpos[self.object_body_ids["water_jug"]]
            tea_pos = self.sim.data.body_xpos[self.object_body_ids["packed_tea"]]
            
            dist_to_water = np.linalg.norm(eef_pos - water_pos)
            dist_to_tea = np.linalg.norm(eef_pos - tea_pos)
            
            # 獎勵明確的選擇行為
            if self.target_drink_name == "water_jug":
                # 如果目標是水壺，獎勵更靠近水壺而遠離茶
                if dist_to_water < dist_to_tea:
                    clarity_reward = 2.0 * (dist_to_tea - dist_to_water) / (dist_to_tea + dist_to_water)
                    return min(clarity_reward, 3.0)
            elif self.target_drink_name == "packed_tea":
                # 如果目標是茶，獎勵更靠近茶而遠離水壺
                if dist_to_tea < dist_to_water:
                    clarity_reward = 2.0 * (dist_to_water - dist_to_tea) / (dist_to_water + dist_to_tea)
                    return min(clarity_reward, 3.0)
            
            return 0.0
            
        except Exception as e:
            return 0.0

    def _get_object_params(self, object_name):
        """獲取物件特定參數"""
        return self.object_specific_params.get(
            object_name, 
            {
                "success_threshold": 0.3,
                "approach_penalty": 0.2,
                "stability_threshold": 0.02,
                "collision_sensitivity": 1.0
            }
        )

    def _check_drink_stability_enhanced(self, params):
        """增強版穩定性檢查，考慮物件特性"""
        try:
            if self.target_drink_name not in self.object_body_ids:
                return True
                
            current_pos = self.sim.data.body_xpos[self.object_body_ids[self.target_drink_name]].copy()
            collision_sens = params["collision_sensitivity"]
            
            # 初始化階段
            if self.initialization_steps < 30:
                self.drink_position_history.append(current_pos)
                self._limit_history_length()
                return True
            
            self.drink_position_history.append(current_pos)
            self._limit_history_length()
            
            if len(self.drink_position_history) < 5:
                return True
            
            # 基於物件敏感度調整穩定性閾值
            adaptive_threshold = params["stability_threshold"] / collision_sens
            
            recent_positions = np.array(self.drink_position_history[-5:])
            position_changes = np.linalg.norm(np.diff(recent_positions, axis=0), axis=1)
            max_change = np.max(position_changes)
            
            is_stable = max_change <= adaptive_threshold
            
            if not is_stable:
                self.consecutive_unstable_steps += 1
                confirmation_needed = max(2, int(3 * collision_sens))  # 敏感物件需要更多確認
                
                if self.consecutive_unstable_steps >= confirmation_needed:
                    if self.drink_stable:
                        drink_name = self._get_drink_display_name()
                        print(f"檢測到{drink_name}不穩定！變化: {max_change:.4f} (閾值: {adaptive_threshold:.4f})")
                        self.drink_stable = False
                    return False
            else:
                if self.consecutive_unstable_steps > 0:
                    self.consecutive_unstable_steps = 0
                    if not self.drink_stable:
                        print(f"{self._get_drink_display_name()}重新穩定")
                        self.drink_stable = True
                return True
                
        except Exception as e:
            print(f"Error in enhanced stability check: {e}")
            return True

    def get_task_analysis(self):
        """獲取任務分析信息，特別針對多目標任務"""
        analysis = self.get_current_task_info()
        
        if self.current_right_object == "mug":
            # 添加馬克杯任務的額外分析
            try:
                eef_pos = self._get_eef_position()
                if eef_pos is not None:
                    water_pos = self.sim.data.body_xpos[self.object_body_ids["water_jug"]]
                    tea_pos = self.sim.data.body_xpos[self.object_body_ids["packed_tea"]]
                    
                    dist_to_water = np.linalg.norm(eef_pos - water_pos)
                    dist_to_tea = np.linalg.norm(eef_pos - tea_pos)
                    
                    analysis.update({
                        "is_multi_target_task": True,
                        "distance_to_water": dist_to_water,
                        "distance_to_tea": dist_to_tea,
                        "choice_clarity": abs(dist_to_water - dist_to_tea),
                        "preferred_target": "water" if dist_to_water < dist_to_tea else "tea",
                        "actual_target": "water" if self.target_drink_name == "water_jug" else "tea"
                    })
            except:
                pass
        else:
            analysis["is_multi_target_task"] = False
        
        # 添加物件特性分析
        if hasattr(self, 'target_drink_name'):
            params = self._get_object_params(self.target_drink_name)
            analysis["object_sensitivity"] = params["collision_sensitivity"]
            analysis["adaptive_thresholds"] = params
        
        return analysis