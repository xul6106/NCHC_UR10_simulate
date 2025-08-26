"""
雙桌面場景 Arena - 在 Robosuite 中創建包含兩個桌子的場景
"""

import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class DualTableArena(Arena):
    """
    包含兩個桌子的競技場
    - 左桌：放置固定物件
    - 右桌：放置隨機物件
    """
    
    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        table_offset=(0, 0, 0.8),
        xml="arenas/dual_table_arena.xml"
    ):
        """
        Args:
            table_full_size (3-tuple): 桌子的完整尺寸 (長, 寬, 高)
            table_friction (3-tuple): 桌子表面摩擦參數
            table_offset (3-tuple): 桌子相對於世界原點的偏移
            xml (str): Arena XML 檔案路徑
        """
        self.table_full_size = np.array(table_full_size)
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)
        self.table_half_size = self.table_full_size / 2
        
        # 計算兩個桌子的位置
        self.left_table_pos = np.array([-0.6, 0, 0]) + self.table_offset
        self.right_table_pos = np.array([0.6, 0, 0]) + self.table_offset
        
        super().__init__(xml=xml)

    def _get_arena_xml(self):
        """
        生成雙桌面 Arena 的 XML
        """
        xml_str = f"""
        <mujoco model="dual_table_arena">
            <asset>
                <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
                <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
                <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
                <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
                <material name="geom" texture="texgeom" texuniform="true"/>
            </asset>
            
            <worldbody>
                <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true" pos="0 0 1.3" specular=".1 .1 .1"/>
                <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 1 1" size="40 40 40" type="plane"/>
                
                <!-- 左桌 -->
                <body name="left_table" pos="{array_to_string(self.left_table_pos)}">
                    <geom name="left_table_collision" pos="0 0 0" size="{array_to_string(self.table_half_size)}" type="box" 
                          friction="{array_to_string(self.table_friction)}" material="geom"/>
                    <geom name="left_table_visual" pos="0 0 0" size="{array_to_string(self.table_half_size)}" type="box" 
                          rgba="0.6 0.4 0.2 1" contype="0" conaffinity="0"/>
                </body>
                
                <!-- 右桌 -->
                <body name="right_table" pos="{array_to_string(self.right_table_pos)}">
                    <geom name="right_table_collision" pos="0 0 0" size="{array_to_string(self.table_half_size)}" type="box" 
                          friction="{array_to_string(self.table_friction)}" material="geom"/>
                    <geom name="right_table_visual" pos="0 0 0" size="{array_to_string(self.table_half_size)}" type="box" 
                          rgba="0.6 0.4 0.2 1" contype="0" conaffinity="0"/>
                </body>
            </worldbody>
        </mujoco>
        """
        return xml_str

    @property
    def table_top_abs(self):
        """返回桌面的絕對高度"""
        return self.table_offset[2] + self.table_half_size[2]
    
    @property
    def left_table_top_abs(self):
        """返回左桌桌面的絕對位置"""
        return self.left_table_pos + np.array([0, 0, self.table_half_size[2]])
    
    @property
    def right_table_top_abs(self):
        """返回右桌桌面的絕對位置"""
        return self.right_table_pos + np.array([0, 0, self.table_half_size[2]])

    def get_object_placement_bounds(self, table="left"):
        """
        獲取指定桌子上物件放置的邊界
        
        Args:
            table (str): "left" 或 "right"
            
        Returns:
            tuple: (min_bounds, max_bounds) 每個都是 3D 座標陣列
        """
        if table == "left":
            table_pos = self.left_table_pos
        else:
            table_pos = self.right_table_pos
        
        # 桌面邊界（留一些邊距）
        margin = 0.1
        table_bounds = self.table_half_size - margin
        
        min_bounds = table_pos + np.array([-table_bounds[0], -table_bounds[1], self.table_half_size[2]])
        max_bounds = table_pos + np.array([table_bounds[0], table_bounds[1], self.table_half_size[2] + 0.2])
        
        return min_bounds, max_bounds