import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion


class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bottle.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class CanObject(MujocoXMLObject):
    """
    Coke can object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/can.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/lemon.xml"), name=name, obj_type="all", duplicate_collision_geoms=True
        )


class MilkObject(MujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/milk.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

   


class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bread.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-nut.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic


class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic


class MilkVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in PickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/milk-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class BreadVisualObject(MujocoXMLObject):
    """
    Visual fiducial of bread loaf (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bread-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class CerealVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class CanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/can-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in PegInHole)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/plate-with-hole.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class DoorObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name, friction=None, damping=None, lock=False):
        xml_path = "objects/door.xml"
        if lock:
            xml_path = "objects/door_lock.xml"
        super().__init__(
            xml_path_completion(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )

        # Set relevant body names
        self.door_body = self.naming_prefix + "door"
        self.frame_body = self.naming_prefix + "frame"
        self.latch_body = self.naming_prefix + "latch"
        self.hinge_joint = self.naming_prefix + "hinge"

        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic

# ========== 自訂物件 for NCHC UR10 project ==========
class MugObject(MujocoXMLObject):
    """
    My custom mug object
    """
    def __init__(self, name, size=0.0012):
        super().__init__(
            xml_path_completion("objects/mug.xml"), 
            name=name, 
            joints=[dict(type="free", damping="0.1")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        # 假設 mug 的原始高度約為 0.12m
        # 這裡設定縮放比例，以達到預設的 0.1m 高度
        scale = size / 0.12
        self.set_scale([scale, scale, scale])


class WineGlassObject(MujocoXMLObject):
    """
    My custom wine glass object
    """
    def __init__(self, name, size=0.13):
        super().__init__(
            xml_path_completion("objects/wine_glass.xml"), 
            name=name, 
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        # 假設 wine glass 的原始高度約為 0.20m
        scale = size / 0.20
        self.set_scale([scale, scale, scale])


class GlassObject(MujocoXMLObject):
    """
    My custom glass object
    """
    def __init__(self, name, size=0.01):
        super().__init__(
            xml_path_completion("objects/glass.xml"), 
            name=name, 
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        # 假設 glass 的原始高度約為 0.18m
        scale = size / 0.18
        self.set_scale([scale, scale, scale])


class TeaPotObject(MujocoXMLObject):
    """
    My custom tea pot object
    """
    def __init__(self, name, size=0.2):
        super().__init__(
            xml_path_completion("objects/tea_pot.xml"), 
            name=name, 
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        # 假設 tea pot 的原始高度約為 0.15m
        scale = size / 0.15
        self.set_scale([scale, scale, scale])


class WineBottleObject(MujocoXMLObject):
    """
    My custom wine bottle object
    """
    def __init__(self, name, size=0.003):
        super().__init__(
            xml_path_completion("objects/wine_bottle.xml"), 
            name=name, 
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        # 假設 wine bottle 的原始高度約為 0.30m
        scale = size / 0.30
        self.set_scale([scale, scale, scale])


class CoffeePotObject(MujocoXMLObject):
    """
    My custom coffee pot object
    """
    def __init__(self, name, size=0.011):
        super().__init__(
            xml_path_completion("objects/coffee_pot.xml"), 
            name=name, 
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        # 假設 coffee pot 的原始高度約為 0.25m
        scale = size / 0.25
        self.set_scale([scale, scale, scale])


class MilkPackObject(MujocoXMLObject):
    """
    My custom Milk carton object
    """

    def __init__(self, name, size=0.0015):
        super().__init__(
            xml_path_completion("objects/milkpack.xml"),
            name=name,
            joints=[dict(type="free", damping="0.1")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        scale = size / 0.30
        self.set_scale([scale, scale, scale])

class WaterJugObject(MujocoXMLObject):
    """
    My custom water jug object
    """
    def __init__(self, name, size=0.004):
        super().__init__(
            xml_path_completion("objects/water_jug.xml"), 
            name=name, 
            joints=[dict(type="free", damping="0.05")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        # 假設 tea pot 的原始高度約為 0.15m
        scale = size / 0.15
        self.set_scale([scale, scale, scale])

class BeerCanObject(MujocoXMLObject):
    """
    My custom beer can object
    """
    def __init__(self, name, size=0.001):
        super().__init__(
            xml_path_completion("objects/beer_can.xml"), 
            name=name, 
            joints=[dict(type="free", damping="0.05")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        # 假設 tea pot 的原始高度約為 0.15m
        scale = size / 0.15
        self.set_scale([scale, scale, scale])

class BeerGlassObject(MujocoXMLObject):
    """
    My custom beer glass object
    """
    def __init__(self, name, size=0.003):
        super().__init__(
            xml_path_completion("objects/beer_glass.xml"), 
            name=name, 
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        # 假設 glass 的原始高度約為 0.18m
        scale = size / 0.18
        self.set_scale([scale, scale, scale])