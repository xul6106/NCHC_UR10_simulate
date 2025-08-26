"""
A script to collect a batch of human demonstrations with rewards recording.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


import os
import numpy as np
from robosuite.wrappers import DataCollectionWrapper


class RewardDataCollectionWrapper(DataCollectionWrapper):
    """Extended DataCollectionWrapper that also records rewards"""
    
    def __init__(self, env, directory):
        super().__init__(env, directory)
        self.rewards = []
        
    def step(self, action):
        # Call the parent's step, which handles state and action collection
        obs, reward, done, info = super().step(action)
        # Record the reward
        self.rewards.append(reward)
        return obs, reward, done, info
    
    def _flush(self):
        """
        Completely override the _flush method to manually save rewards to
        the most recent episode directory created by the parent wrapper.
        This version avoids relying on potentially unstable attributes like self.ep_str.
        """
        # Find the most recently created episode directory
        episode_dirs = sorted(glob(os.path.join(self.directory, "ep_*")), key=os.path.getctime, reverse=True)
        if not episode_dirs:
            print("Error: No episode directory found. Skipping reward save.")
            return
            
        ep_directory = episode_dirs[0]
        
        # Call the parent's flush method to save states, actions, etc.
        super()._flush()
        
        # Save rewards to the current episode data
        if len(self.rewards) > 0:
            rewards_path = os.path.join(ep_directory, "rewards.npz")
            np.savez(rewards_path, rewards=np.array(self.rewards))
            
    def reset(self):
        """Override to also reset the rewards list when an episode resets."""
        # Call the parent's reset method first to handle its internal state
        ret = super().reset()
        # Reset the rewards list for the new episode
        self.rewards = []
        return ret


def collect_human_trajectory(env, device, arm, max_fr):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
    """

    env.reset()
    env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = device.input2action()

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action)
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    ... (註釋保持不變)
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        rewards = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        # Load rewards if available
        rewards_path = os.path.join(directory, ep_directory, "rewards.npz")
        if os.path.exists(rewards_path):
            rewards_data = np.load(rewards_path)
            rewards = rewards_data["rewards"].tolist()
        else:
            print(f"Warning: No rewards file found for episode {ep_directory}")
            rewards = [0.0] * len(actions)  # Default to zero rewards

        if len(states) == 0:
            continue

        print(f"Found demonstration {ep_directory}, success: {success}")
        del states[-1]
        
        if len(rewards) > len(actions):
            rewards = rewards[:-1]
        elif len(rewards) < len(actions):
            rewards.extend([0.0] * (len(actions) - len(rewards)))
            
        assert len(states) == len(actions) == len(rewards)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f_xml:
            xml_str = f_xml.read()
        
        # --- 修改開始 ---
        # 這裡的字串需要被編碼成位元組字串
        ep_data_grp.attrs["model_file"] = xml_str.encode('utf-8')
        ep_data_grp.attrs["successful"] = int(success)

        # write datasets for states, actions, and rewards
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        ep_data_grp.create_dataset("rewards", data=np.array(rewards))
        
        print(f"Saved demo with {len(states)} states, {len(actions)} actions, and {len(rewards)} rewards to HDF5")
        # --- 修改結束 ---

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    # --- 修改開始 ---
    # 所有字串屬性都需要被編碼
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year).encode('utf-8')
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second).encode('utf-8')
    grp.attrs["repository_version"] = suite.__version__.encode('utf-8')
    grp.attrs["env"] = str(env_name).encode('utf-8')
    grp.attrs["env_info"] = env_info.encode('utf-8')
    # --- 修改結束 ---

    f.close()
    print(f"Successfully saved {num_eps} demonstrations to {hdf5_path}")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="agentview",
        help="List of camera names to use for collecting demos. Pass multiple names to enable multiple views. Note: the `mujoco` renderer must be enabled when using multiple views; `mjviewer` is not supported.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        #render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper (now includes rewards)
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = RewardDataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)
    # 初始化一個計數器
    demo_count = 0

    # collect demonstrations
    while True:
        demo_count += 1
        print(f"--- 開始第 {demo_count} 次示範採集 ---")
        collect_human_trajectory(env, device, args.arm, args.max_fr)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)