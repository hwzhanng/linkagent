#!/usr/bin/env python3
"""ROS 2 replacement for the Pi CAN HAL and recovered high-level arm API.

The node starts read-only and never enables or moves an arm during
construction.  FK/IK remain available read-only; motion calls retain all CAN
runtime safety gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from scipy.spatial.transform import Rotation

try:
    from lbot_arm_interfaces.msg import ArmState as ArmStateMessage
    from lbot_arm_interfaces.msg import RobotRunState
    from lbot_arm_interfaces.srv import FK, FollowP, GetJointLimit, IK
    from lbot_arm_interfaces.srv import MoveC, MoveJ, MoveJP, MoveL
except ImportError as error:
    raise SystemExit(
        "lbot_arm_interfaces is not available; source "
        "tools/local_arm_can/ros_ws/install/setup.bash first"
    ) from error

from tools.local_arm_can.advanced_controller import (
    AdvancedArmController,
    RUN_DISABLED,
    RUN_IDLE,
)
from tools.local_arm_can.kinematics import Pose
from tools.local_arm_can.service import ArmCanRuntime


JOINT_NAMES = {
    "right": [f"R{index}_Joint" for index in range(1, 8)],
    "left": [f"L{index}_Joint" for index in range(1, 8)],
}


class LocalArmCanNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(f"{args.arm}_local_lbot_arm", namespace=args.arm)
        self.runtime = ArmCanRuntime(
            args.interface,
            arm=args.arm,
            allow_tx=args.allow_tx,
            allow_can0=args.allow_can0,
            poll_telemetry=args.poll_telemetry,
        )
        self.runtime.start()
        self.arm = args.arm
        self.evo_parameters = args.evo_parameters
        self.physical_confirmation = args.physical_confirmation
        self.max_joint_step = args.max_joint_step
        with open(args.config, encoding="utf-8") as config_file:
            self.config = json.load(config_file)
        if self.evo_parameters is None:
            parameters = self.config["motorevo_parameters"]
            self.evo_parameters = [
                float(parameters[name])
                for name in ("velocity", "kp", "kd", "kp_i", "kd_i", "ki_i")
            ]
        self.advanced = AdvancedArmController(self.runtime, config_path=args.config)
        self.callback_group = ReentrantCallbackGroup()
        self.publisher = self.create_publisher(JointState, "joint_states", 10)
        self.arm_state_publisher = self.create_publisher(ArmStateMessage, "arm_state", 10)
        self.run_state_publisher = self.create_publisher(
            RobotRunState, "robot_run_state", 10
        )
        self.create_subscription(
            JointState, "joint_cmd", self._joint_command, 10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Bool, "teach_cmd", self._teach_command, 10,
            callback_group=self.callback_group,
        )
        self.create_service(
            SetBool, "enable_arm", self._enable, callback_group=self.callback_group
        )
        self.create_service(
            SetBool, "emergency_stop", self._emergency_stop,
            callback_group=self.callback_group,
        )
        self.create_service(
            GetJointLimit, "get_joint_limit", self._get_joint_limit,
            callback_group=self.callback_group,
        )
        self.create_service(FK, "fk", self._fk, callback_group=self.callback_group)
        self.create_service(IK, "ik", self._ik, callback_group=self.callback_group)
        self.create_service(MoveJ, "movej", self._movej, callback_group=self.callback_group)
        self.create_service(MoveJP, "movejp", self._movejp, callback_group=self.callback_group)
        self.create_service(MoveL, "movel", self._movel, callback_group=self.callback_group)
        self.create_service(MoveC, "movec", self._movec, callback_group=self.callback_group)
        self.create_service(FollowP, "followp", self._followp, callback_group=self.callback_group)
        self.create_timer(0.02, self._publish_state, callback_group=self.callback_group)
        self.get_logger().info(
            f"local CAN HAL started on {args.interface}; TX is "
            f"{'enabled' if args.allow_tx else 'locked'}"
        )

    def destroy_node(self) -> bool:
        self.runtime.close()
        return super().destroy_node()

    def _publish_state(self) -> None:
        snapshot = self.runtime.snapshot()
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = JOINT_NAMES[self.arm]
        message.position = [row["position"] or 0.0 for row in snapshot["joints"]]
        message.velocity = [row["velocity"] or 0.0 for row in snapshot["joints"]]
        message.effort = [row["effort"] or 0.0 for row in snapshot["joints"]]
        self.publisher.publish(message)

        arm_state = ArmStateMessage()
        arm_state.joint_state = message
        arm_state.joint_temperature = [
            row["temperature"] or 0.0 for row in snapshot["joints"]
        ]
        if snapshot["all_joints_healthy"]:
            pose = self.advanced.forward(message.position)
            arm_state.pose.position.x = pose.position[0]
            arm_state.pose.position.y = pose.position[1]
            arm_state.pose.position.z = pose.position[2]
            quaternion = Rotation.from_euler(
                "ZYX", [pose.rpy[2], pose.rpy[1], pose.rpy[0]]
            ).as_quat()
            arm_state.pose.orientation.x = float(quaternion[0])
            arm_state.pose.orientation.y = float(quaternion[1])
            arm_state.pose.orientation.z = float(quaternion[2])
            arm_state.pose.orientation.w = float(quaternion[3])
            arm_state.euler.x = pose.rpy[0]
            arm_state.euler.y = pose.rpy[1]
            arm_state.euler.z = pose.rpy[2]
        self.arm_state_publisher.publish(arm_state)

        run_state = RobotRunState()
        run_state.state = self.advanced.run_state if snapshot["software_enabled"] else 22
        self.run_state_publisher.publish(run_state)

    def _get_joint_limit(
        self,
        request: GetJointLimit.Request,
        response: GetJointLimit.Response,
    ) -> GetJointLimit.Response:
        section_name = "default" if request.cmd == "get_default_joint_limit" else "current"
        if request.cmd not in ("get_joint_limit", "get_default_joint_limit"):
            response.success = False
            return response
        limits = self.config["joint_limits"][section_name]
        response.lower_limits = [float(value) for value in limits["lower"]]
        response.upper_limits = [float(value) for value in limits["upper"]]
        response.success = True
        return response

    def _fk(self, request: FK.Request, response: FK.Response) -> FK.Response:
        try:
            pose = self.advanced.forward(list(request.joints))
            response.position.x, response.position.y, response.position.z = pose.position
            response.euler.x, response.euler.y, response.euler.z = pose.rpy
            response.success = True
        except Exception as error:
            self.get_logger().error(f"FK rejected: {error}")
            response.success = False
        return response

    def _ik(self, request: IK.Request, response: IK.Response) -> IK.Response:
        try:
            result = self.advanced.inverse(
                (request.position.x, request.position.y, request.position.z),
                (request.euler.x, request.euler.y, request.euler.z),
                list(request.joints),
            )
            response.joints = list(result.joints) if result.success else []
            response.success = result.success
        except Exception as error:
            self.get_logger().error(f"IK rejected: {error}")
            response.success = False
        return response

    @staticmethod
    def _request_pose(request: object) -> Pose:
        return Pose(
            (request.position.x, request.position.y, request.position.z),
            (request.euler.x, request.euler.y, request.euler.z),
        )

    def _movej(self, request: MoveJ.Request, response: MoveJ.Response) -> MoveJ.Response:
        try:
            response.success = self.advanced.movej(
                list(request.joints), request.speed, request.acce, request.block,
                self.physical_confirmation,
            )
        except Exception as error:
            self.get_logger().error(f"MoveJ rejected: {error}")
            response.success = False
        return response

    def _movejp(self, request: MoveJP.Request, response: MoveJP.Response) -> MoveJP.Response:
        try:
            response.success = self.advanced.movejp(
                self._request_pose(request), request.speed, request.acce, request.block,
                self.physical_confirmation,
            )
        except Exception as error:
            self.get_logger().error(f"MoveJP rejected: {error}")
            response.success = False
        return response

    def _movel(self, request: MoveL.Request, response: MoveL.Response) -> MoveL.Response:
        try:
            response.success = self.advanced.movel(
                self._request_pose(request), request.speed, request.acce, request.block,
                self.physical_confirmation,
            )
        except Exception as error:
            self.get_logger().error(f"MoveL rejected: {error}")
            response.success = False
        return response

    def _movec(self, _request: MoveC.Request, response: MoveC.Response) -> MoveC.Response:
        # The shipped Pi function is a 28-byte stub.  Returning false matches
        # the official controller instead of inventing circular-path behavior.
        response.success = False
        return response

    def _followp(self, request: FollowP.Request, response: FollowP.Response) -> FollowP.Response:
        try:
            response.success = self.advanced.followp(
                self._request_pose(request), self.physical_confirmation
            )
        except Exception as error:
            self.get_logger().error(f"FollowP rejected: {error}")
            response.success = False
        return response

    def _enable(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        try:
            if request.data and self.advanced.run_state not in (RUN_IDLE, RUN_DISABLED):
                raise RuntimeError("cannot enable while a trajectory is active")
            if not request.data:
                self.advanced.cancel()
            self.runtime.set_enabled(request.data, self.physical_confirmation)
            self.advanced.set_enabled_state(request.data)
            response.success = True
            response.message = "enabled" if request.data else "disabled"
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def _emergency_stop(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        # The recovered Pi implementation disables all motors when data=True.
        try:
            if not request.data:
                raise RuntimeError("emergency stop release does not auto-enable the arm")
            self.advanced.cancel()
            self.runtime.set_enabled(False, self.physical_confirmation)
            self.advanced.set_enabled_state(False)
            response.success = True
            response.message = "motors disabled"
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def _teach_command(self, message: Bool) -> None:
        if not message.data:
            self.get_logger().warning("teach_cmd=false does not auto-enable; call enable_arm explicitly")
            return
        try:
            self.advanced.cancel()
            self.runtime.set_enabled(False, self.physical_confirmation)
            self.advanced.set_enabled_state(False)
        except Exception as error:
            self.get_logger().error(str(error))

    def _joint_command(self, message: JointState) -> None:
        try:
            if self.advanced.run_state != RUN_IDLE:
                raise RuntimeError("joint_cmd rejected while an advanced trajectory is active")
            self.advanced.kinematics.validate_joints(
                list(message.position), enforce_limits=True
            )
            current = self.advanced.current_joints()
            if max(
                abs(target - actual)
                for target, actual in zip(message.position, current, strict=True)
            ) > self.max_joint_step:
                raise RuntimeError(
                    f"joint_cmd exceeds {self.max_joint_step:.3f} rad single-step limit"
                )
            self.runtime.command_positions(
                list(message.position),
                self.physical_confirmation,
                self.evo_parameters,
            )
        except Exception as error:
            self.get_logger().error(f"joint command rejected: {error}")


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True)
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--allow-can0", action="store_true")
    parser.add_argument("--allow-tx", action="store_true")
    parser.add_argument("--poll-telemetry", action="store_true")
    parser.add_argument("--physical-confirmation", default="")
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument(
        "--config",
        default=None,
    )
    parser.add_argument(
        "--evo-parameters",
        type=lambda value: [float(item) for item in value.split(",")],
        default=None,
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, ros_args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.config is None:
        args.config = str(Path(__file__).with_name(f"a7_v2_{args.arm}.json"))
    if args.evo_parameters is not None and len(args.evo_parameters) != 6:
        raise SystemExit("--evo-parameters requires six comma-separated values")
    if not 0.0 < args.max_joint_step <= 0.2:
        raise SystemExit("--max-joint-step must be in (0, 0.2] radians")
    rclpy.init(args=ros_args)
    node = LocalArmCanNode(args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        # rclpy's SIGINT handler may already have shut the context down.
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
