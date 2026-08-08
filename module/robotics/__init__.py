"""Public SDK-neutral robot ports, leases, mapping, and deterministic mocks."""

from .bundle import RoboticsBundle
from .clock import Clock, ManualClock, SystemClock
from .control import (
    ControlBusyError,
    ControlLease,
    ControlLeaseManager,
    InvalidControlLeaseError,
)
from .factory import (
    ARM_PORT_REGISTRY,
    HAND_PORT_REGISTRY,
    ROBOTICS_REGISTRY,
    TACTILE_PORT_REGISTRY,
    build_robotics_bundle,
)
from .deployment import (
    ArmEndpointConfig,
    BimanualDeploymentConfig,
    HandEndpointConfig,
)
from .mapping import HandSynergyMapper, MockHandSynergyMapper
from .mock import (
    MockArmControlPort,
    MockHandControlPort,
    MockPickOutcome,
    MockPickScenario,
    MockPickWorld,
    MockRoboticsBundle,
    MockTactilePort,
    build_mock_robotics_bundle,
)
from .ports import (
    ArmControlPort,
    FKResult,
    FollowPoseCommand,
    HandControlPort,
    HandJointPositionCommand,
    HandSession,
    HandSynergyCommand,
    IKResult,
    JointMoveCommand,
    LinearMoveCommand,
    PoseMoveCommand,
    TactilePort,
)
from .system import (
    BimanualRobotInfra,
    BimanualRobotSnapshot,
    EmergencyStopReport,
)
from .shadow import (
    LinkBotG20ShadowBundle,
    ReadOnlyControlLeaseManager,
    build_linkbot_g20_shadow_bundle,
)
from .preflight import (
    DeploymentPreflightReport,
    PreflightCheck,
    run_deployment_preflight,
)
from .fake_gateway import (
    FakeArmSdkGateway,
    FakeG20SdkGateway,
    FakeGatewayFaults,
    FakeGatewayRoboticsBundle,
    build_fake_gateway_robotics_bundle,
)

__all__ = [
    "ARM_PORT_REGISTRY",
    "ArmEndpointConfig",
    "ArmControlPort",
    "BimanualDeploymentConfig",
    "BimanualRobotInfra",
    "BimanualRobotSnapshot",
    "Clock",
    "ControlBusyError",
    "ControlLease",
    "ControlLeaseManager",
    "DeploymentPreflightReport",
    "FKResult",
    "FakeArmSdkGateway",
    "FakeG20SdkGateway",
    "FakeGatewayFaults",
    "FakeGatewayRoboticsBundle",
    "FollowPoseCommand",
    "HAND_PORT_REGISTRY",
    "HandControlPort",
    "HandJointPositionCommand",
    "HandEndpointConfig",
    "HandSession",
    "HandSynergyCommand",
    "HandSynergyMapper",
    "IKResult",
    "InvalidControlLeaseError",
    "JointMoveCommand",
    "LinearMoveCommand",
    "LinkBotG20ShadowBundle",
    "ManualClock",
    "MockArmControlPort",
    "MockHandControlPort",
    "MockHandSynergyMapper",
    "MockPickOutcome",
    "MockPickScenario",
    "MockPickWorld",
    "MockRoboticsBundle",
    "MockTactilePort",
    "PoseMoveCommand",
    "PreflightCheck",
    "ROBOTICS_REGISTRY",
    "ReadOnlyControlLeaseManager",
    "RoboticsBundle",
    "SystemClock",
    "TACTILE_PORT_REGISTRY",
    "TactilePort",
    "EmergencyStopReport",
    "build_robotics_bundle",
    "build_mock_robotics_bundle",
    "build_linkbot_g20_shadow_bundle",
    "build_fake_gateway_robotics_bundle",
    "run_deployment_preflight",
]
