"""Pre-startup asset authoring for the independent Physics-v1 task."""

from __future__ import annotations

import hashlib
from itertools import combinations
import json

import isaaclab.sim as sim_utils
from simulation.tomato_pick.physics_v1 import (
    load_physics_v1_profile,
    quantize_contact_profiles,
    sample_correlated_fruit_profiles,
)


def verify_baked_hand_self_collision_filters(env, env_ids, *, asset_name: str) -> None:
    """Verify the once-authored Physics-v1 collision matrix.

    The 12-body palm/distal collision proxy stores 7 filtered relationships
    and keeps 59 cross-digit/palm self-collision pairs enabled.
    Coarse arm, parked-left-hand, and overlapping mounting carriers have
    contact solving disabled.  This event inspects one clone and constant-size
    metadata on the last clone; it never authors O(num_envs) relationships.
    """

    if env.sim.is_playing():
        raise RuntimeError("Physics-v1 baked collision filters must be verified prestartup")
    if env_ids is not None:
        raise ValueError("Physics-v1 collision verification covers the complete scene")
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = env.sim.stage
    robot_paths = sim_utils.find_matching_prim_paths(env.scene[asset_name].cfg.prim_path)
    if len(robot_paths) != env.scene.num_envs:
        raise RuntimeError(
            f"expected {env.scene.num_envs} Physics-v1 robots, found {len(robot_paths)}"
        )
    expected = {
        "physicsV1SelfCollisionPolicy": "right_distal_envelope_and_palm_only",
        "physicsV1SelfCollisionMatrixSha256": (
            "16653f7ed1d19e6b6bf0a5e25b74ad57c584c74eb548e8af18f3b2a473cd5755"
        ),
        "physicsV1FilteredPairsPerRobot": 2426,
        "physicsV1EnabledPairsPerRobot": 59,
        "physicsV1AdjacentPairsPerRobot": 54,
        "physicsV1ExpectedHandContactBodyCount": 56,
        "physicsV1SelfCollisionRepresentation": (
            "distal_envelope_filtered_pairs_outside_solve_contact_disabled_v2"
        ),
        "physicsV1StoredActiveHandFilteredPairCount": 7,
        "physicsV1OutsideSolveContactDisabledBodyCount": 59,
    }

    first_root = stage.GetPrimAtPath(robot_paths[0])
    last_root = stage.GetPrimAtPath(robot_paths[-1])
    if not first_root.IsValid() or not last_root.IsValid():
        raise RuntimeError("Physics-v1 derived robot reference did not compose")
    for root in (first_root, last_root):
        for key, value in expected.items():
            actual = root.GetCustomDataByKey(key)
            if actual != value:
                raise RuntimeError(
                    f"Physics-v1 baked asset metadata mismatch {key}: "
                    f"expected={value!r}, actual={actual!r}"
                )

    rigid_body_count = 0
    stored_active_hand_pairs = 0
    exposed_right_hand_bodies = 0
    outside_solve_contact_disabled = 0
    actual_contact_reporter_count = 0
    for prim in Usd.PrimRange(first_root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_body_count += 1
        name = prim.GetName()
        exposed_right_hand = name == "right_hand_base_link"
        if not exposed_right_hand:
            for digit in ("thumb", "index", "middle", "ring", "little"):
                prefix = f"right_hand_{digit}_link"
                if name.startswith(prefix):
                    link_index = int(name.removeprefix(prefix))
                    exposed_right_hand = link_index >= 3
                    break
        solve_contact = prim.GetAttribute("physxRigidBody:solveContact").Get()
        if exposed_right_hand:
            exposed_right_hand_bodies += 1
            if solve_contact is False:
                raise RuntimeError(f"exposed hand contact solving disabled: {prim.GetPath()}")
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                stored_active_hand_pairs += len(
                    UsdPhysics.FilteredPairsAPI(prim)
                    .GetFilteredPairsRel()
                    .GetTargets()
                )
        elif solve_contact is False:
            outside_solve_contact_disabled += 1
        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            actual_contact_reporter_count += 1
    if rigid_body_count != 71:
        raise RuntimeError(
            f"Physics-v1 robot topology changed: expected 71 bodies, got {rigid_body_count}"
        )
    if exposed_right_hand_bodies != 12:
        raise RuntimeError(
            "expected 12 palm/distal-pad contact bodies, got "
            f"{exposed_right_hand_bodies}"
        )
    if stored_active_hand_pairs != expected[
        "physicsV1StoredActiveHandFilteredPairCount"
    ]:
        raise RuntimeError(
            "Physics-v1 stored active-hand pair count changed: "
            f"expected=7, actual={stored_active_hand_pairs}"
        )
    if outside_solve_contact_disabled != expected[
        "physicsV1OutsideSolveContactDisabledBodyCount"
    ]:
        raise RuntimeError(
            "Physics-v1 out-of-scope solveContact boundary changed: "
            f"expected=59, actual={outside_solve_contact_disabled}"
        )
    # The derived topology contains 56 bodies across both hands.  Physics-v1
    # sensors intentionally activate only the 28 bodies of the task-active
    # right hand; the parked left hand does not need contact telemetry.
    expected_active_contact_bodies = (
        expected["physicsV1ExpectedHandContactBodyCount"] // 2
    )
    if actual_contact_reporter_count < expected_active_contact_bodies:
        raise RuntimeError(
            "Physics-v1 contact sensor activation did not cover the active hand: "
            f"expected>={expected_active_contact_bodies}, "
            f"actual={actual_contact_reporter_count}"
        )

    env._physics_v1_hand_self_collision_filter = {
        "enabled": True,
        "asset_baked": True,
        "runtime_authored_pair_count": 0,
        "policy": expected["physicsV1SelfCollisionPolicy"],
        "pairs_per_robot": expected["physicsV1FilteredPairsPerRobot"],
        "baked_pair_count": expected["physicsV1FilteredPairsPerRobot"],
        "stored_active_hand_pair_count": stored_active_hand_pairs,
        "outside_solve_contact_disabled_body_count": (
            outside_solve_contact_disabled
        ),
        "verified_robot_instance_count": len(robot_paths),
        "adjacent_joint_pair_count": expected["physicsV1AdjacentPairsPerRobot"],
        "allowed_nonadjacent_hand_pair_count": expected[
            "physicsV1EnabledPairsPerRobot"
        ],
        "contact_reporter_body_count": actual_contact_reporter_count,
        "expected_hand_contact_body_count": expected[
            "physicsV1ExpectedHandContactBodyCount"
        ],
        "expected_active_hand_contact_body_count": expected_active_contact_bodies,
        "unique_relative_pair_count": expected["physicsV1FilteredPairsPerRobot"],
        "pair_matrix_sha256": expected["physicsV1SelfCollisionMatrixSha256"],
        "nonadjacent_hand_pairs_enabled": True,
        "active_hand_side": "right",
        "external_contact_body_count": exposed_right_hand_bodies,
        "self_collision_envelope_body_count": 12,
        "same_digit_nonadjacent_collision": False,
        "cross_digit_collision": "distal_envelope_only",
        "distal_palm_collision": True,
        "arm_self_collision_deferred_to_simplified_collider_asset": True,
    }


def author_hand_self_collision_filters(env, env_ids, *, asset_name: str) -> None:
    """Filter only directly joint-connected hand bodies.

    Physics-v1 enables articulation self collision at spawn time.  Adjacent
    rigid bodies share joint boundary geometry, so their collision is filtered
    explicitly.  The task-active right hand keeps cross-digit collisions and
    distal/palm collisions; same-chain and mounting geometry are filtered.
    This preserves the thumb-vs-finger barrier without letting coarse proximal
    convex hulls inject energy into the articulation.
    """

    if env.sim.is_playing():
        raise RuntimeError(
            "Physics-v1 self-collision filters must be authored before physics starts"
        )
    if env_ids is not None:
        raise ValueError(
            "Physics-v1 self-collision filters must cover every environment"
        )
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = env.sim.stage
    robot_paths = sim_utils.find_matching_prim_paths(env.scene[asset_name].cfg.prim_path)
    if len(robot_paths) != env.scene.num_envs:
        raise RuntimeError(
            f"expected {env.scene.num_envs} Physics-v1 robots, found {len(robot_paths)}"
        )

    pair_records: list[tuple[str, str, str]] = []
    authored_count = 0
    expected_pairs_per_robot: int | None = None
    adjacent_joint_pair_count = 0
    scope_isolation_pair_count = 0
    allowed_nonadjacent_hand_pair_count = 0
    reporter_body_count = 0
    for robot_path in robot_paths:
        root = stage.GetPrimAtPath(robot_path)
        robot_records: list[tuple[str, str, str]] = []
        robot_body_paths = set()
        filtered_body_pairs = set()
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint) or "_hand_" not in prim.GetName():
                continue
            joint = UsdPhysics.Joint(prim)
            body0_targets = joint.GetBody0Rel().GetTargets()
            body1_targets = joint.GetBody1Rel().GetTargets()
            if len(body0_targets) != 1 or len(body1_targets) != 1:
                raise RuntimeError(
                    "Physics-v1 hand joint must connect exactly two rigid bodies: "
                    f"{prim.GetPath()} body0={body0_targets} body1={body1_targets}"
                )
            body0_path, body1_path = body0_targets[0], body1_targets[0]
            robot_body_paths.update((body0_path, body1_path))
            body0_prim = stage.GetPrimAtPath(body0_path)
            body1_prim = stage.GetPrimAtPath(body1_path)
            if not body0_prim.HasAPI(UsdPhysics.RigidBodyAPI) or not body1_prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                raise RuntimeError(
                    "Physics-v1 adjacent hand filter target is not a rigid body: "
                    f"{body0_path}, {body1_path}"
                )
            filtered_pairs = (
                UsdPhysics.FilteredPairsAPI(body0_prim)
                if body0_prim.HasAPI(UsdPhysics.FilteredPairsAPI)
                else UsdPhysics.FilteredPairsAPI.Apply(body0_prim)
            )
            relation = filtered_pairs.CreateFilteredPairsRel()
            if body1_path not in relation.GetTargets():
                relation.AddTarget(body1_path)
            if body1_path not in relation.GetTargets():
                raise RuntimeError(
                    f"failed to author adjacent hand filter {body0_path} -> {body1_path}"
                )
            robot_prefix = f"{robot_path}/"
            relative0 = str(body0_path).removeprefix(robot_prefix)
            relative1 = str(body1_path).removeprefix(robot_prefix)
            robot_records.append((f"adjacent:{prim.GetName()}", relative0, relative1))
            filtered_body_pairs.add(frozenset((str(body0_path), str(body1_path))))
            authored_count += 1
            adjacent_joint_pair_count += 1

        all_robot_body_paths = sorted(
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        if len(all_robot_body_paths) != 71:
            raise RuntimeError(
                "Physics-v1 robot topology changed; expected 71 rigid bodies, "
                f"found {len(all_robot_body_paths)} below {robot_path}"
            )

        def hand_side(path: str) -> str | None:
            if "/right_hand_base_link" in path:
                return "right"
            if "/left_hand_base_link" in path:
                return "left"
            return None

        def digit_and_link(path: str) -> tuple[str, int] | None:
            name = path.rsplit("/", 1)[-1]
            for digit in ("thumb", "index", "middle", "ring", "little"):
                prefix = f"right_hand_{digit}_link"
                if name.startswith(prefix):
                    return digit, int(name.removeprefix(prefix))
            return None

        def keep_active_hand_collision(body0_path: str, body1_path: str) -> bool:
            if hand_side(body0_path) != "right" or hand_side(body1_path) != "right":
                return False
            name0 = body0_path.rsplit("/", 1)[-1]
            name1 = body1_path.rsplit("/", 1)[-1]
            digit0 = digit_and_link(body0_path)
            digit1 = digit_and_link(body1_path)
            if name0 == "right_hand_base_link" and digit1 is not None:
                return digit1[1] >= 2
            if name1 == "right_hand_base_link" and digit0 is not None:
                return digit0[1] >= 2
            if digit0 is None or digit1 is None:
                return False
            return digit0[0] != digit1[0]

        # Articulation self-collision is a root-wide switch.  The source arm
        # asset was not authored for global self collision and produces solver
        # impulses when every torso/arm pair is enabled.  Physics-v1 therefore
        # isolates self collision to non-adjacent bodies within each hand.
        # This is an explicit first asset layer, not a joint-limit workaround.
        for body0_path, body1_path in combinations(all_robot_body_paths, 2):
            pair = frozenset((body0_path, body1_path))
            if pair in filtered_body_pairs:
                continue
            if keep_active_hand_collision(body0_path, body1_path):
                allowed_nonadjacent_hand_pair_count += 1
                continue
            body0_prim = stage.GetPrimAtPath(body0_path)
            filtered_pairs = (
                UsdPhysics.FilteredPairsAPI(body0_prim)
                if body0_prim.HasAPI(UsdPhysics.FilteredPairsAPI)
                else UsdPhysics.FilteredPairsAPI.Apply(body0_prim)
            )
            relation = filtered_pairs.CreateFilteredPairsRel()
            body1_sdf_path = stage.GetPrimAtPath(body1_path).GetPath()
            if body1_sdf_path not in relation.GetTargets():
                relation.AddTarget(body1_sdf_path)
            filtered_body_pairs.add(pair)
            robot_prefix = f"{robot_path}/"
            robot_records.append(
                (
                    "outside_same_hand_scope",
                    body0_path.removeprefix(robot_prefix),
                    body1_path.removeprefix(robot_prefix),
                )
            )
            authored_count += 1
            scope_isolation_pair_count += 1

        robot_records.sort()
        if expected_pairs_per_robot is None:
            expected_pairs_per_robot = len(robot_records)
        elif len(robot_records) != expected_pairs_per_robot:
            raise RuntimeError(
                "Physics-v1 hand collision filter count differs across clones: "
                f"expected={expected_pairs_per_robot}, actual={len(robot_records)}"
            )
        pair_records.extend(robot_records)
        for body_path in sorted(robot_body_paths):
            body_prim = stage.GetPrimAtPath(body_path)
            reporter = (
                PhysxSchema.PhysxContactReportAPI(body_prim)
                if body_prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                else PhysxSchema.PhysxContactReportAPI.Apply(body_prim)
            )
            reporter.CreateThresholdAttr().Set(0.0)
            reporter_body_count += 1

    if adjacent_joint_pair_count != 54 * env.scene.num_envs:
        raise RuntimeError(
            "Physics-v1 asset topology changed; expected 54 directly connected "
            "hand pairs per robot, found "
            f"{adjacent_joint_pair_count / env.scene.num_envs}"
        )
    if allowed_nonadjacent_hand_pair_count != 286 * env.scene.num_envs:
        raise RuntimeError(
            "Physics-v1 expected 286 enabled task-hand pairs per robot, "
            f"found {allowed_nonadjacent_hand_pair_count / env.scene.num_envs}"
        )
    if expected_pairs_per_robot != 2199:
        raise RuntimeError(
            "Physics-v1 expected 2199 filtered robot pairs per robot, found "
            f"{expected_pairs_per_robot}"
        )
    expected_total = expected_pairs_per_robot * env.scene.num_envs
    if authored_count != expected_total:
        raise RuntimeError(
            f"expected {expected_total} adjacent hand filters, authored {authored_count}"
        )

    unique_records = sorted(set(pair_records))
    matrix_payload = json.dumps(
        unique_records, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")
    env._physics_v1_hand_self_collision_filter = {
        "enabled": True,
        "policy": "right_cross_digit_and_distal_palm_only",
        "pairs_per_robot": expected_pairs_per_robot,
        "authored_pair_count": authored_count,
        "adjacent_joint_pair_count": adjacent_joint_pair_count,
        "scope_isolation_pair_count": scope_isolation_pair_count,
        "allowed_nonadjacent_hand_pair_count": (
            allowed_nonadjacent_hand_pair_count
        ),
        "contact_reporter_body_count": reporter_body_count,
        "unique_relative_pair_count": len(unique_records),
        "pair_matrix_sha256": hashlib.sha256(matrix_payload).hexdigest(),
        "nonadjacent_hand_pairs_enabled": True,
        "active_hand_side": "right",
        "same_digit_nonadjacent_collision": False,
        "cross_digit_collision": True,
        "distal_palm_collision": True,
        "arm_self_collision_deferred_to_simplified_collider_asset": True,
    }


def author_passive_hand_mimics(
    env, env_ids, *, asset_name: str, hand_side: str = "right"
) -> None:
    """Author native PhysX mimics for the task-active L20 hand."""

    if env.sim.is_playing():
        raise RuntimeError("Physics-v1 hand mimics must be authored before physics starts")
    if env_ids is not None:
        raise ValueError("Physics-v1 hand mimics must cover every environment")
    from pxr import PhysxSchema, Usd, UsdPhysics

    from simulation.tomato_pick.isaac.env import (
        MIMIC_FOLLOWER_JOINT_NAMES,
        RIGHT_HAND_MIMIC_RAD,
    )
    from simulation.tomato_pick.physics_v1 import load_physics_v1_profile

    hand = load_physics_v1_profile()["hand"]
    if hand_side not in {"left", "right"}:
        raise ValueError("hand_side must be 'left' or 'right'")
    side_prefix = f"{hand_side}_hand_"
    robot_paths = sim_utils.find_matching_prim_paths(env.scene[asset_name].cfg.prim_path)
    if len(robot_paths) != env.scene.num_envs:
        raise RuntimeError(
            f"expected {env.scene.num_envs} Physics-v1 robots, found {len(robot_paths)}"
        )
    axis_tokens = {
        UsdPhysics.Tokens.x: UsdPhysics.Tokens.rotX,
        UsdPhysics.Tokens.y: UsdPhysics.Tokens.rotY,
        UsdPhysics.Tokens.z: UsdPhysics.Tokens.rotZ,
    }
    authored = 0
    for robot_path in robot_paths:
        root = env.sim.stage.GetPrimAtPath(robot_path)
        joints = {
            prim.GetName(): prim
            for prim in Usd.PrimRange(root)
            if prim.IsA(UsdPhysics.RevoluteJoint)
        }
        for follower_name in MIMIC_FOLLOWER_JOINT_NAMES:
            if not follower_name.startswith(side_prefix):
                continue
            side = (
                "right_hand_"
                if follower_name.startswith("right_hand_")
                else "left_hand_"
            )
            right_name = follower_name.replace("left_hand_", "right_hand_", 1)
            driver_right, multiplier, _legacy_offset = RIGHT_HAND_MIMIC_RAD[right_name]
            driver_name = driver_right.replace("right_hand_", side, 1)
            try:
                driver_prim = joints[driver_name]
                follower_prim = joints[follower_name]
            except KeyError as exc:
                raise RuntimeError(
                    f"Physics-v1 mimic joint was not found below {robot_path}: {exc}"
                ) from exc
            follower_axis = axis_tokens[
                UsdPhysics.RevoluteJoint(follower_prim).GetAxisAttr().Get()
            ]
            driver_axis = axis_tokens[
                UsdPhysics.RevoluteJoint(driver_prim).GetAxisAttr().Get()
            ]
            mimic = PhysxSchema.PhysxMimicJointAPI.Apply(
                follower_prim, follower_axis
            )
            reference = mimic.GetReferenceJointRel()
            reference.ClearTargets(True)
            reference.AddTarget(driver_prim.GetPath())
            mimic.GetReferenceJointAxisAttr().Set(driver_axis)
            # PhysX enforces q_follower + G*q_driver + gamma = 0.
            # Therefore G=-m and gamma=-offset encode
            # q_follower=m*q_driver+offset.  The historical -0.04545 rad
            # software offset requests a negative follower angle at an open
            # hand, which conflicts with the follower's zero lower limit and
            # creates preload.  Physics-v1 starts at a feasible zero offset;
            # the hardware capture must calibrate any real dead band.
            mimic.GetGearingAttr().Set(-multiplier)
            mimic.GetOffsetAttr().Set(-hand["passive_mimic_offset_rad"])
            mimic.CreateNaturalFrequencyAttr().Set(
                hand["passive_mimic_natural_frequency"]
            )
            mimic.CreateDampingRatioAttr().Set(
                hand["passive_mimic_damping_ratio"]
            )
            authored += 1
    expected = 5 * env.scene.num_envs
    if authored != expected:
        raise RuntimeError(f"expected {expected} L20 mimics, authored {authored}")
    env._physics_v1_authored_mimic_count = authored
    env._physics_v1_authored_mimic_side = hand_side


def verify_baked_passive_hand_mimics(
    env, env_ids, *, asset_name: str, hand_side: str = "right"
) -> None:
    """Verify clone-composed native mimics without per-environment authoring."""

    if env.sim.is_playing():
        raise RuntimeError("Physics-v1 baked hand mimics must be verified prestartup")
    if env_ids is not None:
        raise ValueError("Physics-v1 baked mimic verification covers the complete scene")
    if hand_side != "right":
        raise ValueError("Physics-v1 bakes native mimics only for the task-active right hand")
    from pxr import PhysxSchema, Usd, UsdPhysics

    from simulation.tomato_pick.isaac.env import RIGHT_HAND_MIMIC_RAD

    profile = load_physics_v1_profile()
    asset = profile["robot_asset"]
    hand = profile["hand"]
    robot_paths = sim_utils.find_matching_prim_paths(env.scene[asset_name].cfg.prim_path)
    if len(robot_paths) != env.scene.num_envs:
        raise RuntimeError(
            f"expected {env.scene.num_envs} Physics-v1 robots, found {len(robot_paths)}"
        )
    expected_metadata = {
        "physicsV1PassiveMimicPolicy": "right_hand_native_physx_baked",
        "physicsV1PassiveMimicCount": 5,
        "physicsV1PassiveMimicSha256": asset["passive_mimic_sha256"],
    }
    stage = env.sim.stage
    for robot_path in (robot_paths[0], robot_paths[-1]):
        root = stage.GetPrimAtPath(robot_path)
        if not root.IsValid():
            raise RuntimeError(f"Physics-v1 robot clone is invalid: {robot_path}")
        for key, expected in expected_metadata.items():
            actual = root.GetCustomDataByKey(key)
            if actual != expected:
                raise RuntimeError(
                    f"Physics-v1 baked mimic metadata mismatch {key}: "
                    f"expected={expected!r}, actual={actual!r}"
                )

    first_root = stage.GetPrimAtPath(robot_paths[0])
    joints = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(first_root)
        if prim.IsA(UsdPhysics.RevoluteJoint)
    }
    axis_tokens = {
        UsdPhysics.Tokens.x: UsdPhysics.Tokens.rotX,
        UsdPhysics.Tokens.y: UsdPhysics.Tokens.rotY,
        UsdPhysics.Tokens.z: UsdPhysics.Tokens.rotZ,
    }
    verified = 0
    for follower_name, (driver_name, multiplier, _legacy_offset) in (
        RIGHT_HAND_MIMIC_RAD.items()
    ):
        follower = joints[follower_name]
        driver = joints[driver_name]
        follower_axis = axis_tokens[
            UsdPhysics.RevoluteJoint(follower).GetAxisAttr().Get()
        ]
        driver_axis = axis_tokens[
            UsdPhysics.RevoluteJoint(driver).GetAxisAttr().Get()
        ]
        if not follower.HasAPI(PhysxSchema.PhysxMimicJointAPI, follower_axis):
            raise RuntimeError(f"baked mimic API missing on {follower.GetPath()}")
        mimic = PhysxSchema.PhysxMimicJointAPI(follower, follower_axis)
        if mimic.GetReferenceJointRel().GetTargets() != [driver.GetPath()]:
            raise RuntimeError(f"baked mimic driver mismatch for {follower_name}")
        values = {
            "axis": mimic.GetReferenceJointAxisAttr().Get() == driver_axis,
            "gearing": abs(float(mimic.GetGearingAttr().Get()) + multiplier) < 1.0e-6,
            "offset": abs(
                float(mimic.GetOffsetAttr().Get())
                + float(hand["passive_mimic_offset_rad"])
            )
            < 1.0e-6,
            "natural_frequency": abs(
                float(mimic.GetNaturalFrequencyAttr().Get())
                - float(hand["passive_mimic_natural_frequency"])
            )
            < 1.0e-6,
            "damping_ratio": abs(
                float(mimic.GetDampingRatioAttr().Get())
                - float(hand["passive_mimic_damping_ratio"])
            )
            < 1.0e-6,
        }
        if not all(values.values()):
            raise RuntimeError(f"baked mimic parameters mismatch {follower_name}: {values}")
        verified += 1
    if verified != 5:
        raise RuntimeError(f"expected five baked right-hand mimics, found {verified}")
    env._physics_v1_authored_mimic_count = verified * env.scene.num_envs
    env._physics_v1_authored_mimic_side = hand_side
    env._physics_v1_mimics_baked = True
    env._physics_v1_runtime_authored_mimic_count = 0


def bind_hand_physics_material(env, env_ids, *, asset_name: str) -> None:
    """Bind the Physics-v1 pad material above instanced collision geometry.

    The LinkBot USD uses instance proxies for visual/collision meshes.  Isaac
    Lab's generic recursive binder cannot author on those proxies.  A physics
    material binding on each non-instanced rigid-body parent is inherited by
    the collision shapes and leaves the source USD untouched.
    """

    if env.sim.is_playing():
        raise RuntimeError("Physics-v1 hand material must be bound before physics starts")
    if env_ids is not None:
        raise ValueError("Physics-v1 hand material binding must cover every environment")
    from pxr import Usd, UsdPhysics, UsdShade

    from isaaclab.sim.utils.stage import get_current_stage
    from simulation.tomato_pick.physics_v1 import load_physics_v1_profile

    stage = get_current_stage()
    contact = load_physics_v1_profile()["contact"]
    material_path = "/World/PhysicsMaterials/PhysicsV1FingerPad"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=contact["finger_static_friction"],
        dynamic_friction=contact["finger_dynamic_friction"],
        restitution=0.0,
        friction_combine_mode=contact["friction_combine_mode"],
        restitution_combine_mode=contact["restitution_combine_mode"],
    )
    material_cfg.func(material_path, material_cfg)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not material:
        raise RuntimeError("Physics-v1 finger-pad material was not created")

    robot_paths = sim_utils.find_matching_prim_paths(env.scene[asset_name].cfg.prim_path)
    if len(robot_paths) != env.scene.num_envs:
        raise RuntimeError(
            f"expected {env.scene.num_envs} Physics-v1 robots, found {len(robot_paths)}"
        )
    bound_bodies = 0
    inherited_colliders = 0
    for robot_path in robot_paths:
        root = stage.GetPrimAtPath(robot_path)
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or "_hand_" not in prim.GetName():
                continue
            binding = (
                UsdShade.MaterialBindingAPI(prim)
                if prim.HasAPI(UsdShade.MaterialBindingAPI)
                else UsdShade.MaterialBindingAPI.Apply(prim)
            )
            binding.Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            bound_bodies += 1
            for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                inherited = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial(
                    "physics"
                )[0]
                if inherited and inherited.GetPath() == material.GetPath():
                    inherited_colliders += 1
    if bound_bodies == 0 or inherited_colliders == 0:
        raise RuntimeError(
            "Physics-v1 finger-pad material did not reach hand collision shapes: "
            f"bound_bodies={bound_bodies}, inherited_colliders={inherited_colliders}"
        )
    env._physics_v1_hand_material_binding = {
        "material_path": material_path,
        "bound_body_count": bound_bodies,
        "inherited_collider_count": inherited_colliders,
    }


def verify_baked_hand_physics_material(env, env_ids, *, asset_name: str) -> None:
    """Verify the asset-level pad material on one clone and constant metadata."""

    if env.sim.is_playing():
        raise RuntimeError("Physics-v1 baked hand material must be verified prestartup")
    if env_ids is not None:
        raise ValueError("Physics-v1 baked hand material verification covers the scene")
    from pxr import Usd, UsdPhysics, UsdShade

    profile = load_physics_v1_profile()
    asset = profile["robot_asset"]
    contact = profile["contact"]
    robot_paths = sim_utils.find_matching_prim_paths(env.scene[asset_name].cfg.prim_path)
    if len(robot_paths) != env.scene.num_envs:
        raise RuntimeError(
            f"expected {env.scene.num_envs} Physics-v1 robots, found {len(robot_paths)}"
        )
    expected_metadata = {
        "physicsV1HandMaterialPolicy": "both_hands_parent_binding_baked",
        "physicsV1HandMaterialBodyCount": asset["hand_material_body_count"],
    }
    stage = env.sim.stage
    for robot_path in (robot_paths[0], robot_paths[-1]):
        root = stage.GetPrimAtPath(robot_path)
        for key, expected in expected_metadata.items():
            actual = root.GetCustomDataByKey(key)
            if actual != expected:
                raise RuntimeError(
                    f"Physics-v1 baked hand material metadata mismatch {key}: "
                    f"expected={expected!r}, actual={actual!r}"
                )

    first_root = stage.GetPrimAtPath(robot_paths[0])
    material_path = first_root.GetPath().AppendPath(
        "PhysicsMaterials/PhysicsV1FingerPad"
    )
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not material:
        raise RuntimeError(f"baked Physics-v1 hand material missing: {material_path}")
    material_api = UsdPhysics.MaterialAPI(material.GetPrim())
    material_values = {
        "static_friction": abs(
            float(material_api.GetStaticFrictionAttr().Get())
            - float(contact["finger_static_friction"])
        )
        < 1.0e-6,
        "dynamic_friction": abs(
            float(material_api.GetDynamicFrictionAttr().Get())
            - float(contact["finger_dynamic_friction"])
        )
        < 1.0e-6,
        "restitution": abs(float(material_api.GetRestitutionAttr().Get())) < 1.0e-6,
    }
    if not all(material_values.values()):
        raise RuntimeError(f"baked hand material parameters mismatch: {material_values}")
    bound_bodies = 0
    inherited_colliders = 0
    for prim in Usd.PrimRange(first_root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or "_hand_" not in prim.GetName():
            continue
        inherited = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
        if not inherited or inherited.GetPath() != material.GetPath():
            raise RuntimeError(f"baked hand material binding missing on {prim.GetPath()}")
        bound_bodies += 1
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collider_material = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial(
                "physics"
            )[0]
            if collider_material and collider_material.GetPath() == material.GetPath():
                inherited_colliders += 1
    if bound_bodies != asset["hand_material_body_count"] or inherited_colliders == 0:
        raise RuntimeError(
            "baked hand material coverage mismatch: "
            f"bodies={bound_bodies}, colliders={inherited_colliders}"
        )
    env._physics_v1_hand_material_binding = {
        "material_path": str(material_path),
        "bound_body_count": bound_bodies * env.scene.num_envs,
        "inherited_collider_count": inherited_colliders * env.scene.num_envs,
        "asset_baked": True,
        "runtime_authored_binding_count": 0,
    }


def author_correlated_fruit_profiles(env, env_ids, *, asset_name: str) -> None:
    """Author one correlated, episode-stable fruit profile per environment."""

    if env.sim.is_playing():
        raise RuntimeError("Physics-v1 fruit profiles must be authored before physics starts")
    if env_ids is not None:
        raise ValueError("Physics-v1 prestartup profile must cover every environment")
    profile = load_physics_v1_profile()
    contact = profile["contact"]
    randomization = profile["randomization"]
    profiles = sample_correlated_fruit_profiles(env.scene.num_envs)
    contact_profiles = quantize_contact_profiles(
        profiles,
        friction_bins=randomization["contact_friction_bins"],
        compliance_bins=randomization["contact_compliance_bins"],
    )
    axes = profiles["axes_m"]

    prim_paths = sim_utils.find_matching_prim_paths(env.scene[asset_name].cfg.prim_path)
    if len(prim_paths) != env.scene.num_envs:
        raise RuntimeError(
            f"expected {env.scene.num_envs} Physics-v1 tomato prims, found {len(prim_paths)}"
        )
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = env.sim.stage
    material_bank: dict[int, UsdShade.Material] = {}
    for index in range(env.scene.num_envs):
        material_bin = int(contact_profiles["material_bin"][index])
        if material_bin in material_bank:
            continue
        friction_bin = int(contact_profiles["friction_bin"][index])
        compliance_bin = int(contact_profiles["compliance_bin"][index])
        material_path = (
            "/World/PhysicsMaterials/"
            f"PhysicsV1Tomato_f{friction_bin}_c{compliance_bin}"
        )
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(
                contact_profiles["material_static_friction"][index]
            ),
            dynamic_friction=float(
                contact_profiles["material_dynamic_friction"][index]
            ),
            restitution=contact["restitution"],
            friction_combine_mode=contact["friction_combine_mode"],
            restitution_combine_mode=contact["restitution_combine_mode"],
            compliant_contact_stiffness=float(
                contact_profiles["material_stiffness_n_m"][index]
            ),
            compliant_contact_damping=float(
                contact_profiles["material_damping_n_s_m"][index]
            ),
        )
        material_cfg.func(material_path, material_cfg)
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        if not material:
            raise RuntimeError(
                f"Physics-v1 tomato material was not created: {material_path}"
            )
        material_bank[material_bin] = material

    bound_fruit_count = 0
    inherited_collider_count = 0
    for index, prim_path in enumerate(prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        scale_op = next(
            (
                op
                for op in xformable.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeScale
            ),
            None,
        )
        if scale_op is None:
            scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3d(*(float(value) for value in axes[index])))
        mesh = stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")
        UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Set("convexHull")
        material = material_bank[int(contact_profiles["material_bin"][index])]
        binding = (
            UsdShade.MaterialBindingAPI(prim)
            if prim.HasAPI(UsdShade.MaterialBindingAPI)
            else UsdShade.MaterialBindingAPI.Apply(prim)
        )
        binding.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        bound_fruit_count += 1
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                continue
            inherited = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial(
                "physics"
            )[0]
            if inherited and inherited.GetPath() == material.GetPath():
                inherited_collider_count += 1

    if bound_fruit_count != env.scene.num_envs:
        raise RuntimeError(
            "Physics-v1 did not bind every tomato profile: "
            f"bound={bound_fruit_count}, expected={env.scene.num_envs}"
        )
    if inherited_collider_count < env.scene.num_envs:
        raise RuntimeError(
            "Physics-v1 tomato materials did not reach all collision shapes: "
            f"inherited={inherited_collider_count}, expected>={env.scene.num_envs}"
        )

    env._prestartup_fruit_axes_m = axes
    env._prestartup_fruit_density_kg_m3 = profiles["density_kg_m3"]
    env._prestartup_physics_v1_profiles = profiles
    env._prestartup_contact_material_profiles = contact_profiles
    env._physics_v1_fruit_material_binding = {
        "material_count": len(material_bank),
        "bound_fruit_count": bound_fruit_count,
        "inherited_collider_count": inherited_collider_count,
    }


__all__ = [
    "author_correlated_fruit_profiles",
    "author_hand_self_collision_filters",
    "author_passive_hand_mimics",
    "bind_hand_physics_material",
    "verify_baked_hand_physics_material",
    "verify_baked_passive_hand_mimics",
    "verify_baked_hand_self_collision_filters",
]
