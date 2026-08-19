"""V2 palm-supported five-finger enclosure environment.

The scene and low-level safety envelope come from the frozen v1 environment;
the adapter changes the policy contract, action temporal filtering and reward
semantics.  In particular, the v2 stage-2 reward has no positive three-finger
``secure_now`` term and the distal residuals are rewarded only together with
measured contact.
"""

from __future__ import annotations

import torch
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm

from simulation.ppo_tomato_grasp.physx_tomato_env import PhysxTomatoGraspEnv
from simulation.ppo_tomato_grasp.physx_enveloping_env import EnvelopingPhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp_v2.contract import V2_ACTOR_OBSERVATION_DIM, V2_CRITIC_STATE_DIM
from simulation.ppo_tomato_grasp_v2.action_filter import filter_distal_residual
from simulation.ppo_tomato_grasp_v2.normalization import normalize_semi_axes
from simulation.ppo_tomato_grasp_v2.snapshot_env import TerminalSnapshotEnvelopingEnv
from simulation.ppo_tomato_grasp_v2.collision_proxies import author_fingertip_collision_proxies


@configclass
class V2EnvelopingPhysxTomatoGraspEnvCfg(EnvelopingPhysxTomatoGraspEnvCfg):
    """V2 contract and conservative reward defaults."""

    observation_space = V2_ACTOR_OBSERVATION_DIM
    state_space = V2_CRITIC_STATE_DIM
    action_space = 14
    # Expose the accepted mass prior as a first-class v2 field.  The base
    # event also receives this value, but keeping one config source makes
    # matrix reports and checkpoint snapshots reproducible.
    accepted_mass_kg = (0.050, 0.250)
    density_range_kg_m3 = (700.0, 1050.0)
    # The v2 contact probe found proximal link3 contact while link4 tips stayed
    # 4--6 cm from the fruit under the old 0.25 residual span.  Increase only
    # the simulation enclosure adapter's distal reach; every target remains
    # clamped to LEFT_SAFE_* bounds and the archived 10D task is untouched.
    distal_residual_span_fraction = 0.50
    # Optional v2-only widening for index/middle/ring/little root residuals.
    # ``None`` preserves the historical 10D/14D mapper exactly.  A non-None
    # value is an uncalibrated simulation reach prior and must be justified
    # by the kinematic primitive audit before any PPO run uses it.
    per_digit_root_residual_span_fraction = None
    # The imported USD link4 collision boxes are only ~2 mm.  Add an explicit
    # 4 mm-radius sim-only fingertip pad so measured distal contact represents
    # a plausible finger pad rather than a kinematic fallback.  This remains
    # disabled from all hardware paths and must be calibrated before sim2real.
    fingertip_collision_proxy_enabled = True
    fingertip_collision_proxy_radius_m = 0.008
    # The USD fingertip collision meshes are uncalibrated.  Keep the authored
    # proxy visible to the simulator for geometry inspection, but use a
    # distance-based tactile surrogate by default so a large proxy cannot
    # launch the fruit during exploration.  Hardware remains untouched.
    fingertip_collision_proxy_physical = False
    # Diagnostic-only isolation mode: when enabled, the base hand collision
    # meshes are disabled and only the authored v2 fingertip spheres remain
    # physical (along with the palm support).  This is useful for separating
    # proxy geometry from uncalibrated vendor link4 collision boxes; it is not
    # a final sim2real setting and is never propagated to hardware.
    proxy_only_collision = False
    virtual_distal_contact_enabled = True
    virtual_distal_contact_gain_n_per_m = 45.0
    virtual_distal_contact_margin_m = 0.001
    # Audited runs can retain the virtual pad for dense reach shaping while
    # requiring the terminal gate to use the raw ContactSensor latch.
    strict_require_measured_distal_contact = False
    # P3 geometry sweep knobs.  They only alter the simulation collision
    # authoring and are persisted in the resolved checkpoint config; they are
    # never copied into a hardware command path.
    tomato_contact_offset_m = 0.002
    tomato_rest_offset_m = 0.0
    fingertip_proxy_contact_offset_m = 0.002
    fingertip_proxy_rest_offset_m = 0.0
    # New audited branch: keep distal residuals out of free space until the
    # previous policy step has observed palm support.  This makes the intended
    # sequence explicit (fruit settles on palm, then fingers envelope) rather
    # than allowing distal shaping to push the fruit away from the support.
    palm_first_distal_gate = False
    palm_first_coarse_scale = 1.0
    palm_first_coarse_max_scale = 1.0
    palm_first_stable_steps = 8
    palm_first_ramp_steps = 60
    # Optional P2 curriculum: release each straight-finger distal residual
    # only after that finger's own link3 has carried measured contact for a
    # short dwell.  Disabled by default so archived v2 branches are bitwise
    # comparable; enabled explicitly for the staged distal-contact branch.
    per_digit_distal_gate = False
    per_digit_distal_link3_threshold_n = 0.04
    per_digit_distal_link3_stable_steps = 4
    per_digit_distal_min_episode_step = 0
    # Isaac Sim 4.5 may expose no usable link3 force for the imported hand.
    # An explicitly opted-in near-surface gate keeps distal release causal:
    # palm support + episode dwell + that digit's fingertip entering the
    # fruit shell.  It is not a contact/success signal.
    per_digit_distal_near_surface_enabled = False
    per_digit_distal_near_surface_distance_m = 0.043
    per_digit_distal_near_surface_min_episode_step = 20
    # Optional simulation-only contact bootstrap.  A small positive distal
    # residual is offered only after palm support and only while a finger is
    # within a bounded reach corridor.  It is a learnable exploration aid,
    # not a terminal pose or a hardware command.
    per_digit_distal_probe_enabled = False
    per_digit_distal_probe_distance_m = 0.064
    per_digit_distal_probe_action = 0.14
    per_digit_distal_probe_action_per_digit = None
    per_digit_distal_probe_start_episode_step = 20
    per_digit_distal_probe_ramp_steps = 80
    # Optional staggered ramp: index/middle/ring/little receive the bounded
    # distal bootstrap with a small temporal offset instead of all four
    # loading at once.  This is a simulation-only exploration aid intended to
    # reduce the single-step object impulse while preserving independent
    # fingertip participation; it is never a success or hardware signal.
    per_digit_distal_probe_staggered = False
    per_digit_distal_probe_digit_interval_steps = 20
    virtual_palm_support_enabled = True
    virtual_palm_support_margin_m = 0.002
    virtual_palm_support_gain_n_per_m = 300.0
    # The 14-D manifold's distal residuals originally moved only link4.  On
    # the imported hand that body is too short to reach a large fruit unless
    # the corresponding finger's near/proximal joint advances with it.  This
    # bounded v2-only assist couples each straight-finger distal residual to
    # its own residual channel (index..little) only while the fingertip is
    # approaching the fruit and no distal load has been measured.  It is an
    # action-space reach aid, not a fixed pose or a hardware command.
    distal_proximal_reach_assist_gain = 0.30
    distal_proximal_reach_assist_gain_per_digit = (0.45, 0.45, 0.30, 0.25)
    distal_proximal_reach_assist_distance_m = 0.070
    # Rate-limit the coarse action channels to avoid a one-step PhysX impulse
    # when the policy transitions from open hand to the enclosure corridor.
    # The archived environment keeps its historical direct targets.
    coarse_action_smoothing_alpha = 1.0
    # v2 dense terms are gated by valid in-hand support.  They make the
    # five-finger topology (including the thumb) materially preferable to a
    # low-force four-finger/three-finger shortcut while retaining the strict
    # terminal verifier as the only success authority.
    v2_all_five_palm_reward = 6.0
    v2_all_four_distal_reward = 3.0
    v2_thumb_with_distal_reward = 2.5
    v2_in_hand_support_reward = 1.5
    # Optional measured-only shaping.  Unlike the virtual distal channel,
    # this term is sourced from the raw link4 ContactSensors and is therefore
    # useful for escaping a virtual-contact local optimum.  It remains a
    # dense per-digit diagnostic reward; the strict terminal gate is still
    # the authority for success.
    v2_measured_distal_contact_reward = 0.0
    # Continuous raw link4 load shaping, normalized by the contact threshold
    # rather than the broad safety ceiling.  This preserves a small learning
    # signal below the binary contact gate without treating virtual load as
    # measured tactile feedback.
    v2_measured_distal_force_reward = 0.0
    # P1 trust-region shaping.  These terms are only an additional pre-pick
    # safety signal; they do not relax the measured-contact success gate.
    # The inherited environment already has weak jitter/centre penalties, but
    # the audited trace showed long-lived action saturation after first palm
    # contact, so v2 makes the failure mode explicit and measurable.
    v2_action_saturation_penalty = 0.40
    v2_action_delta_penalty = 0.20
    v2_center_offset_penalty = 1.25
    v2_center_velocity_penalty = 1.00
    v2_force_rate_penalty = 0.03
    # After palm support is established, damp only fresh coarse/root motion
    # when the fruit is already moving out of the palm.  Distal tactile
    # channels remain available; this is a reversible simulation safety
    # envelope rather than a success shortcut.
    palm_first_motion_hold_enabled = False
    palm_first_motion_hold_scale = 0.25
    palm_first_motion_hold_speed_m_s = 0.08
    palm_first_motion_hold_offset_ratio = 0.55
    def __post_init__(self) -> None:
        # ``events`` is a mutable inherited config field.  It is not available
        # as a class attribute after IsaacLab's configclass dataclass wrapper,
        # so append the v2 event on each fresh config instance.
        self.events["v2_fingertip_collision_proxies"] = EventTerm(
            func=author_fingertip_collision_proxies,
            mode="prestartup",
            params={
                "radius_m": float(self.fingertip_collision_proxy_radius_m),
                "local_xyz": (0.0, 0.0, 0.0),
                "physical_collision": bool(self.fingertip_collision_proxy_physical),
                "contact_offset_m": float(self.fingertip_proxy_contact_offset_m),
                "rest_offset_m": float(self.fingertip_proxy_rest_offset_m),
            },
        )
    # A slower distal command filter lets coarse/root positioning happen
    # quickly while fingertips settle against the fruit instead of oscillating.
    distal_action_filter_alpha = 0.70
    # Stage-2 three-finger support is a diagnostic intermediate event only.
    secure_reward_weight = 0.0
    post_break_support_reward_weight = 0.0
    post_break_contact_reward_weight = 1.0
    post_break_missing_digit_penalty_weight = 2.5
    strict_reward_weight = 2.5
    strict_distal_reward_weight = 1.5
    strict_hold_steps = 20
    strict_palm_first_margin_steps = 4
    # v2-only exploration aid: give the palm and root joints a bounded
    # pregrasp window before height-based drop termination.  Success still
    # requires the strict five-finger/palm gate; this only prevents the
    # initial support transient from making the distal action unreachable.
    pregrasp_drop_grace_steps = 90
    # New profile code sets this explicitly; keep v1's stage-1 object prior
    # as a safe fallback when constructing the config interactively.
    v2_profile_name = "v2_band1_match"


class V2EnvelopingPhysxTomatoGraspEnv(TerminalSnapshotEnvelopingEnv):
    """V2 observation/action/reward adapter over the validated PhysX scene."""

    cfg: V2EnvelopingPhysxTomatoGraspEnvCfg

    def __init__(self, cfg: V2EnvelopingPhysxTomatoGraspEnvCfg, render_mode: str | None = None, **kwargs):
        # Configclass fields can be overridden after ``__post_init__`` (the
        # CLI/probe uses this for the physical fingertip audit).  Refresh the
        # event parameters here so the authored USD actually receives the
        # resolved radius and collision-enabled flag; otherwise the event
        # would silently keep the class-default ``False`` value.
        proxy_event = cfg.events.get("v2_fingertip_collision_proxies")
        if proxy_event is not None:
            proxy_event.params["radius_m"] = float(cfg.fingertip_collision_proxy_radius_m)
            proxy_event.params["physical_collision"] = bool(cfg.fingertip_collision_proxy_physical)
            proxy_event.params["contact_offset_m"] = float(cfg.fingertip_proxy_contact_offset_m)
            proxy_event.params["rest_offset_m"] = float(cfg.fingertip_proxy_rest_offset_m)
        tomato_event = cfg.events.get("tomato_geometry")
        if tomato_event is not None:
            tomato_event.params["accepted_mass_kg"] = tuple(float(v) for v in cfg.accepted_mass_kg)
            tomato_event.params["contact_offset_m"] = float(cfg.tomato_contact_offset_m)
            tomato_event.params["rest_offset_m"] = float(cfg.tomato_rest_offset_m)
        self._v2_distal_action = None
        self._v2_snapshot_enabled = False
        super().__init__(cfg, render_mode, **kwargs)
        self._v2_contact_age = torch.zeros(self.num_envs, 5, device=self.device)
        # The base v1 bookkeeping selected ``thumb_link4`` for the thumb
        # channel, while the imported hand's actual fingertip/contact sensor
        # is ``thumb_link5``.  Keep this correction v2-only so archived v1
        # checkpoints and their contract remain untouched.
        self._sensor_groups = (
            ("thumb_contact", "thumb_contact"),
            ("index_contact", "index_contact_link3"),
            ("middle_contact", "middle_contact_link3"),
            ("ring_contact", "ring_contact_link3"),
            ("little_contact", "little_contact_link3"),
        )
        self._v2_lost_contact_age = torch.zeros(self.num_envs, 5, device=self.device)
        self._v2_distal_action = torch.zeros(self.num_envs, 4, device=self.device)
        self._v2_first_contact_step = torch.full((self.num_envs, 5), -1, dtype=torch.long, device=self.device)
        self._v2_contact_steps = torch.zeros(self.num_envs, 5, device=self.device)
        self._v2_pre_break_contact_steps = torch.zeros(self.num_envs, 5, device=self.device)
        self._v2_post_break_contact_steps = torch.zeros(self.num_envs, 5, device=self.device)
        self._v2_force_integral = torch.zeros(self.num_envs, 5, device=self.device)
        self._v2_max_lost_contact_gap = torch.zeros(self.num_envs, 5, device=self.device)
        self._v2_link3_force = torch.zeros(self.num_envs, 5, device=self.device)
        # Keep raw ContactSensor output separate from the simulation-only
        # fingertip proxy.  The latter is useful for reachability shaping, but
        # it must never be reported as measured tactile contact.
        self._v2_measured_link4_force = torch.zeros(self.num_envs, 5, device=self.device)
        self._v2_link4_force = torch.zeros(self.num_envs, 5, device=self.device)
        self._v2_measured_palm_force = torch.zeros(self.num_envs, device=self.device)
        # Keep terminal-safe peak diagnostics separate from the instantaneous
        # palm channel.  The strict termination path uses the peak contact
        # buffers, so a terminal snapshot must expose the same evidence when
        # a high-mass reset causes a drop/overload before the next policy step.
        self._v2_peak_palm_force = torch.zeros(self.num_envs, device=self.device)
        self._v2_distal_contact_observed = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._v2_measured_distal_contact_observed = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._v2_measured_link4_force_integral = torch.zeros(self.num_envs, 4, device=self.device)
        self._v2_measured_distal_contact_steps = torch.zeros(self.num_envs, 4, device=self.device)
        self._v2_measured_distal_contact_streak = torch.zeros(self.num_envs, 4, device=self.device)
        self._v2_measured_distal_max_streak = torch.zeros(self.num_envs, 4, device=self.device)
        self._v2_palm_support_age = torch.zeros(self.num_envs, device=self.device)
        self._v2_link3_support_age = torch.zeros(self.num_envs, 4, device=self.device)
        self.register_snapshot_fields({
            "first_contact_step": ("_v2_first_contact_step", torch.long),
            "contact_steps": ("_v2_contact_steps", torch.float32),
            "pre_break_contact_steps": ("_v2_pre_break_contact_steps", torch.float32),
            "post_break_contact_steps": ("_v2_post_break_contact_steps", torch.float32),
            "force_integral": ("_v2_force_integral", torch.float32),
            "max_lost_contact_gap": ("_v2_max_lost_contact_gap", torch.float32),
            "link3_force": ("_v2_link3_force", torch.float32),
            "measured_link4_force": ("_v2_measured_link4_force", torch.float32),
            "link4_force": ("_v2_link4_force", torch.float32),
            "measured_palm_force": ("_v2_measured_palm_force", torch.float32),
            "peak_palm_force": ("_v2_peak_palm_force", torch.float32),
            "distal_contact_observed": ("_v2_distal_contact_observed", torch.bool),
            "measured_distal_contact_observed": ("_v2_measured_distal_contact_observed", torch.bool),
            "measured_link4_force_integral": ("_v2_measured_link4_force_integral", torch.float32),
            "measured_distal_contact_steps": ("_v2_measured_distal_contact_steps", torch.float32),
            "measured_distal_max_streak": ("_v2_measured_distal_max_streak", torch.float32),
            "link3_support_age": ("_v2_link3_support_age", torch.float32),
            "strict_opposition_min": ("strict_opposition_min", torch.float32),
            "strict_hemisphere_max": ("strict_hemisphere_max", torch.float32),
            "strict_coverage_current": ("strict_coverage_current", torch.float32),
        })
        self._v2_snapshot_enabled = True
        self._reset_idx(self.robot._ALL_INDICES)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_v2_contact_age"):
            return
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device)
        # Reset hooks are invoked from Isaac Lab's inference context.  Clone
        # the bookkeeping buffers once so later partial resets are mutable
        # tensors when PPO is running outside inference mode.
        with torch.inference_mode(False):
            self._v2_contact_age = self._v2_contact_age.clone()
            self._v2_lost_contact_age = self._v2_lost_contact_age.clone()
            self._v2_distal_action = self._v2_distal_action.clone()
            self._v2_first_contact_step = self._v2_first_contact_step.clone()
            self._v2_contact_steps = self._v2_contact_steps.clone()
            self._v2_pre_break_contact_steps = self._v2_pre_break_contact_steps.clone()
            self._v2_post_break_contact_steps = self._v2_post_break_contact_steps.clone()
            self._v2_force_integral = self._v2_force_integral.clone()
            self._v2_max_lost_contact_gap = self._v2_max_lost_contact_gap.clone()
            self._v2_link3_force = self._v2_link3_force.clone()
            self._v2_measured_link4_force = self._v2_measured_link4_force.clone()
            self._v2_link4_force = self._v2_link4_force.clone()
            self._v2_measured_palm_force = self._v2_measured_palm_force.clone()
            self._v2_peak_palm_force = self._v2_peak_palm_force.clone()
            self._v2_distal_contact_observed = self._v2_distal_contact_observed.clone()
            self._v2_measured_distal_contact_observed = self._v2_measured_distal_contact_observed.clone()
            self._v2_measured_link4_force_integral = self._v2_measured_link4_force_integral.clone()
            self._v2_measured_distal_contact_steps = self._v2_measured_distal_contact_steps.clone()
            self._v2_measured_distal_contact_streak = self._v2_measured_distal_contact_streak.clone()
            self._v2_measured_distal_max_streak = self._v2_measured_distal_max_streak.clone()
            self._v2_palm_support_age = self._v2_palm_support_age.clone()
            self._v2_link3_support_age = self._v2_link3_support_age.clone()
            self._v2_contact_age[ids] = 0.0
            self._v2_lost_contact_age[ids] = 0.0
            self._v2_distal_action[ids] = 0.0
            self._v2_first_contact_step[ids] = -1
            self._v2_contact_steps[ids] = 0.0
            self._v2_pre_break_contact_steps[ids] = 0.0
            self._v2_post_break_contact_steps[ids] = 0.0
            self._v2_force_integral[ids] = 0.0
            self._v2_max_lost_contact_gap[ids] = 0.0
            self._v2_link3_force[ids] = 0.0
            self._v2_measured_link4_force[ids] = 0.0
            self._v2_link4_force[ids] = 0.0
            self._v2_measured_palm_force[ids] = 0.0
            self._v2_peak_palm_force[ids] = 0.0
            self._v2_distal_contact_observed[ids] = False
            self._v2_measured_distal_contact_observed[ids] = False
            self._v2_measured_link4_force_integral[ids] = 0.0
            self._v2_measured_distal_contact_steps[ids] = 0.0
            self._v2_measured_distal_contact_streak[ids] = 0.0
            self._v2_measured_distal_max_streak[ids] = 0.0
            self._v2_palm_support_age[ids] = 0.0
            self._v2_link3_support_age[ids] = 0.0

    @torch.inference_mode()
    def _update_strict_geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Use measured link4 contact, not joint excursion, for final readiness.

        The inherited verifier's excursion flag is retained as a diagnostic
        for proximal-only motion, but it can be satisfied in free space.  V2
        requires every straight finger's measured link4 channel to have
        crossed the contact threshold at least once in the episode.  The
        parent method still updates the common palm/coverage/hold buffers; its
        provisional latch is replaced by this stricter measured-contact latch.
        """
        previous_latched = self.strict_success_latched.clone()
        self.strict_success_latched.zero_()
        strict_now, _ = super()._update_strict_geometry()
        if self.cfg.strict_require_measured_distal_contact:
            measured_distal_ready = self._v2_measured_distal_contact_observed.all(-1)
        else:
            measured_distal_ready = self._v2_distal_contact_observed.all(-1)
        movement_ready = self.strict_distal_participation.all(-1)
        margin = self.strict_first_three - self.strict_first_palm
        # A pregrasp drop grace is only an exploration aid.  It must never
        # turn a fruit resting on the floor or outside the palm work area into
        # a strict enclosure.  Reapply the physical in-hand/height condition
        # after the inherited geometry update and clear the hold streak when
        # it is violated.
        tomato_pos = self.tomato.data.root_pos_w
        in_hand = torch.linalg.vector_norm(tomato_pos - self.grasp_center, dim=-1) <= self.cfg.max_center_distance_m
        supported_height = tomato_pos[:, 2] >= self.cfg.drop_height_m
        valid_support = in_hand & supported_height
        strict_now &= valid_support
        self.strict_hold_streak = torch.where(
            valid_support,
            self.strict_hold_streak,
            torch.zeros_like(self.strict_hold_streak),
        )
        ready = (
            (self.strict_first_palm >= 0)
            & (self.strict_first_three >= 0)
            & (margin >= self.cfg.strict_palm_first_margin_steps)
            & (self.strict_hold_streak >= self.cfg.strict_hold_steps)
            & measured_distal_ready
            & movement_ready
            & ~self.strict_proximal_only
            & ~self.dropped
            & valid_support
        )
        if self.cfg.strict_enveloping_require_stem_break:
            ready &= self.stem_broke
        newly = ready & ~previous_latched
        self.strict_success_latched = previous_latched | ready
        return strict_now, newly

    @torch.inference_mode()
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        bounded = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        alpha_coarse = float(getattr(self.cfg, "coarse_action_smoothing_alpha", 1.0))
        alpha_coarse = max(0.0, min(1.0, alpha_coarse))
        if alpha_coarse < 1.0 and hasattr(self, "actions") and self.actions.shape == bounded.shape:
            bounded = self.actions + alpha_coarse * (bounded - self.actions)
        # The evaluator may use a per-digit tuple to slow the already stable
        # digits while releasing a geometrically marginal finger faster.
        # Keep scalar configs unchanged and pass vectors through to the pure
        # filter helper for broadcast-safe GPU arithmetic.
        alpha = getattr(self.cfg, "distal_action_filter_alpha", 0.70)
        if isinstance(alpha, (tuple, list)):
            alpha = bounded.new_tensor(tuple(float(value) for value in alpha))
        filtered = filter_distal_residual(self._v2_distal_action, bounded, alpha=alpha)
        filtered_distal = filtered[..., 10:]
        # Do not spend the distal residual in free space.  Coarse/root
        # channels first bring each straight fingertip near the fruit; only
        # then is the corresponding residual released.  Contact alone can be
        # sparse in Isaac Sim 4.5, so a bounded kinematic near-surface gate is
        # retained as an explicit diagnostic rather than silently rewarding
        # arbitrary distal motion.
        tips = self._tip_positions()[:, 1:]
        distance = torch.linalg.vector_norm(
            tips - self.tomato.data.root_pos_w.unsqueeze(1), dim=-1
        )
        near_surface = distance < float(
            getattr(self.cfg, "per_digit_distal_near_surface_distance_m", 0.045)
        )
        measured_contact = self.contact_forces[:, 1:] >= self.cfg.contact_threshold_n
        distal_gate = (near_surface | measured_contact).float()
        filtered_distal = filtered_distal * distal_gate
        palm_age = getattr(self, "_v2_palm_support_age", torch.zeros(self.num_envs, device=self.device))
        stable_steps = float(getattr(self.cfg, "palm_first_stable_steps", 1))
        if bool(getattr(self.cfg, "palm_first_distal_gate", False)):
            palm_ready = (palm_age >= stable_steps).unsqueeze(-1)
            filtered_distal = filtered_distal * palm_ready.float()
        if bool(getattr(self.cfg, "per_digit_distal_gate", False)):
            link3_ready = self._v2_link3_support_age >= float(
                getattr(self.cfg, "per_digit_distal_link3_stable_steps", 1)
            )
            episode_ready = self.episode_length_buf >= int(
                getattr(self.cfg, "per_digit_distal_min_episode_step", 0)
            )
            link3_ready = link3_ready & episode_ready.unsqueeze(-1)
            if bool(getattr(self.cfg, "per_digit_distal_near_surface_enabled", False)):
                near_ready = (
                    near_surface
                    & (palm_age >= stable_steps).unsqueeze(-1)
                    & (self.episode_length_buf >= int(
                        getattr(self.cfg, "per_digit_distal_near_surface_min_episode_step", 20)
                    )).unsqueeze(-1)
                )
                link3_ready = link3_ready | near_ready
            filtered_distal = filtered_distal * link3_ready.float()
        if bool(getattr(self.cfg, "per_digit_distal_probe_enabled", False)):
            probe_ready = (
                (distance < float(getattr(self.cfg, "per_digit_distal_probe_distance_m", 0.064)))
                & (palm_age >= stable_steps).unsqueeze(-1)
                & (self.episode_length_buf >= int(
                    getattr(self.cfg, "per_digit_distal_probe_start_episode_step", 20)
                )).unsqueeze(-1)
                & ~measured_contact
            )
            probe_start = float(getattr(self.cfg, "per_digit_distal_probe_start_episode_step", 20))
            ramp_steps = max(float(getattr(self.cfg, "per_digit_distal_probe_ramp_steps", 80)), 1.0)
            if bool(getattr(self.cfg, "per_digit_distal_probe_staggered", False)):
                # Each finger starts its own ramp later, but keeps its
                # progress after reaching the target.  The per-digit gate and
                # measured-contact mask below still remove the bootstrap as
                # soon as that finger has real link4 load.
                interval = max(float(getattr(self.cfg, "per_digit_distal_probe_digit_interval_steps", 20)), 0.0)
                elapsed = self.episode_length_buf.float().unsqueeze(-1) - probe_start
                offsets = torch.arange(4, device=self.device, dtype=elapsed.dtype).view(1, 4) * interval
                probe_ramp = ((elapsed - offsets) / ramp_steps).clamp(0.0, 1.0)
            else:
                probe_ramp = ((self.episode_length_buf.float() - probe_start) / ramp_steps).clamp(0.0, 1.0).unsqueeze(-1)
            probe_actions = getattr(self.cfg, "per_digit_distal_probe_action_per_digit", None)
            if probe_actions is None:
                probe_base = probe_ramp.new_full((4,), float(getattr(self.cfg, "per_digit_distal_probe_action", 0.14)))
            else:
                probe_base = probe_ramp.new_tensor(tuple(float(value) for value in probe_actions))
            probe = probe_ramp * probe_base.unsqueeze(0)
            filtered_distal = (filtered_distal + probe * probe_ready.float()).clamp(-1.0, 1.0)
        coarse_scale = float(getattr(self.cfg, "palm_first_coarse_scale", 1.0))
        if coarse_scale < 1.0:
            palm_ready = (palm_age >= stable_steps).unsqueeze(-1)
            ramp_steps = max(float(getattr(self.cfg, "palm_first_ramp_steps", 1)), 1.0)
            ramp = ((palm_age - stable_steps) / ramp_steps).clamp(0.0, 1.0)
            coarse_start = float(getattr(self.cfg, "palm_first_coarse_scale", 1.0))
            coarse_max = max(coarse_start, float(getattr(self.cfg, "palm_first_coarse_max_scale", 1.0)))
            dynamic_scale = coarse_start + (coarse_max - coarse_start) * ramp
            coarse = filtered[..., :10].clone()
            # Global flexion and proximal curl are the two channels most likely
            # to drive an unsupported fruit off the palm.  Their progress is
            # slowed until support is observed; opposition/spread remain live.
            scale = torch.where(palm_ready, dynamic_scale.unsqueeze(-1), torch.full_like(palm_ready, coarse_scale))
            coarse[:, 0:2] = coarse[:, 0:2] * scale
            filtered = torch.cat((coarse, filtered_distal), dim=-1)
        if bool(getattr(self.cfg, "palm_first_motion_hold_enabled", False)):
            # Do not let a new coarse command amplify an outward impulse once
            # the palm is carrying the object.  Holding toward the previous
            # command preserves a learnable action path while avoiding a
            # one-step chase of a fruit that is already leaving the palm.
            tomato_pos = self.tomato.data.root_pos_w
            center_offset = torch.linalg.vector_norm(
                tomato_pos - self.grasp_center, dim=-1
            ) / max(float(self.cfg.max_center_distance_m), 1.0e-6)
            center_speed = self._relative_speed()
            palm_ready = self._v2_palm_support_age >= float(
                getattr(self.cfg, "palm_first_stable_steps", 1)
            )
            outward = (
                palm_ready
                & (center_speed >= float(self.cfg.palm_first_motion_hold_speed_m_s))
                & (center_offset >= float(self.cfg.palm_first_motion_hold_offset_ratio))
            )
            hold = max(0.0, min(1.0, float(self.cfg.palm_first_motion_hold_scale)))
            coarse = filtered[..., :10]
            previous = self.actions
            coarse = torch.where(
                outward.unsqueeze(-1),
                previous[..., :10] + hold * (coarse - previous[..., :10]),
                coarse,
            )
            filtered = torch.cat((coarse, filtered_distal), dim=-1)
        # Couple each distal residual to its matching straight-finger
        # near/proximal residual while approaching the fruit.  The coupling
        # is deliberately bounded and released once a distal load is present;
        # this preserves the policy's independent fingertip DOF while fixing
        # the imported hand's short link4 geometry.  Action channels 6..9 are
        # index..little residuals and 10..13 are their distal residuals.
        assist_gate = (
            (distance < float(self.cfg.distal_proximal_reach_assist_distance_m))
            & ~measured_contact
        ).float()
        if bool(getattr(self.cfg, "palm_first_distal_gate", False)):
            palm_ready = (palm_age >= stable_steps).unsqueeze(-1)
            assist_gate = assist_gate * palm_ready.float()
        if bool(getattr(self.cfg, "per_digit_distal_gate", False)):
            link3_ready = self._v2_link3_support_age >= float(
                getattr(self.cfg, "per_digit_distal_link3_stable_steps", 1)
            )
            episode_ready = self.episode_length_buf >= int(
                getattr(self.cfg, "per_digit_distal_min_episode_step", 0)
            )
            link3_ready = link3_ready & episode_ready.unsqueeze(-1)
            if bool(getattr(self.cfg, "per_digit_distal_near_surface_enabled", False)):
                near_ready = (
                    near_surface
                    & (palm_age >= stable_steps).unsqueeze(-1)
                    & (self.episode_length_buf >= int(
                        getattr(self.cfg, "per_digit_distal_near_surface_min_episode_step", 20)
                    )).unsqueeze(-1)
                )
                link3_ready = link3_ready | near_ready
            assist_gate = assist_gate * link3_ready.float()
        gains = getattr(self.cfg, "distal_proximal_reach_assist_gain_per_digit", None)
        if gains is None:
            gain_tensor = filtered_distal.new_full((4,), float(self.cfg.distal_proximal_reach_assist_gain))
        else:
            gain_tensor = filtered_distal.new_tensor(tuple(float(value) for value in gains))
        assist = gain_tensor.unsqueeze(0) * filtered_distal * assist_gate
        coarse = filtered[..., :10].clone()
        coarse[..., 6:10] = (coarse[..., 6:10] + assist).clamp(-1.0, 1.0)
        filtered = torch.cat((coarse, filtered_distal), dim=-1)
        if bool(getattr(self.cfg, "palm_first_motion_hold_enabled", False)):
            # Apply the same envelope once more after distal-to-proximal
            # assist.  Otherwise the assist term could immediately recreate
            # the coarse impulse that the first pass just damped.
            tomato_pos = self.tomato.data.root_pos_w
            center_offset = torch.linalg.vector_norm(
                tomato_pos - self.grasp_center, dim=-1
            ) / max(float(self.cfg.max_center_distance_m), 1.0e-6)
            center_speed = self._relative_speed()
            palm_ready = self._v2_palm_support_age >= float(
                getattr(self.cfg, "palm_first_stable_steps", 1)
            )
            outward = (
                palm_ready
                & (center_speed >= float(self.cfg.palm_first_motion_hold_speed_m_s))
                & (center_offset >= float(self.cfg.palm_first_motion_hold_offset_ratio))
            )
            hold = max(0.0, min(1.0, float(self.cfg.palm_first_motion_hold_scale)))
            coarse = torch.where(
                outward.unsqueeze(-1),
                self.actions[..., :10] + hold * (filtered[..., :10] - self.actions[..., :10]),
                filtered[..., :10],
            )
            filtered = torch.cat((coarse, filtered_distal), dim=-1)
        self._v2_distal_action = filtered_distal
        super()._pre_physics_step(filtered)

    @torch.inference_mode()
    def _read_contact_forces(self) -> torch.Tensor:
        forces = super()._read_contact_forces()
        # Parent reads the raw palm ContactSensor before v2's optional
        # support-surface proxy is merged.  Preserve that value for an
        # auditable measured-vs-virtual report.
        self._v2_measured_palm_force = self.palm_force.clone()
        if self.cfg.virtual_palm_support_enabled:
            # The authored palm support cube is the single physical support
            # collider, while the legacy palm ContactSensor is attached to
            # base_link (whose collision is intentionally disabled).  Expose
            # that support as a bounded simulation-only tactile channel when
            # the fruit bottom is inside the support surface footprint.  It
            # is not a claim about a calibrated hardware palm sensor.
            tomato_pos = self.tomato.data.root_pos_w
            bottom = tomato_pos[:, 2] - self.tomato_axes[:, 2]
            gap = float(self.cfg.palm_top_z_m) - bottom
            half_x, half_y = self.cfg.palm_support_surface_size_m[0] * 0.5, self.cfg.palm_support_surface_size_m[1] * 0.5
            xy = tomato_pos[:, :2] - self.palm_support_point[:, :2]
            inside = (xy[:, 0].abs() <= half_x) & (xy[:, 1].abs() <= half_y)
            penetration = (float(self.cfg.virtual_palm_support_margin_m) - gap).clamp_min(0.0)
            virtual_palm = (
                float(self.cfg.palm_support_force_n)
                + float(self.cfg.virtual_palm_support_gain_n_per_m) * penetration
            ).clamp_min(0.0)
            virtual_palm = torch.where(inside & (gap <= float(self.cfg.virtual_palm_support_margin_m)), virtual_palm, torch.zeros_like(virtual_palm))
            self.palm_force = torch.maximum(self.palm_force, virtual_palm)
            self.palm_contact_point_w = torch.where(inside.unsqueeze(-1), self.palm_support_point, self.palm_contact_point_w)
            self.palm_contact_point_valid |= inside & (virtual_palm >= self.cfg.palm_support_force_n)
        # Update the support age only after the physical and optional virtual
        # palm channels have been merged.  This buffer is a temporal safety
        # envelope, not a success latch: losing palm support immediately
        # closes the distal gate again.
        self._v2_palm_support_age = torch.where(
            self.palm_force >= self.cfg.palm_support_force_n,
            self._v2_palm_support_age + 1.0,
            torch.zeros_like(self._v2_palm_support_age),
        )
        self._v2_peak_palm_force = torch.maximum(self._v2_peak_palm_force, self.palm_force)
        virtual_force = torch.zeros((self.num_envs, 4), device=self.device)
        if self.cfg.virtual_distal_contact_enabled:
            # Sim-only tactile pad model.  The imported link4 bodies are
            # roughly 1--2 mm boxes and cannot represent a compliant fingertip
            # pad.  Estimate a bounded normal load when the link4 body centre
            # enters the tomato-radius + authored-pad-radius shell.  This is
            # exposed as an uncalibrated proxy channel and is never a hardware
            # sensor claim.
            tips = self._tip_positions()[:, 1:]
            distance = torch.linalg.vector_norm(
                tips - self.tomato.data.root_pos_w.unsqueeze(1), dim=-1
            )
            fruit_radius = self.tomato_axes.mean(-1, keepdim=True)
            shell = fruit_radius + float(self.cfg.fingertip_collision_proxy_radius_m) + float(self.cfg.virtual_distal_contact_margin_m)
            penetration = (shell - distance).clamp_min(0.0)
            virtual_force = (
                float(self.cfg.contact_threshold_n)
                + float(self.cfg.virtual_distal_contact_gain_n_per_m) * penetration
            ).clamp_min(0.0).minimum(self._ceiling[1:].unsqueeze(0))
            virtual_force = torch.where(penetration > 0.0, virtual_force, torch.zeros_like(virtual_force))
            forces = forces.clone()
            forces[:, 1:] = torch.maximum(forces[:, 1:], virtual_force)
            self.contact_forces = forces
            self.force_history[:, 0] = forces
        active = forces >= self.cfg.contact_threshold_n
        self._v2_contact_age = torch.where(active, self._v2_contact_age + 1.0, torch.zeros_like(self._v2_contact_age))
        self._v2_lost_contact_age = torch.where(~active, self._v2_lost_contact_age + 1.0, torch.zeros_like(self._v2_lost_contact_age))
        self._v2_max_lost_contact_gap = torch.maximum(self._v2_max_lost_contact_gap, self._v2_lost_contact_age)
        step = self.episode_length_buf.to(self.device).unsqueeze(-1)
        self._v2_first_contact_step = torch.where(
            (self._v2_first_contact_step < 0) & active,
            step.expand_as(self._v2_first_contact_step), self._v2_first_contact_step,
        )
        self._v2_contact_steps += active.float()
        self._v2_pre_break_contact_steps += (active & ~self.stem_broke.unsqueeze(-1)).float()
        self._v2_post_break_contact_steps += (active & self.stem_broke.unsqueeze(-1)).float()
        self._v2_force_integral += forces * float(self.cfg.policy_dt_s)

        def sensor_force(name: str) -> torch.Tensor:
            sensor = self.scene.sensors[name]
            return torch.linalg.vector_norm(sensor.data.net_forces_w[:, 0], dim=-1).nan_to_num(0.0).clamp_min(0.0)

        self._v2_link3_force = torch.stack((
            torch.zeros_like(forces[:, 0]), sensor_force("index_contact_link3"),
            sensor_force("middle_contact_link3"), sensor_force("ring_contact_link3"),
            sensor_force("little_contact_link3"),
        ), dim=-1)
        link3_active = self._v2_link3_force[:, 1:] >= float(
            getattr(self.cfg, "per_digit_distal_link3_threshold_n", self.cfg.contact_threshold_n)
        )
        self._v2_link3_support_age = torch.where(
            link3_active,
            self._v2_link3_support_age + 1.0,
            torch.zeros_like(self._v2_link3_support_age),
        )
        # The v2 imported hand exposes the thumb fingertip as ``thumb_contact``
        # (the archived v1 bookkeeping used the obsolete ``thumb_contact_link4``
        # name).  Keep this correction v2-only: the simulation proxy, contact
        # observation and closed-loop controller must all see the same thumb
        # channel, while archived v1 checkpoints remain untouched.
        measured_link4 = torch.stack((
            sensor_force("thumb_contact"), sensor_force("index_contact"),
            sensor_force("middle_contact"), sensor_force("ring_contact"),
            sensor_force("little_contact"),
        ), dim=-1)
        self._v2_measured_link4_force = measured_link4
        self._v2_link4_force = measured_link4.clone()
        if self.cfg.virtual_distal_contact_enabled:
            self._v2_link4_force[:, 1:] = torch.maximum(
                self._v2_link4_force[:, 1:], virtual_force
            )
        measured_distal_active = self._v2_measured_link4_force[:, 1:] >= self.cfg.contact_threshold_n
        self._v2_measured_distal_contact_observed |= measured_distal_active
        self._v2_measured_link4_force_integral += self._v2_measured_link4_force[:, 1:] * float(self.cfg.policy_dt_s)
        self._v2_measured_distal_contact_steps += measured_distal_active.float()
        self._v2_measured_distal_contact_streak = torch.where(
            measured_distal_active,
            self._v2_measured_distal_contact_streak + 1.0,
            torch.zeros_like(self._v2_measured_distal_contact_streak),
        )
        self._v2_measured_distal_max_streak = torch.maximum(
            self._v2_measured_distal_max_streak,
            self._v2_measured_distal_contact_streak,
        )
        self._v2_distal_contact_observed |= self._v2_link4_force[:, 1:] >= self.cfg.contact_threshold_n
        return forces

    @torch.inference_mode()
    def _get_observations(self) -> dict[str, torch.Tensor]:
        tomato_pos = self.tomato.data.root_pos_w
        tomato_quat = self.tomato.data.root_quat_w
        tomato_lin_vel = getattr(self.tomato.data, "root_lin_vel_w", torch.zeros_like(tomato_pos))
        tomato_ang_vel = getattr(self.tomato.data, "root_ang_vel_w", torch.zeros_like(tomato_pos))
        center = ((tomato_pos - self.palm_support_point) / self.cfg.grasp_reference_scale_m).clamp(-1.0, 1.0)
        axes = normalize_semi_axes(self.tomato_axes)
        from simulation.ppo_tomato_grasp.tomato_stem_physics import quat_to_rot_matrix_6d
        orientation = quat_to_rot_matrix_6d(tomato_quat)
        twist = torch.cat(((tomato_lin_vel / 0.20).clamp(-1.0, 1.0), (tomato_ang_vel / 4.0).clamp(-1.0, 1.0)), dim=-1)
        q = self.robot.data.joint_pos[:, self.active_joint_ids]
        qd = self.robot.data.joint_vel[:, self.active_joint_ids]
        q_norm = (2.0 * (q - self._lower) / (self._upper - self._lower) - 1.0).clamp(-1.0, 1.0)
        qd_norm = (qd / self._velocity.clamp_min(1.0e-3)).clamp(-1.0, 1.0)
        force = self._delayed_tactile_force()
        force_rate = ((force - self.force_history[:, 1]) / self.cfg.policy_dt_s / 20.0).clamp(-1.0, 1.0)
        active = (force >= self.cfg.contact_threshold_n).float()
        measured = self.contact_points_valid.float()
        proximal_distal_ratio = self._v2_link4_force / (
            self._v2_link3_force + self._v2_link4_force + 1.0e-4
        )
        tactile = torch.stack(
            (
                (force / self._ceiling).clamp(0.0, 1.0),
                force_rate,
                active,
                (self._v2_contact_age / 20.0).clamp(0.0, 1.0),
                (self._v2_lost_contact_age / 20.0).clamp(0.0, 1.0),
                1.0 - (self.sensor_dropout | self.sensor_stale).float(),
                proximal_distal_ratio.clamp(0.0, 1.0),
            ),
            dim=-1,
        ).reshape(self.num_envs, -1)
        points, _ = self._geometry_contact_points()
        rel_points = points - tomato_pos.unsqueeze(1)
        # The strict geometry buffers are updated during reward, but the
        # current geometric features can be computed directly for a causal
        # observation at the next policy step.
        radial = torch.linalg.vector_norm(rel_points, dim=-1).clamp(0.0, 0.10) / 0.10
        coverage = (self.strict_coverage_current / 360.0).clamp(0.0, 1.0)
        opposition = ((self.strict_opposition_current + 1.0) * 0.5).clamp(0.0, 1.0)
        hemisphere = self.strict_hemisphere_current.clamp(0.0, 1.0)
        geometry = torch.cat(
            (coverage.unsqueeze(-1), opposition.unsqueeze(-1), hemisphere.unsqueeze(-1),
             radial, measured.mean(-1, keepdim=True), active.all(-1, keepdim=True).float()), dim=-1
        )
        palm = torch.stack(
            (
                (self.palm_force / (1.25 * self.cfg.safe_force_n)).clamp(0.0, 1.0),
                (self.palm_force >= self.cfg.palm_support_force_n).float(),
                self.palm_contact_point_valid.float(),
                (torch.linalg.vector_norm(tomato_pos - self.grasp_center, dim=-1) / self.cfg.max_center_distance_m).clamp(0.0, 1.0),
                (self._relative_speed() / self.cfg.max_relative_speed_m_s).clamp(0.0, 1.0),
            ), dim=-1
        )
        phase = torch.stack(
            (
                (self.prepick_stable_steps.float() / max(float(self.cfg.prepick_hold_steps), 1.0)).clamp(0.0, 1.0),
                self.pick_started.float(), self.stem_broke.float(),
                (self.pick_progress_m / max(float(self.cfg.pick_displacement_m), 1.0e-6)).clamp(0.0, 1.0),
                (self.strict_hold_streak.float() / max(float(self.cfg.strict_hold_steps), 1.0)).clamp(0.0, 1.0),
                (self.strict_first_palm >= 0).float(), (self.strict_first_three >= 0).float(),
                # During stage 1 this slot exposes elapsed episode phase so a
                # policy can represent a staged approach teacher without a
                # hidden wall-clock state.  After stem break it becomes the
                # post-break hold clock required by stage 2.
                torch.where(
                    self.stem_broke,
                    (self.post_break_steps.float() / 40.0).clamp(0.0, 1.0),
                    (self.episode_length_buf.float() / max(float(self.max_episode_length), 1.0)).clamp(0.0, 1.0),
                ),
            ), dim=-1
        )
        safety = torch.stack(
            (
                (1.0 - force / self._ceiling).amin(-1).clamp(-1.0, 1.0),
                (1.0 - self.palm_force / (1.25 * self.cfg.safe_force_n)).clamp(-1.0, 1.0),
                1.0 - (torch.linalg.vector_norm(tomato_pos - self.grasp_center, dim=-1) / self.cfg.max_center_distance_m).clamp(0.0, 1.0),
                1.0 - (self._relative_speed() / self.cfg.max_relative_speed_m_s).clamp(0.0, 1.0),
                1.0 - self.damage_score.clamp(0.0, 1.0),
                1.0 - (self.actions.abs() > 0.99).float().mean(-1),
            ), dim=-1
        )
        actor = torch.cat((center, axes, orientation, twist, q_norm, qd_norm, tactile, palm, geometry, phase, self.actions, safety), dim=-1)
        if actor.shape[-1] != V2_ACTOR_OBSERVATION_DIM:
            raise RuntimeError(f"v2 actor observation width drifted: {actor.shape[-1]}")
        actor = torch.nan_to_num(actor, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        self._cached_actor = actor
        return {"policy": actor}

    @torch.inference_mode()
    def _get_states(self) -> torch.Tensor:
        if self._cached_actor is None:
            self._get_observations()
        tomato_pos = self.tomato.data.root_pos_w
        points, _ = self._geometry_contact_points()
        contact_points = (points - tomato_pos.unsqueeze(1)) / 0.08
        palm_point = torch.where(
            self.palm_contact_point_valid.unsqueeze(-1),
            (self.palm_contact_point_w - tomato_pos) / 0.08,
            torch.zeros((self.num_envs, 3), device=self.device),
        )
        privileged = torch.cat(
            (
                (self.contact_forces / 6.0).clamp(0.0, 2.0),
                (self.palm_force / 8.0).clamp(0.0, 2.0).unsqueeze(-1),
                contact_points.clamp(-2.0, 2.0).flatten(-2), palm_point.clamp(-2.0, 2.0),
                (self.contact_counts / 2.0).clamp(0.0, 2.0),
                (self.object_mass / 0.30).clamp(0.0, 2.0).unsqueeze(-1),
                self.object_friction.clamp(0.0, 2.0).unsqueeze(-1),
                (self.object_stiffness / 9000.0).clamp(0.0, 2.0).unsqueeze(-1),
                self.damage_score.clamp(0.0, 1.0).unsqueeze(-1),
                torch.stack(tuple(self.stem_load_ratios[name] for name in ("tension", "shear", "bending", "torsion")), dim=-1).clamp(0.0, 2.0),
                (self._relative_speed() / 0.50).clamp(0.0, 2.0).unsqueeze(-1),
                ((self.tactile_gain - 1.0) / 0.18).clamp(-1.0, 1.0),
                (self.tactile_bias / 0.03).clamp(-1.0, 1.0),
            ), dim=-1
        )
        state = torch.cat((self._cached_actor, privileged), dim=-1)
        if state.shape[-1] != V2_CRITIC_STATE_DIM:
            raise RuntimeError(f"v2 critic state width drifted: {state.shape[-1]}")
        return torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

    @torch.inference_mode()
    def _get_rewards(self) -> torch.Tensor:
        # Bypass EnvelopingPhysxTomatoGraspEnv's extra three-finger strict
        # reward; call the validated physical reward chain directly.
        reward = PhysxTomatoGraspEnv._get_rewards(self)
        strict_now, newly = self._update_strict_geometry()
        self.success = self.strict_success_latched.clone()
        contact = self.contact_forces >= self.cfg.contact_threshold_n
        palm = self.palm_force >= self.cfg.palm_support_force_n
        tomato_pos = self.tomato.data.root_pos_w
        in_hand = torch.linalg.vector_norm(tomato_pos - self.grasp_center, dim=-1) <= self.cfg.max_center_distance_m
        all_five_palm = contact.all(-1) & palm
        distal_contact = self._v2_link4_force[:, 1:] >= self.cfg.contact_threshold_n
        all_four_distal = distal_contact.all(-1)
        thumb_with_distal = contact[:, 0] & (distal_contact.sum(-1) >= 3)
        contact_quality = contact.float().mean(-1)
        coverage_progress = (self.strict_coverage_current / max(float(self.cfg.strict_angular_coverage_min_deg), 1.0)).clamp(0.0, 1.0)
        distal_delta = (self.robot.data.joint_pos[:, self.active_joint_ids] - self.strict_initial_joint_pos).abs()[:, [12, 13, 14, 15]]
        distal_progress = (distal_delta / max(float(self.cfg.strict_distal_participation_delta_rad), 1.0e-3)).clamp(0.0, 1.0)
        # A distal residual is useful only when its corresponding finger is
        # actually carrying contact; free-space curling earns no bonus.
        distal_contact = self._v2_link4_force[:, 1:] >= self.cfg.contact_threshold_n
        distal_contact_progress = (distal_progress * distal_contact.float()).mean(-1)
        measured_distal_contact = self._v2_measured_link4_force[:, 1:] >= self.cfg.contact_threshold_n
        palm_ready = palm & in_hand
        # Contact before palm support is a destabilizing pinch shortcut.  Keep
        # the inherited proximity signal, but subtract it until the fruit is
        # demonstrably carried by the palm; distal load quality is likewise
        # only useful after support.  This is a reward-shaping change, never a
        # relaxation of the strict terminal gate.
        reward += 1.4 * contact_quality + 1.6 * all_five_palm.float()
        reward -= 2.0 * contact_quality * (~palm_ready).float()
        reward += float(self.cfg.v2_all_five_palm_reward) * (all_five_palm & in_hand).float()
        reward += float(self.cfg.v2_all_four_distal_reward) * (all_four_distal & palm_ready).float()
        reward += float(self.cfg.v2_thumb_with_distal_reward) * (thumb_with_distal & palm_ready).float()
        reward += float(self.cfg.v2_in_hand_support_reward) * palm_ready.float()
        # Give each distal channel a bounded, per-digit signal before the
        # all-four conjunction is reached.  This is still gated by valid
        # in-hand support and measured/virtual link4 contact; free-space
        # distal excursion receives no reward.
        reward += 2.0 * (distal_contact.float().sum(-1) / 4.0) * palm_ready.float()
        measured_weight = float(getattr(self.cfg, "v2_measured_distal_contact_reward", 0.0))
        if measured_weight > 0.0:
            # Give each real link4 contact its own bounded signal.  This is
            # deliberately separate from the virtual contact reward above;
            # a virtual pad cannot satisfy this term.
            reward += measured_weight * (measured_distal_contact.float().sum(-1) / 4.0) * palm_ready.float()
        measured_force_weight = float(getattr(self.cfg, "v2_measured_distal_force_reward", 0.0))
        if measured_force_weight > 0.0:
            measured_load_quality = (
                self._v2_measured_link4_force[:, 1:] / max(float(self.cfg.contact_threshold_n), 1.0e-6)
            ).clamp(0.0, 1.0).mean(-1)
            reward += measured_force_weight * measured_load_quality * palm_ready.float()
        reward += 1.2 * coverage_progress * (contact.sum(-1) >= 4).float()
        reward += 1.0 * distal_contact_progress * (contact.sum(-1) >= 4).float()
        # Continuous measured distal-load shaping makes the new link4 proxy
        # observable before all-four strict readiness.  It is normalized by
        # each digit's safety ceiling and never uses joint travel alone.
        distal_load_quality = (
            self._v2_link4_force[:, 1:] / self._ceiling[1:].unsqueeze(0)
        ).clamp(0.0, 1.0).mean(-1)
        reward += 1.8 * distal_load_quality * palm_ready.float()
        reward -= 1.0 * distal_load_quality * (~palm_ready).float()

        # P1 trust-region shaping: before the stem is broken, avoid the
        # failure mode observed in the checkpoint trace where root/coarse
        # channels remain at +/-1 and push the fruit out of the palm.  These
        # penalties are intentionally continuous and do not act after the
        # harvest phase, where tactile correction is expected.
        pre_pick = (~self.stem_broke) & (~self.pick_started)
        saturation = (self.actions.abs() > 0.95).float().mean(-1)
        action_step = self.action_delta.abs().mean(-1)
        center_ratio = (
            torch.linalg.vector_norm(tomato_pos - self.grasp_center, dim=-1)
            / max(float(self.cfg.max_center_distance_m), 1.0e-6)
        ).clamp(0.0, 1.5)
        center_speed_ratio = (
            self._relative_speed() / max(float(self.cfg.max_relative_speed_m_s), 1.0e-6)
        ).clamp(0.0, 2.0)
        force_rate = (
            (self.contact_forces - self.previous_contact_forces).abs() / max(float(self.step_dt), 1.0e-6)
        ).amax(-1)
        reward -= pre_pick.float() * (
            float(self.cfg.v2_action_saturation_penalty) * saturation
            + float(self.cfg.v2_action_delta_penalty) * action_step
            + float(self.cfg.v2_center_offset_penalty) * center_ratio
            + float(self.cfg.v2_center_velocity_penalty) * center_speed_ratio
            + float(self.cfg.v2_force_rate_penalty) * (force_rate / 30.0).clamp(0.0, 2.0)
        )
        reward += 3.0 * strict_now.float() + 12.0 * newly.float()
        if self.cfg.training_stage >= 2:
            post = self.stem_broke
            reward += post.float() * (2.0 * contact_quality + 1.5 * all_five_palm.float())
            reward -= post.float() * (2.0 * (1.0 - contact_quality) + 0.8 * (~palm).float())
        log = self.extras.setdefault("log", {})
        log["v2_all_five_palm_rate"] = all_five_palm.float().mean()
        log["v2_strict_new_success_rate"] = newly.float().mean()
        log["v2_distal_contact_progress"] = distal_contact_progress.mean()
        log["v2_distal_load_quality"] = distal_load_quality.mean()
        log["v2_geometry_measured_rate"] = self.contact_points_valid.float().mean()
        log["v2_action_saturation_rate"] = saturation.mean()
        log["v2_action_delta_mean"] = action_step.mean()
        log["v2_center_offset_ratio"] = center_ratio.mean()
        log["v2_center_speed_ratio"] = center_speed_ratio.mean()
        log["v2_force_rate_max_n_s"] = force_rate.mean()
        return reward


__all__ = ["V2EnvelopingPhysxTomatoGraspEnv", "V2EnvelopingPhysxTomatoGraspEnvCfg"]
