"""Exclusive, expiring same-side arm/hand control leases."""

from __future__ import annotations

from dataclasses import dataclass

from proto import ArmSide

from .clock import Clock


class ControlBusyError(RuntimeError):
    """The requested arm/hand pair is already owned."""


class InvalidControlLeaseError(RuntimeError):
    """A command used a stale, expired, released, or mismatched lease."""


@dataclass(frozen=True, slots=True)
class ControlLease:
    lease_id: str
    owner_id: str
    side: ArmSide
    controller_epoch: int
    expires_monotonic_ns: int

    def __post_init__(self) -> None:
        for field_name in ("lease_id", "owner_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())
        if not isinstance(self.side, ArmSide):
            try:
                object.__setattr__(self, "side", ArmSide(self.side))
            except (TypeError, ValueError) as exc:
                raise ValueError("side must be left or right") from exc
        if (
            isinstance(self.controller_epoch, bool)
            or not isinstance(self.controller_epoch, int)
            or self.controller_epoch <= 0
        ):
            raise ValueError("controller_epoch must be an integer > 0")
        if (
            isinstance(self.expires_monotonic_ns, bool)
            or not isinstance(self.expires_monotonic_ns, int)
            or self.expires_monotonic_ns <= 0
        ):
            raise ValueError("expires_monotonic_ns must be an integer > 0")


class ControlLeaseManager:
    """Own at most one active lease per side and reject stale epochs."""

    def __init__(self, clock: Clock) -> None:
        if not isinstance(clock, Clock):
            raise TypeError("clock must implement Clock")
        self._clock = clock
        self._leases: dict[ArmSide, ControlLease] = {}
        self._epoch = 0

    def acquire(
        self,
        *,
        owner_id: str,
        side: ArmSide,
        ttl_s: float,
    ) -> ControlLease:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must be a non-empty string")
        if not isinstance(side, ArmSide):
            side = ArmSide(side)
        if isinstance(ttl_s, bool) or not isinstance(ttl_s, (int, float)) or ttl_s <= 0:
            raise ValueError("ttl_s must be a real number > 0")
        self._discard_expired(side)
        current = self._leases.get(side)
        if current is not None:
            raise ControlBusyError(
                f"{side.value} arm/hand is owned by {current.owner_id!r}"
            )
        self._epoch += 1
        lease = ControlLease(
            lease_id=f"lease-{side.value}-{self._epoch}",
            owner_id=owner_id.strip(),
            side=side,
            controller_epoch=self._epoch,
            expires_monotonic_ns=(
                self._clock.monotonic_ns() + round(float(ttl_s) * 1_000_000_000)
            ),
        )
        self._leases[side] = lease
        return lease

    def validate(self, lease: ControlLease, *, side: ArmSide) -> None:
        if not isinstance(lease, ControlLease):
            raise TypeError("lease must be a ControlLease")
        if not isinstance(side, ArmSide):
            side = ArmSide(side)
        self._discard_expired(side)
        current = self._leases.get(side)
        if current is None or current != lease or lease.side is not side:
            raise InvalidControlLeaseError("control lease is not active for this side")

    def release(self, lease: ControlLease) -> None:
        if not isinstance(lease, ControlLease):
            raise TypeError("lease must be a ControlLease")
        current = self._leases.get(lease.side)
        if current == lease:
            del self._leases[lease.side]

    def active(self, side: ArmSide) -> ControlLease | None:
        if not isinstance(side, ArmSide):
            side = ArmSide(side)
        self._discard_expired(side)
        return self._leases.get(side)

    def revoke(self, side: ArmSide) -> None:
        if not isinstance(side, ArmSide):
            side = ArmSide(side)
        self._leases.pop(side, None)

    def _discard_expired(self, side: ArmSide) -> None:
        current = self._leases.get(side)
        if (
            current is not None
            and self._clock.monotonic_ns() >= current.expires_monotonic_ns
        ):
            del self._leases[side]


__all__ = [
    "ControlBusyError",
    "ControlLease",
    "ControlLeaseManager",
    "InvalidControlLeaseError",
]
