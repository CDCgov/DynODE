from __future__ import annotations

from typing import Any

from dynode.structure.bins import AgeBin


def introduction_age_mask(
    *,
    strain: Any,
    age_bins: list[AgeBin] | tuple[AgeBin, ...],
) -> list[int]:
    """Convert introduction_ages into a mask over model age bins."""
    if not strain.is_introduced or strain.introduction_ages is None:
        return [0 for _ in age_bins]

    missing = [age for age in strain.introduction_ages if age not in age_bins]

    if missing:
        raise ValueError(
            f"Strain {strain.name!r} has introduction_ages not present "
            f"in model age bins: {missing}."
        )

    return [
        1 if age_bin in strain.introduction_ages else 0 for age_bin in age_bins
    ]
