"""
analytics.py - Calculation module for the Smart Electricity Bill Analyzer.

Scope: Maharashtra MSEDCL LT-I Domestic tariff (project reference rates).

IMPORTANT:
- The tariff rates below are the PROJECT'S REFERENCE RATES. Verify them
  against the current official MSEDCL tariff order before final submission.
- energy_charge() returns the ESTIMATED ENERGY CHARGE ONLY. It is NOT the
  complete electricity bill (fixed charges, wheeling charges, fuel
  adjustment, electricity duty, taxes, etc. are not included).

This module is independent of Streamlit, SQLite, OCR, files and web APIs.
"""

import math
import numbers

# ---------------------------------------------------------------------------
# Reference tariff data
# Each tuple is (width of the slab in kWh, rate in rupees per kWh).
# A width of None means "no upper limit" (the last slab).
# ---------------------------------------------------------------------------
SLABS = [
    (100, 3.96),
    (200, 10.80),
    (200, 15.03),
    (None, 17.60),
]

# Cumulative upper limits of slabs 1, 2 and 3 (100, 100+200, 100+200+200).
SLAB_BOUNDARIES = [100, 300, 500]


def _validate_units(value, name: str = "units") -> float:
    """Check that `value` is a finite, non-negative number; return it as float.

    Raises:
        TypeError: if value is not a real number (bool and text are rejected).
        ValueError: if value is NaN, infinite or negative.
    """
    # bool is technically a number in Python (True == 1), but it is almost
    # certainly a bug here, so we reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")

    value = float(value)

    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be a finite number, got {value}.")
    if value < 0:
        raise ValueError(f"{name} cannot be negative, got {value}.")

    return value


def energy_charge(units: float) -> dict:
    """Calculate the estimated ENERGY CHARGE using telescopic (progressive) billing.

    Each slab is charged only for the units that fall inside that slab.
    This is the energy charge only, not the full electricity bill.

    Args:
        units: Consumption in kWh (non-negative, finite).

    Returns:
        {
            "total": float,   # total energy charge in rupees (2 decimals)
            "breakdown": [    # one entry per slab, usable in pd.DataFrame()
                {"slab": 1, "units": ..., "rate": ..., "cost": ...},
                ...
            ],
        }

    Raises:
        TypeError / ValueError: for invalid input.
    """
    units = _validate_units(units, "units")

    remaining = units          # units not yet assigned to a slab
    total = 0.0                # running total, NOT rounded until the end
    breakdown = []

    for slab_number, (width, rate) in enumerate(SLABS, start=1):
        if width is None:
            units_in_slab = remaining              # last slab takes everything left
        else:
            units_in_slab = min(remaining, width)  # at most the slab's width

        cost = units_in_slab * rate
        total += cost
        remaining -= units_in_slab

        breakdown.append(
            {
                "slab": slab_number,
                "units": units_in_slab,
                "rate": rate,
                "cost": round(cost, 2),   # rounded for display only
            }
        )

    return {"total": round(total, 2), "breakdown": breakdown}


def current_slab(units: float) -> int:
    """Return the slab number (1 to 4) that `units` falls in.

    Boundaries are inclusive on the upper side:
        0   - 100 units : slab 1
        >100 - 300 units: slab 2
        >300 - 500 units: slab 3
        >500 units      : slab 4

    Raises:
        TypeError / ValueError: for invalid input.
    """
    units = _validate_units(units, "units")

    for index, boundary in enumerate(SLAB_BOUNDARIES, start=1):
        if units <= boundary:
            return index
    return len(SLAB_BOUNDARIES) + 1   # above 500 -> slab 4
from datetime import datetime, timedelta
import pandas as pd


def estimate_daily_load(units, start_date, days=30):
    """Estimate daily consumption from total billed units."""
    if days <= 0:
        raise ValueError("Billing days must be positive.")

    units = _validate_units(units)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    daily_units = units / days

    records = []

    for i in range(days):
        current_date = start + timedelta(days=i)

        records.append({
    "date": current_date.date(),
    "kwh": round(daily_units, 4),
    "is_weekend": current_date.weekday() >= 5,
    "day_type": "Weekend" if current_date.weekday() >= 5 else "Weekday"
})

    return pd.DataFrame(records)


def daily_reference_levels(units, days):
    """Calculate daily equivalents of monthly slab boundaries."""
    _validate_units(units)

    if days <= 0:
        raise ValueError("Billing days must be positive.")

    return {
        "slab_daily_kwh": {
            100: 100 / days,
            300: 300 / days,
            500: 500 / days
        }
    }
def project_units(units, elapsed_days, cycle_days=30):
    """Project electricity usage for a full billing cycle."""
    units = _validate_units(units)

    if elapsed_days <= 0 or cycle_days <= 0:
        raise ValueError("Billing days must be positive.")

    return round((units / elapsed_days) * cycle_days, 2)


def slab_alert(units):
    """Return slab status based on projected consumption."""
    units = _validate_units(units)

    if units >= 500:
        return {
            "level": "CRITICAL",
            "boundary": 500,
            "headroom": 0,
            "message": "High electricity consumption! You are in the highest usage range."
        }

    if units >= 450:
        return {
            "level": "WARNING",
            "boundary": 500,
            "headroom": 500 - units,
            "message": "Warning! You are approaching the 500-unit slab boundary."
        }

    if units >= 300:
        boundary = 500
    elif units >= 100:
        boundary = 300
    else:
        boundary = 100

    return {
        "level": "SAFE",
        "boundary": boundary,
        "headroom": boundary - units,
        "message": f"Your projected consumption is {units:.1f} units. Next slab boundary: {boundary} units."
    }

# ---------------------------------------------------------
# APPLIANCE USAGE OPTIMIZER
# ---------------------------------------------------------

def get_default_appliances():
    """Return illustrative appliance power and daily usage values."""
    return [
        {"name": "Fan", "watts": 75, "hours_day": 8.0},
        {"name": "LED Bulb", "watts": 10, "hours_day": 6.0},
        {"name": "Television", "watts": 100, "hours_day": 4.0},
        {"name": "Refrigerator", "watts": 150, "hours_day": 8.0},
        {"name": "Air Conditioner", "watts": 1500, "hours_day": 6.0},
        {"name": "Washing Machine", "watts": 500, "hours_day": 1.0},
        {"name": "Laptop", "watts": 65, "hours_day": 5.0},
        {"name": "Water Heater", "watts": 2000, "hours_day": 1.0},
    ]


def appliance_monthly_kwh(appliance):
    """Calculate estimated monthly energy use for an appliance."""
    watts = _validate_units(appliance["watts"], "watts")
    hours = _validate_units(appliance["hours_day"], "hours_day")

    if hours > 24:
        raise ValueError("Daily usage cannot exceed 24 hours.")

    return round(watts * hours * 30 / 1000, 2)


def optimize_appliances(current_units, target_units, appliances):
    """Suggest daily usage reductions to approach a target."""
    current_units = _validate_units(current_units, "current_units")
    target_units = _validate_units(target_units, "target_units")

    if not isinstance(appliances, list):
        raise TypeError("Appliances must be a list.")

    needed_saving = max(0.0, current_units - target_units)
    recommendations = []

    # Prioritize appliances with greater estimated monthly usage.
    ordered = sorted(
        appliances,
        key=appliance_monthly_kwh,
        reverse=True
    )

    for appliance in ordered:
        if needed_saving <= 0:
            break

        watts = _validate_units(appliance["watts"], "watts")
        hours = _validate_units(appliance["hours_day"], "hours_day")

        if hours > 24:
            raise ValueError("Daily usage cannot exceed 24 hours.")

        if watts == 0 or hours == 0:
            continue

        available_kwh = watts * hours * 30 / 1000
        saving = min(needed_saving, available_kwh)
        reduce_hours = saving * 1000 / (watts * 30)

        recommendations.append({
            "name": appliance["name"],
            "reduce_hours_per_day": round(reduce_hours, 2),
            "estimated_kwh_saving": round(saving, 2),
        })

        needed_saving -= saving

    return recommendations


def summarize_optimization(current_units, target_units, recommendations):
    """Summarize estimated energy and rupee savings."""
    current_units = _validate_units(current_units, "current_units")
    target_units = _validate_units(target_units, "target_units")

    saved_kwh = sum(
        item["estimated_kwh_saving"]
        for item in recommendations
    )

    remaining_units = max(0.0, current_units - saved_kwh)

    original_cost = energy_charge(current_units)["total"]
    new_cost = energy_charge(remaining_units)["total"]

    rupee_saving = round(original_cost - new_cost, 2)
    target_met = remaining_units <= target_units + 0.01

    if target_met:
        message = (
            f"Your target is achievable with the selected appliance "
            f"reductions. Estimated savings: {saved_kwh:.2f} kWh/month."
        )
    else:
        message = (
            f"Selected appliances could save approximately "
            f"{saved_kwh:.2f} kWh/month, but the target may require "
            f"additional changes."
        )

    return {
        "message": message,
        "total_kwh_saving": round(saved_kwh, 2),
        "total_rupee_saving": rupee_saving,
        "remaining_units": round(remaining_units, 2),
        "target_met": target_met,
    }

