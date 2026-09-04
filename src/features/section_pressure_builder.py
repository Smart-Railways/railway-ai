"""
RailKit Section Pressure Builder — Phase 5.8

Computes per-section operational pressure from real RailKit captures
using validated section mapping evidence.

Architecture:
    RailKit capture JSONs
        → SectionEvidenceBuilder (Phase 5.7)
        → section-level delay extraction from timelines
        → per-section operational pressure [0, 1]
        → MaintenanceDecisionEngine (existing)
        → BlockOptimizer (existing)

Critical design rules:
    - NEVER silently assign train-level pressure to every section
    - Unknown/unvalidated sections remain distinguishable from zero pressure
    - Only sections with OBSERVED_ORDERED evidence get pressure computed
    - Sections without evidence get pressure = None (not 0.0)
"""

import json
import glob
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.data.section_evidence_builder import (
    SectionEvidenceBuilder,
    OBSERVED_ORDERED,
)


@dataclass
class SectionPressure:
    """Operational pressure for a single section derived from real data."""
    section_id: str
    delay_risk: float              # max delay / 60, clipped [0,1]
    delay_intensity: float         # avg delay / 60, clipped [0,1]
    route_pressure: float          # station count fraction [0,1]
    operational_pressure: float    # weighted composite [0,1]
    evidence_strength: str         # STRONG / MODERATE / WEAK
    contributing_trains: list      # train numbers contributing
    observation_count: int         # number of snapshots used
    data_source: str = "railkit"


class RailKitSectionPressureBuilder:
    """
    Builds section-level operational pressure from real RailKit data.

    Flow:
        1. Load section evidence (from Phase 5.7 CSV or SectionEvidenceBuilder)
        2. Filter to OBSERVED_ORDERED sections only
        3. For each validated section, extract delay data from timeline
        4. Compute canonical pressure formula per section
        5. Return dict suitable for PlanningService.plan_with_validated_mapping()

    Sections without OBSERVED_ORDERED evidence receive NO pressure
    (None, not 0.0) — they are genuinely unknown, not zero-pressure.
    """

    # Canonical weights — must match RealOperationalFeatureBuilder
    DELAY_RISK_WEIGHT = 0.60
    DELAY_INTENSITY_WEIGHT = 0.25
    ROUTE_PRESSURE_WEIGHT = 0.15

    def __init__(
        self,
        captures_dir: str = "data/raw_real/railkit",
        evidence_csv: str = "data/processed_real/railkit_section_mapping_evidence.csv",
    ):
        self.captures_dir = captures_dir
        self.evidence_csv = evidence_csv

    def _load_evidence(self) -> pd.DataFrame:
        """Load section evidence CSV from Phase 5.7."""
        if not os.path.exists(self.evidence_csv):
            raise FileNotFoundError(
                f"Section evidence CSV not found: {self.evidence_csv}. "
                f"Run Phase 5.7 (SectionEvidenceBuilder) first."
            )
        return pd.read_csv(self.evidence_csv)

    def _get_validated_sections(self, evidence_df: pd.DataFrame) -> dict:
        """
        Identify sections with OBSERVED_ORDERED evidence.

        Returns: {section_id: {
            "origin_codes": set of observed origin codes,
            "dest_codes": set of observed destination codes,
            "trains": set of contributing train numbers,
            "count": observation count,
        }}
        """
        ordered = evidence_df[evidence_df["status"] == OBSERVED_ORDERED].copy()

        if ordered.empty:
            return {}

        validated = {}
        for section_id, group in ordered.groupby("section_id"):
            validated[section_id] = {
                "origin_codes": set(group["origin_station"].dropna().unique()),
                "dest_codes": set(group["destination_station"].dropna().unique()),
                "trains": set(str(t) for t in group["train_number"].unique()),
                "count": len(group),
            }

        return validated

    def _load_capture_timeline(self, filepath: str) -> tuple:
        """Load a capture and extract timeline + metadata."""
        with open(filepath, "r") as f:
            data = json.load(f)

        if "response" in data:
            train_data = data["response"].get("data", {})
        elif "data" in data:
            train_data = data["data"]
        else:
            train_data = data

        timeline = train_data.get("timeline", [])
        train_no = str(data.get("train_number") or train_data.get("trainNo", "UNKNOWN"))

        return timeline, train_no

    def _extract_section_delays(
        self,
        timeline: list,
        origin_codes: set,
        dest_codes: set,
    ) -> list:
        """
        Extract delay values for stations between origin and destination.

        Only extracts delays from stations within the section boundary.
        """
        # Find origin and destination indices
        origin_idx = None
        dest_idx = None

        for i, entry in enumerate(timeline):
            code = entry.get("stationCode", "")
            if code in origin_codes and origin_idx is None:
                origin_idx = i
            if code in dest_codes and origin_idx is not None:
                dest_idx = i
                break

        if origin_idx is None or dest_idx is None:
            return []

        # Extract delays from stations in the section
        delays = []
        for i in range(origin_idx, dest_idx + 1):
            entry = timeline[i]
            for movement_key in ("arrival", "departure"):
                movement = entry.get(movement_key, {}) or {}
                delay = movement.get("delay")

                if delay is None or delay == "":
                    continue

                text = str(delay).lower().strip()
                if text == "on time":
                    delays.append(0.0)
                    continue

                digits = "".join(c for c in text if c.isdigit() or c == ".")
                if digits:
                    try:
                        delays.append(float(digits))
                    except ValueError:
                        continue

        return delays

    def build_section_pressure(self) -> dict:
        """
        Build per-section operational pressure from real RailKit data.

        Returns: {section_id: SectionPressure} for validated sections only.
        Sections without OBSERVED_ORDERED evidence are NOT in the result.
        """
        evidence_df = self._load_evidence()
        validated = self._get_validated_sections(evidence_df)

        if not validated:
            return {}

        # Load all capture files
        pattern = os.path.join(self.captures_dir, "live_train_*.json")
        files = sorted(glob.glob(pattern))

        # Collect delays per section across all captures
        section_delays = {sid: [] for sid in validated}
        section_station_counts = {sid: [] for sid in validated}

        for filepath in files:
            timeline, train_no = self._load_capture_timeline(filepath)
            if not timeline:
                continue

            for section_id, info in validated.items():
                if train_no not in info["trains"]:
                    # This train doesn't contribute evidence for this section
                    continue

                delays = self._extract_section_delays(
                    timeline, info["origin_codes"], info["dest_codes"]
                )

                if delays:
                    section_delays[section_id].extend(delays)

                # Count stations in section for route pressure
                origin_idx = None
                dest_idx = None
                for i, entry in enumerate(timeline):
                    code = entry.get("stationCode", "")
                    if code in info["origin_codes"] and origin_idx is None:
                        origin_idx = i
                    if code in info["dest_codes"] and origin_idx is not None:
                        dest_idx = i
                        break
                if origin_idx is not None and dest_idx is not None:
                    section_station_counts[section_id].append(
                        dest_idx - origin_idx + 1
                    )

        # Compute pressure per section
        results = {}
        for section_id, info in validated.items():
            delays = section_delays[section_id]
            station_counts = section_station_counts[section_id]

            if not delays and not station_counts:
                # Evidence exists but no delay data extractable
                # Still mark as having evidence, with zero delays
                max_delay = 0.0
                avg_delay = 0.0
            else:
                max_delay = max(delays) if delays else 0.0
                avg_delay = (sum(delays) / len(delays)) if delays else 0.0

            avg_stations = (
                sum(station_counts) / len(station_counts)
                if station_counts else 0.0
            )

            # Canonical pressure formula (matching RealOperationalFeatureBuilder)
            delay_risk = min(max_delay / 60.0, 1.0)
            delay_intensity = min(avg_delay / 60.0, 1.0)
            route_pressure = min(avg_stations / 100.0, 1.0)

            operational_pressure = (
                self.DELAY_RISK_WEIGHT * delay_risk
                + self.DELAY_INTENSITY_WEIGHT * delay_intensity
                + self.ROUTE_PRESSURE_WEIGHT * route_pressure
            )
            operational_pressure = max(0.0, min(1.0, operational_pressure))

            # Determine evidence strength
            count = info["count"]
            train_count = len(info["trains"])
            if count >= 2 and train_count >= 2:
                strength = "STRONG"
            elif count >= 2:
                strength = "STRONG"
            elif count == 1:
                strength = "MODERATE"
            else:
                strength = "WEAK"

            results[section_id] = SectionPressure(
                section_id=section_id,
                delay_risk=round(delay_risk, 6),
                delay_intensity=round(delay_intensity, 6),
                route_pressure=round(route_pressure, 6),
                operational_pressure=round(operational_pressure, 6),
                evidence_strength=strength,
                contributing_trains=sorted(info["trains"]),
                observation_count=count,
            )

        return results

    def get_pressure_dict(self) -> dict:
        """
        Get section pressure as a flat dict suitable for
        PlanningService.plan_with_validated_mapping().

        Returns: {section_id: pressure_float}
        Only includes sections with OBSERVED_ORDERED evidence.
        """
        pressures = self.build_section_pressure()
        return {
            sid: sp.operational_pressure
            for sid, sp in pressures.items()
        }

    def get_pressure_report(self) -> pd.DataFrame:
        """
        Get a DataFrame summarizing section pressure for reporting.
        """
        pressures = self.build_section_pressure()
        if not pressures:
            return pd.DataFrame()

        records = []
        for sid, sp in sorted(pressures.items()):
            records.append({
                "section_id": sp.section_id,
                "delay_risk": sp.delay_risk,
                "delay_intensity": sp.delay_intensity,
                "route_pressure": sp.route_pressure,
                "operational_pressure": sp.operational_pressure,
                "evidence_strength": sp.evidence_strength,
                "contributing_trains": ", ".join(sp.contributing_trains),
                "observation_count": sp.observation_count,
                "data_source": sp.data_source,
            })

        return pd.DataFrame(records)
