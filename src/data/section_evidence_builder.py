"""
Section Evidence Builder — Phase 5.7

Extracts section mapping evidence from real RailKit capture files.
Reads raw JSON captures, identifies corridor station appearances in train
timelines, and produces structured evidence records for section validation.

This module is READ-ONLY with respect to existing data:
- Reads from: data/raw_real/railkit/live_train_*.json
- Writes to: data/processed_real/railkit_section_mapping_evidence.csv (new file)
- Does NOT modify: config/railkit_section_mapping.json
"""

import csv
import json
import glob
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


# Verified RailKit station codes for corridor stations.
# These were resolved by inspecting actual train timelines:
#   JHS  -> VGLJ  (Virangana Lakshmibai Jhansi Jn, discovered in 12002 timeline index 55)
#   VAD  -> BRC   (Vadodara/Baroda, discovered in 12904/20164/22946 timelines)
#   SRT  -> ST    (Surat, discovered in 12904/20164/22946 timelines)
#   MUM  -> MMCT  (Mumbai Central, discovered in 22946 timeline)
CORRIDOR_SECTIONS = [
    ("NDL-MTJ-01", ["NDLS"], ["MTJ"]),
    ("NDL-MTJ-02", ["NDLS"], ["MTJ"]),
    ("MTJ-AGC-01", ["MTJ"], ["AGC"]),
    ("AGC-GWL-01", ["AGC"], ["GWL"]),
    ("GWL-JHS-01", ["GWL"], ["VGLJ"]),
    ("JHS-BINA-01", ["VGLJ"], ["BINA"]),
    ("BINA-BPL-01", ["BINA"], ["BPL"]),
    ("BPL-RTM-01", ["BPL"], ["RTM"]),
    ("RTM-VAD-01", ["RTM"], ["BRC"]),
    ("VAD-SRT-01", ["BRC"], ["ST"]),
    ("SRT-MUM-01", ["ST"], ["MMCT"]),
]

# Evidence statuses
OBSERVED_ORDERED = "OBSERVED_ORDERED"
ORIGIN_ONLY = "ORIGIN_ONLY"
DESTINATION_ONLY = "DESTINATION_ONLY"
NOT_OBSERVED = "NOT_OBSERVED"
REVERSE_ORDER = "REVERSE_ORDER"
INVALID_SAME_POSITION = "INVALID_SAME_POSITION"

EVIDENCE_FIELDS = [
    "train_number",
    "capture_file",
    "section_id",
    "origin_station",
    "destination_station",
    "origin_index",
    "destination_index",
    "status",
    "timeline_station_count",
]


@dataclass
class SectionEvidence:
    """A single piece of evidence for a corridor section from a RailKit capture."""
    train_number: str
    capture_file: str
    section_id: str
    origin_station: Optional[str]
    destination_station: Optional[str]
    origin_index: Optional[int]
    destination_index: Optional[int]
    status: str
    timeline_station_count: int


class SectionEvidenceBuilder:
    """
    Builds section mapping evidence from real RailKit capture files.

    Reads all live_train_*.json files from the captures directory,
    extracts station sequences from timelines, and checks each corridor
    section for ordered station pairs.

    Does NOT modify config/railkit_section_mapping.json.
    """

    def __init__(self, captures_dir: str = "data/raw_real/railkit"):
        self.captures_dir = captures_dir
        self.sections = CORRIDOR_SECTIONS

    def _load_capture(self, filepath: str) -> dict:
        """Load a single RailKit capture JSON file."""
        with open(filepath, "r") as f:
            return json.load(f)

    def _extract_timeline(self, data: dict) -> tuple:
        """
        Extract the timeline array and train metadata from a capture.

        RailKit JSON structure (verified):
            response.data.timeline[]
            response.data.trainNo
            response.data.trainName
            response.data.totalStations
        """
        if "response" in data:
            resp = data["response"]
            if "data" in resp:
                train_data = resp["data"]
                return train_data.get("timeline", []), train_data

        if "data" in data:
            train_data = data["data"]
            return train_data.get("timeline", []), train_data

        if "timeline" in data:
            return data["timeline"], data

        return [], data

    def _build_station_index(self, timeline: list) -> dict:
        """
        Build a mapping from station code to list of timeline indices.

        Returns: {station_code: [index0, index1, ...]}
        """
        station_indices = {}
        for i, entry in enumerate(timeline):
            code = entry.get("stationCode", "")
            if code:
                if code not in station_indices:
                    station_indices[code] = []
                station_indices[code].append(i)
        return station_indices

    def _find_station(self, station_codes: list, station_indices: dict) -> tuple:
        """
        Find the first occurrence of any of the given station codes.

        Returns: (found_index, actual_code) or (None, None)
        """
        for code in station_codes:
            if code in station_indices:
                return station_indices[code][0], code
        return None, None

    def extract_evidence_from_capture(self, filepath: str) -> list:
        """
        Extract section evidence records from a single capture file.

        Returns a list of SectionEvidence records.
        """
        fname = os.path.basename(filepath)
        data = self._load_capture(filepath)
        timeline, train_data = self._extract_timeline(data)

        train_no = str(
            data.get("train_number")
            or train_data.get("trainNo", "UNKNOWN")
        )
        timeline_count = len(timeline)

        if timeline_count == 0:
            return []

        station_indices = self._build_station_index(timeline)
        evidence_records = []

        for section_id, origin_codes, dest_codes in self.sections:
            orig_idx, orig_actual = self._find_station(origin_codes, station_indices)
            dest_idx, dest_actual = self._find_station(dest_codes, station_indices)

            if orig_idx is not None and dest_idx is not None:
                if orig_idx < dest_idx:
                    status = OBSERVED_ORDERED
                elif orig_idx == dest_idx:
                    status = INVALID_SAME_POSITION
                else:
                    status = REVERSE_ORDER
            elif orig_idx is not None:
                status = ORIGIN_ONLY
            elif dest_idx is not None:
                status = DESTINATION_ONLY
            else:
                status = NOT_OBSERVED

            # Only record evidence where at least one station was found
            if status != NOT_OBSERVED:
                evidence_records.append(SectionEvidence(
                    train_number=train_no,
                    capture_file=fname,
                    section_id=section_id,
                    origin_station=orig_actual,
                    destination_station=dest_actual,
                    origin_index=orig_idx,
                    destination_index=dest_idx,
                    status=status,
                    timeline_station_count=timeline_count,
                ))

        return evidence_records

    def build_all_evidence(self) -> list:
        """
        Extract section evidence from all capture files.

        Returns a list of SectionEvidence records across all captures.
        """
        pattern = os.path.join(self.captures_dir, "live_train_*.json")
        files = sorted(glob.glob(pattern))

        all_evidence = []
        for filepath in files:
            records = self.extract_evidence_from_capture(filepath)
            all_evidence.extend(records)

        return all_evidence

    def evidence_summary(self, evidence: list) -> dict:
        """
        Summarize evidence by section.

        Returns: {section_id: {
            "total": count,
            "observed_ordered": count,
            "trains": [train_numbers],
            "status": "STRONG" | "MODERATE" | "WEAK" | "NO_EVIDENCE"
        }}
        """
        from collections import defaultdict

        by_section = defaultdict(list)
        for ev in evidence:
            by_section[ev.section_id].append(ev)

        summary = {}
        for section_id, _, _ in self.sections:
            records = by_section.get(section_id, [])
            ordered = [r for r in records if r.status == OBSERVED_ORDERED]
            trains = list(set(r.train_number for r in ordered))

            if len(ordered) >= 2 and len(trains) >= 2:
                strength = "STRONG"
            elif len(ordered) >= 2:
                strength = "STRONG"
            elif len(ordered) == 1:
                strength = "MODERATE"
            else:
                strength = "WEAK" if records else "NO_EVIDENCE"

            summary[section_id] = {
                "total": len(records),
                "observed_ordered": len(ordered),
                "trains": trains,
                "strength": strength,
            }

        return summary

    def write_evidence_csv(
        self,
        evidence: list,
        output_path: str = "data/processed_real/railkit_section_mapping_evidence.csv",
    ) -> str:
        """
        Write evidence records to CSV.

        Returns the output path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVIDENCE_FIELDS)
            writer.writeheader()
            for ev in evidence:
                row = asdict(ev)
                # Replace None with empty string for CSV clarity
                row = {k: ("" if v is None else v) for k, v in row.items()}
                writer.writerow(row)

        return output_path
