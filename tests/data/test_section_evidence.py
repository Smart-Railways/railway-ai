"""
Tests for SectionEvidenceBuilder — Phase 5.7

Validates:
- Correct extraction from real RailKit capture JSON structure
- Proper handling of renamed station codes (JHS→VGLJ, VAD→BRC, SRT→ST)
- Evidence status classification (OBSERVED_ORDERED, ORIGIN_ONLY, etc.)
- Evidence summary strength ratings
- CSV output format
"""

import json
import os
import tempfile
import csv
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.section_evidence_builder import (
    SectionEvidenceBuilder,
    SectionEvidence,
    OBSERVED_ORDERED,
    ORIGIN_ONLY,
    DESTINATION_ONLY,
    NOT_OBSERVED,
    REVERSE_ORDER,
)


def _make_timeline_entry(station_code, station_name="Test Station"):
    """Helper to create a minimal timeline entry."""
    return {
        "stationCode": station_code,
        "stationName": station_name,
        "type": "station",
        "status": "on_time",
    }


def _make_capture_json(train_number, timeline_stations):
    """
    Helper to create a minimal RailKit capture JSON structure.
    
    Matches the verified real structure:
        response.data.timeline[]
        response.data.trainNo
    """
    timeline = [_make_timeline_entry(code) for code in timeline_stations]
    return {
        "train_number": str(train_number),
        "response": {
            "data": {
                "trainNo": str(train_number),
                "trainName": f"TEST TRAIN {train_number}",
                "totalStations": len(timeline),
                "timeline": timeline,
            }
        },
    }


def _write_capture_file(tmpdir, train_number, timeline_stations, suffix=""):
    """Write a capture JSON to a temp directory and return the path."""
    data = _make_capture_json(train_number, timeline_stations)
    fname = f"live_train_{train_number}_test{suffix}.json"
    path = os.path.join(tmpdir, fname)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestTimelineExtraction:
    """Test that timeline is correctly extracted from various JSON structures."""

    def test_response_data_timeline_structure(self):
        """Verify extraction from real response.data.timeline structure."""
        builder = SectionEvidenceBuilder()
        data = _make_capture_json("12002", ["NDLS", "MTJ", "AGC"])
        timeline, train_data = builder._extract_timeline(data)
        assert len(timeline) == 3
        assert train_data["trainNo"] == "12002"

    def test_empty_timeline(self):
        """Handle capture with no timeline gracefully."""
        builder = SectionEvidenceBuilder()
        data = {"response": {"data": {"trainNo": "99999", "timeline": []}}}
        timeline, _ = builder._extract_timeline(data)
        assert len(timeline) == 0

    def test_missing_response_key(self):
        """Handle unexpected structure without crashing."""
        builder = SectionEvidenceBuilder()
        data = {"something_else": {}}
        timeline, _ = builder._extract_timeline(data)
        assert len(timeline) == 0


class TestStationIndexBuilding:
    """Test station code to index mapping."""

    def test_simple_station_sequence(self):
        builder = SectionEvidenceBuilder()
        timeline = [
            _make_timeline_entry("NDLS"),
            _make_timeline_entry("MTJ"),
            _make_timeline_entry("AGC"),
        ]
        indices = builder._build_station_index(timeline)
        assert indices["NDLS"] == [0]
        assert indices["MTJ"] == [1]
        assert indices["AGC"] == [2]

    def test_duplicate_station_codes(self):
        """Some stations may appear multiple times (arrival + departure)."""
        builder = SectionEvidenceBuilder()
        timeline = [
            _make_timeline_entry("NDLS"),
            _make_timeline_entry("NDLS"),
            _make_timeline_entry("MTJ"),
        ]
        indices = builder._build_station_index(timeline)
        assert indices["NDLS"] == [0, 1]

    def test_empty_station_code_ignored(self):
        builder = SectionEvidenceBuilder()
        timeline = [
            _make_timeline_entry("NDLS"),
            {"stationCode": "", "stationName": "?"},
            _make_timeline_entry("MTJ"),
        ]
        indices = builder._build_station_index(timeline)
        assert "" not in indices
        assert len(indices) == 2


class TestRenamedStationCodes:
    """
    Verify that renamed station codes are handled correctly.
    
    Critical mappings discovered from real data:
        JHS  -> VGLJ  (Virangana Lakshmibai Jhansi Jn)
        VAD  -> BRC   (Vadodara / Baroda)
        SRT  -> ST    (Surat)
        MUM  -> MMCT  (Mumbai Central)
    """

    def test_jhs_as_vglj(self):
        """GWL-JHS-01 should match GWL -> VGLJ (not JHS)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "12002", [
                "NDLS", "MTJ", "AGC", "GWL",
                "STLA", "STLI",  # intermediate stations
                "VGLJ",  # This is Jhansi in RailKit
                "BJI", "KHJ",
                "BINA", "BPL",
            ])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()

            gwl_jhs = [e for e in evidence
                       if e.section_id == "GWL-JHS-01"
                       and e.status == OBSERVED_ORDERED]
            assert len(gwl_jhs) == 1
            assert gwl_jhs[0].origin_station == "GWL"
            assert gwl_jhs[0].destination_station == "VGLJ"

    def test_jhs_bina_as_vglj_bina(self):
        """JHS-BINA-01 should match VGLJ -> BINA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "12002", [
                "NDLS", "MTJ", "AGC", "GWL",
                "VGLJ", "BINA", "BPL",
            ])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()

            jhs_bina = [e for e in evidence
                        if e.section_id == "JHS-BINA-01"
                        and e.status == OBSERVED_ORDERED]
            assert len(jhs_bina) == 1
            assert jhs_bina[0].origin_station == "VGLJ"
            assert jhs_bina[0].destination_station == "BINA"

    def test_vad_as_brc(self):
        """RTM-VAD-01 should match RTM -> BRC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "12904", [
                "RTM", "BRC", "ST",
            ])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()

            rtm_vad = [e for e in evidence
                       if e.section_id == "RTM-VAD-01"
                       and e.status == OBSERVED_ORDERED]
            assert len(rtm_vad) == 1
            assert rtm_vad[0].destination_station == "BRC"

    def test_srt_as_st(self):
        """VAD-SRT-01 should match BRC -> ST."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "22946", [
                "BRC", "ST", "MMCT",
            ])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()

            vad_srt = [e for e in evidence
                       if e.section_id == "VAD-SRT-01"
                       and e.status == OBSERVED_ORDERED]
            assert len(vad_srt) == 1
            assert vad_srt[0].origin_station == "BRC"
            assert vad_srt[0].destination_station == "ST"

    def test_mum_as_mmct(self):
        """SRT-MUM-01 should match ST -> MMCT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "22946", [
                "BRC", "ST", "MMCT",
            ])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()

            srt_mum = [e for e in evidence
                       if e.section_id == "SRT-MUM-01"
                       and e.status == OBSERVED_ORDERED]
            assert len(srt_mum) == 1
            assert srt_mum[0].destination_station == "MMCT"


class TestEvidenceStatusClassification:
    """Test correct status assignment for different station pair scenarios."""

    def test_observed_ordered(self):
        """Both stations present in correct order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "12002", ["NDLS", "MTJ", "AGC"])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()

            ndl_mtj = [e for e in evidence
                       if e.section_id == "NDL-MTJ-01"
                       and e.status == OBSERVED_ORDERED]
            assert len(ndl_mtj) == 1

    def test_origin_only(self):
        """Only origin station present, destination missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train that has GWL but not VGLJ (JHS)
            _write_capture_file(tmpdir, "99999", ["GWL", "OTHER"])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()

            gwl_jhs = [e for e in evidence
                       if e.section_id == "GWL-JHS-01"]
            assert len(gwl_jhs) == 1
            assert gwl_jhs[0].status == ORIGIN_ONLY

    def test_destination_only(self):
        """Only destination station present, origin missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "99999", ["OTHER", "BINA"])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()

            jhs_bina = [e for e in evidence
                        if e.section_id == "JHS-BINA-01"]
            assert len(jhs_bina) == 1
            assert jhs_bina[0].status == DESTINATION_ONLY

    def test_reverse_order(self):
        """Both stations present but in reverse order (return journey)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "12001", ["MTJ", "NDLS"])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()

            ndl_mtj = [e for e in evidence
                       if e.section_id == "NDL-MTJ-01"]
            assert len(ndl_mtj) == 1
            assert ndl_mtj[0].status == REVERSE_ORDER

    def test_not_observed_excluded(self):
        """Sections with no station matches should not generate records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "99999", ["OTHER1", "OTHER2"])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()

            assert len(evidence) == 0


class TestEvidenceSummary:
    """Test evidence summary and strength classification."""

    def test_strong_evidence_multi_train(self):
        """Multiple ordered observations from different trains = STRONG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "12002", ["NDLS", "MTJ"], suffix="_1")
            _write_capture_file(tmpdir, "12904", ["NDLS", "MTJ"], suffix="_2")
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()
            summary = builder.evidence_summary(evidence)

            assert summary["NDL-MTJ-01"]["strength"] == "STRONG"
            assert summary["NDL-MTJ-01"]["observed_ordered"] == 2

    def test_moderate_evidence(self):
        """Single ordered observation = MODERATE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "22946", ["ST", "MMCT"])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()
            summary = builder.evidence_summary(evidence)

            assert summary["SRT-MUM-01"]["strength"] == "MODERATE"

    def test_no_evidence(self):
        """No observations at all = NO_EVIDENCE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "99999", ["OTHER1", "OTHER2"])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()
            summary = builder.evidence_summary(evidence)

            assert summary["NDL-MTJ-01"]["strength"] == "NO_EVIDENCE"


class TestCSVOutput:
    """Test CSV evidence file generation."""

    def test_csv_written_with_correct_headers(self):
        """Verify CSV output has correct columns and data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "12002", ["NDLS", "MTJ", "AGC"])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()

            output_path = os.path.join(tmpdir, "test_evidence.csv")
            builder.write_evidence_csv(evidence, output_path=output_path)

            assert os.path.exists(output_path)

            with open(output_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) > 0
            assert "train_number" in rows[0]
            assert "section_id" in rows[0]
            assert "status" in rows[0]
            assert "origin_station" in rows[0]
            assert "destination_station" in rows[0]

    def test_csv_none_values_as_empty_string(self):
        """None values (missing station) should be empty strings in CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_capture_file(tmpdir, "99999", ["GWL", "OTHER"])
            builder = SectionEvidenceBuilder(captures_dir=tmpdir)
            evidence = builder.build_all_evidence()

            output_path = os.path.join(tmpdir, "test_evidence.csv")
            builder.write_evidence_csv(evidence, output_path=output_path)

            with open(output_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            origin_only = [r for r in rows if r["status"] == "ORIGIN_ONLY"]
            assert len(origin_only) > 0
            assert origin_only[0]["destination_station"] == ""


class TestRealCaptureIntegration:
    """
    Integration test using actual capture files if available.

    Skipped if real data directory doesn't exist.
    """

    @pytest.mark.skipif(
        not os.path.exists("data/raw_real/railkit"),
        reason="Real capture data not available",
    )
    def test_real_captures_produce_evidence(self):
        """Verify real captures produce expected evidence structure."""
        builder = SectionEvidenceBuilder()
        evidence = builder.build_all_evidence()

        assert len(evidence) > 0

        # Every evidence record should have required fields
        for ev in evidence:
            assert ev.train_number
            assert ev.capture_file
            assert ev.section_id
            assert ev.status in [
                OBSERVED_ORDERED,
                ORIGIN_ONLY,
                DESTINATION_ONLY,
                REVERSE_ORDER,
            ]
            assert ev.timeline_station_count > 0

    @pytest.mark.skipif(
        not os.path.exists("data/raw_real/railkit"),
        reason="Real capture data not available",
    )
    def test_12002_covers_upper_corridor(self):
        """Train 12002 (Shatabdi) should cover NDLS through BPL."""
        builder = SectionEvidenceBuilder()
        evidence = builder.build_all_evidence()

        evidence_12002 = [e for e in evidence if e.train_number == "12002"]
        ordered_sections = set(
            e.section_id for e in evidence_12002
            if e.status == OBSERVED_ORDERED
        )

        # 12002 verified to cover NDLS[0]->MTJ[21]->AGC[29]->GWL[42]
        # ->VGLJ[55]->BINA[73]->BPL[91]
        expected = {
            "NDL-MTJ-01", "NDL-MTJ-02",
            "MTJ-AGC-01",
            "AGC-GWL-01",
            "GWL-JHS-01",
            "JHS-BINA-01",
            "BINA-BPL-01",
        }
        assert expected.issubset(ordered_sections), (
            f"Missing sections: {expected - ordered_sections}"
        )
