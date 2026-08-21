"""Tests for the configuration layer (TyreCompound, Track, Car, loader)."""

import pytest

from f1strategist.config.tyre_compound import TyreCompound
from f1strategist.config.track import Track
from f1strategist.config.car import Car
from f1strategist.config.loader import load_tracks, load_compounds, load_cars


class TestTyreCompound:
    def test_degradation_matches_blueprint_worked_example(self):
        """Blueprint §7.1 example (Soft: alpha=0.04, cliff=18, beta=0.12)."""
        soft = TyreCompound(name="Soft", deg_coeff=0.04, cliff_threshold=18)

        assert soft.degradation(5) == pytest.approx(0.04 * 25)          # 1.0
        assert soft.degradation(12) == pytest.approx(0.04 * 144)        # 5.8
        assert soft.degradation(18) == pytest.approx(0.04 * 324)        # 13.0 (cliff)
        assert soft.degradation(22) == pytest.approx(12.96 + 0.12 * 16)  # 14.9
        assert soft.degradation(25) == pytest.approx(12.96 + 0.12 * 49)  # 18.9

    def test_degradation_is_zero_below_age_one(self):
        soft = TyreCompound("Soft", 0.04, 18)
        assert soft.degradation(0) == 0.0

    def test_degradation_accelerates_past_cliff(self):
        soft = TyreCompound("Soft", 0.04, 18)
        # Blueprint model: beyond the cliff the penalty is α·t² + β·(age−t)²,
        # so the marginal per-lap penalty keeps growing and eventually exceeds
        # the pre-cliff rate (for Soft that happens from age ~25 onward).
        pre_cliff_marginal = soft.degradation(18) - soft.degradation(17)  # 1.40 s/lap
        late_marginal = soft.degradation(26) - soft.degradation(25)       # 1.80 s/lap
        assert late_marginal > pre_cliff_marginal
        # Degradation is strictly increasing — tyres never get faster with age.
        for age in range(1, 40):
            assert soft.degradation(age) > soft.degradation(age - 1)

    def test_negative_deg_coeff_rejected(self):
        with pytest.raises(ValueError):
            TyreCompound("Bad", -0.1, 18)


class TestTrack:
    def test_factory_lookup_by_name(self):
        monza = Track("Monza", 82.0, 53, 22.0, 0.12)
        monaco = Track("Monaco", 74.0, 78, 25.0, 0.30)
        assert Track.from_name("monaco", [monza, monaco]) is monaco

    def test_unknown_track_raises(self):
        with pytest.raises(KeyError):
            Track.from_name("Baku", [Track("Monza", 82.0, 53)])

    def test_invalid_probability_rejected(self):
        with pytest.raises(ValueError):
            Track("Bad", 82.0, 53, sc_probability=1.5)


class TestCar:
    def test_fuel_penalty_scales_with_remaining_fuel(self):
        car = Car("default", fuel_load_kg=110.0, fuel_burn_per_lap=1.5,
                  fuel_time_effect=0.03, driver_sigma=0.15)
        assert car.fuel_time_penalty(110.0) == pytest.approx(3.3)
        assert car.fuel_time_penalty(0.0) == 0.0
        # Heavier car = slower lap, but lighter as fuel burns off (FR-3).
        assert car.fuel_time_penalty(80.0) < car.fuel_time_penalty(110.0)


class TestLoader:
    def test_loads_all_config_entities(self):
        tracks = load_tracks()
        compounds = load_compounds()
        cars = load_cars()
        assert {t.name for t in tracks} >= {"Monza", "Monaco"}
        assert {c.name for c in compounds} >= {"Soft", "Medium", "Hard"}
        assert len(cars) >= 1
