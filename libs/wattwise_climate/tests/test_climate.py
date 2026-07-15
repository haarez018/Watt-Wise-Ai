from wattwise_climate import (
    DEFAULT_ZONE,
    city_to_zone,
    load_climate_reference_tables,
    zone_month_avg_temp,
)


def test_load_climate_reference_tables_has_expected_columns() -> None:
    tables = load_climate_reference_tables()
    assert {"city", "state", "zone"} <= set(tables["cities"].columns)
    assert "zone" in tables["zone_temps"].columns
    assert "jan" in tables["zone_temps"].columns


def test_city_to_zone_matches_known_city() -> None:
    tables = load_climate_reference_tables()
    known_city = str(tables["cities"].iloc[0]["city"])
    expected_zone = str(tables["cities"].iloc[0]["zone"])
    assert city_to_zone(known_city) == expected_zone


def test_city_to_zone_is_case_and_whitespace_insensitive() -> None:
    tables = load_climate_reference_tables()
    known_city = str(tables["cities"].iloc[0]["city"])
    expected_zone = str(tables["cities"].iloc[0]["zone"])
    assert city_to_zone(f"  {known_city.upper()}  ") == expected_zone


def test_city_to_zone_falls_back_to_default_for_unknown_city() -> None:
    assert city_to_zone("Nonexistent City XYZ") == DEFAULT_ZONE


def test_city_to_zone_falls_back_to_default_for_missing_city() -> None:
    assert city_to_zone(None) == DEFAULT_ZONE
    assert city_to_zone("") == DEFAULT_ZONE


def test_zone_month_avg_temp_returns_a_value_for_every_month() -> None:
    for month_index in range(1, 13):
        temp = zone_month_avg_temp(DEFAULT_ZONE, month_index)
        assert -10.0 < temp < 55.0  # sanity band for Indian monthly averages


def test_zone_month_avg_temp_falls_back_to_default_zone_for_unknown_zone() -> None:
    fallback = zone_month_avg_temp("nonexistent_zone", 6)
    expected = zone_month_avg_temp(DEFAULT_ZONE, 6)
    assert fallback == expected
