from wattwise_tariffs import (
    TariffModel,
    build_tariff_lookup,
    compute_bill_amount_paise,
    load_tariff_reference_tables,
)


def test_bescom_free_block_yields_only_fixed_charge() -> None:
    tables = load_tariff_reference_tables()
    tariffs = build_tariff_lookup(tables)
    bescom = tariffs["bescom"]

    amount = compute_bill_amount_paise(units_kwh=150, tariff=bescom, sanctioned_load_kw=6.0)
    assert amount == round(145.0 * 6.0 * 100)


def test_bescom_charges_above_free_block() -> None:
    tables = load_tariff_reference_tables()
    tariffs = build_tariff_lookup(tables)
    bescom = tariffs["bescom"]

    amount = compute_bill_amount_paise(units_kwh=300, tariff=bescom, sanctioned_load_kw=6.0)
    expected_rupees = 100 * 6.82 + 145.0 * 6.0
    assert amount == round(expected_rupees * 100)


def test_tneb_telescopic_slabs_apply_in_order() -> None:
    tables = load_tariff_reference_tables()
    tariffs = build_tariff_lookup(tables)
    tneb = tariffs["tneb"]

    # 150 units: 100 free + 50 at slab-2 rate (4.70)
    amount = compute_bill_amount_paise(units_kwh=150, tariff=tneb, sanctioned_load_kw=4.0)
    expected_rupees = 50 * 4.70
    assert amount == round(expected_rupees * 100)


def test_tod_generic_uses_blended_rate() -> None:
    tables = load_tariff_reference_tables()
    tariffs = build_tariff_lookup(tables)
    tod = tariffs["tod_generic"]
    assert tod.is_tod is True
    assert tod.tod_rate_per_unit > 0

    amount = compute_bill_amount_paise(units_kwh=100, tariff=tod, sanctioned_load_kw=3.0)
    assert amount == round(100 * tod.tod_rate_per_unit * 100)


def test_tariff_model_is_a_plain_dataclass_not_pandas() -> None:
    tables = load_tariff_reference_tables()
    tariffs = build_tariff_lookup(tables)
    assert isinstance(tariffs["tneb"], TariffModel)
    assert isinstance(tariffs["tneb"].slabs, list)


def test_compute_bill_amount_paise_is_zero_for_zero_units() -> None:
    tables = load_tariff_reference_tables()
    tariffs = build_tariff_lookup(tables)
    amount = compute_bill_amount_paise(units_kwh=0, tariff=tariffs["tneb"], sanctioned_load_kw=4.0)
    assert amount == 0
