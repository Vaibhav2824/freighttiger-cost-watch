"""The graded numbers: sample rows, an independent brute-force recomputation, no look-ahead, input checks."""
import csv
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd
import pytest

from assistant.metrics import HISTORY_WEEKS, load_shipments, weekly_metrics
from assistant.pipeline import SAMPLE_FORMAT, SHIPMENTS, output_columns, output_row
from assistant.verdict import Verdict


@pytest.fixture(scope="module")
def weekly():
    return weekly_metrics(load_shipments(SHIPMENTS))


def reference_metrics(path) -> dict:
    """Brute force with csv + dicts only, written from the brief's wording; shares no code with metrics.py."""
    cost, tkm, rtype = defaultdict(float), defaultdict(float), {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = date.fromisoformat(r["shipment_date"])
            key = (f"{r['origin']}-{r['destination']}", d - timedelta(days=d.weekday()))
            cost[key] += float(r["freight_cost_inr"])
            tkm[key] += float(r["quantity_tonnes"]) * float(r["distance_km"])
            rtype[key[0]] = r["route_type"]
    cpk = {k: cost[k] / tkm[k] for k in cost}
    out = {}
    for (route, week), value in cpk.items():
        prior = [cpk[(route, week - timedelta(weeks=i))] for i in range(1, HISTORY_WEEKS + 1)
                 if (route, week - timedelta(weeks=i)) in cpk]
        peers = [v for (r, w), v in cpk.items() if w == week and r != route and rtype[r] == rtype[route]]
        out[(route, week)] = (value, sum(prior) / len(prior) if prior else None,
                              sum(peers) / len(peers) if peers else None)
    return out


def test_matches_independent_reference_on_every_route_week(weekly):
    ref = reference_metrics(SHIPMENTS)
    assert len(ref) == len(weekly) == 728
    for r in weekly.itertuples():
        value, hist, peer = ref[(r.route, r.week_of.date())]
        assert r.cost_per_tonne_km == pytest.approx(value, rel=1e-12)
        assert (pd.isna(r.own_history_avg) and hist is None) or r.own_history_avg == pytest.approx(hist, rel=1e-12)
        assert r.peer_avg == pytest.approx(peer, rel=1e-12)


def test_reproduces_sample_output_rows_exactly(weekly):
    # csv module, first 5 fields only: the sample's 3rd row has unquoted commas in its free-text reason.
    with open(SAMPLE_FORMAT, newline="", encoding="utf-8") as f:
        sample = [line[:5] for line in list(csv.reader(f))[1:]]
    assert len(sample) == 3
    for route, week, *expected in sample:
        r = next(weekly[(weekly.route == route) & (weekly.week_of == week)].itertuples())
        out = output_row(r, Verdict(False, None, None, ()), "")
        assert [out["cost_per_tonne_km"], out["vs_own_history"], out["vs_similar_routes"]] == expected
        assert r.is_anomaly  # every sample row is one our flag rule raises


def test_cost_always_has_two_decimals(weekly):
    r = next(weekly.itertuples())
    out = output_row(r._replace(cost_per_tonne_km=4.1), Verdict(False, None, None, ()), "")
    assert out["cost_per_tonne_km"] == "4.10"


def test_own_history_has_no_look_ahead():
    ships = load_shipments(SHIPMENTS)
    cutoff = pd.Timestamp("2025-01-01")
    tampered = ships.assign(freight_cost_inr=ships.freight_cost_inr.where(ships.shipment_date < cutoff,
                                                                          ships.freight_cost_inr * 10))
    before, after = weekly_metrics(ships), weekly_metrics(tampered)
    past = before.week_of < cutoff - pd.Timedelta(days=6)  # weeks that end before the tampering starts
    pd.testing.assert_series_equal(before[past].own_history_avg, after[past].own_history_avg)
    pd.testing.assert_series_equal(before[past].peer_avg, after[past].peer_avg)


def test_short_history_uses_all_prior_weeks(weekly):
    r = weekly[weekly.route == "Mumbai-Pune"].reset_index(drop=True)
    assert pd.isna(r.own_history_avg[0]) and r.history_weeks_used[0] == 0
    assert r.own_history_avg[3] == pytest.approx(r.cost_per_tonne_km[:3].mean())
    assert r.own_history_avg[20] == pytest.approx(r.cost_per_tonne_km[12:20].mean())  # 8 strictly prior
    assert not r.is_anomaly[0]  # no history -> cannot be called "rising"


def test_episodes_group_a_slow_creep(weekly):
    mp = weekly[(weekly.route == "Mumbai-Pune") & weekly.is_anomaly]
    late = mp[mp.week_of >= "2025-09-15"]
    assert (late.episode_start == pd.Timestamp("2025-09-15")).all()
    assert list(late.episode_week_no) == list(range(1, len(late) + 1))
    assert late.vs_pre_episode_pct.iloc[-1] > late.vs_own_pct.iloc[-1]  # the creep the 8-week baseline hides


def test_output_columns_are_the_sample_contract():
    assert output_columns() == ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
                                "flagged", "matched_note_id", "reason"]


@pytest.mark.parametrize("change, message", [
    (lambda df: df.drop(columns=["distance_km"]), "missing columns"),
    (lambda df: df.assign(quantity_tonnes=0), "non-positive"),
    (lambda df: df.assign(freight_cost_inr=None), "missing values"),
])
def test_bad_input_fails_fast_with_a_clear_message(tmp_path, change, message):
    path = tmp_path / "bad.csv"
    change(pd.read_csv(SHIPMENTS, nrows=5)).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_shipments(path)
