"""Weekly cost per tonne-km per route, the two graded baselines, and the flag rule."""
import pandas as pd

HISTORY_WEEKS = 8      # "vs own history" window, fixed by the brief
OWN_RISE_PCT = 10.0    # flag when >= this much above the route's own trailing average ...
PEER_RISE_PCT = 20.0   # ... or when rising AND >= this much above same-type peers
EPISODE_GAP_WEEKS = 4  # flagged weeks this close together on a route count as one ongoing episode

REQUIRED_COLUMNS = ["shipment_id", "origin", "destination", "route_type", "quantity_tonnes",
                    "distance_km", "freight_cost_inr", "shipment_date"]


def load_shipments(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    bad = (df[REQUIRED_COLUMNS].isna().any(axis=1)
           | (df.quantity_tonnes <= 0) | (df.distance_km <= 0) | (df.freight_cost_inr < 0))
    if bad.any():
        raise ValueError(f"{path}: {bad.sum()} rows have missing values or non-positive "
                         f"quantity/distance, e.g. {df.loc[bad, 'shipment_id'].head(3).tolist()}")
    return df


def weekly_metrics(shipments: pd.DataFrame) -> pd.DataFrame:
    """One row per route-week with cost/tonne-km, both baselines and the anomaly flags."""
    df = shipments.assign(
        route=shipments.origin + "-" + shipments.destination,
        week_of=shipments.shipment_date - pd.to_timedelta(shipments.shipment_date.dt.weekday, unit="D"),
        tonne_km=shipments.quantity_tonnes * shipments.distance_km,
    )
    weekly = (df.groupby(["route", "route_type", "week_of"], as_index=False)
                .agg(freight_cost=("freight_cost_inr", "sum"), tonne_km=("tonne_km", "sum"),
                     shipments=("shipment_id", "size"))
                .sort_values(["route", "week_of"], ignore_index=True))
    cpk = weekly.freight_cost / weekly.tonne_km  # total cost / total tonne-km (matches the sample rows)
    weekly["cost_per_tonne_km"] = cpk

    # Calendar window [week - 8w, week): strictly prior weeks only, robust to gaps in the data.
    trailing = (weekly.set_index("week_of").groupby("route").cost_per_tonne_km
                      .rolling(f"{7 * HISTORY_WEEKS}D", closed="left"))
    weekly["own_history_avg"] = trailing.mean().to_numpy()
    weekly["history_weeks_used"] = trailing.count().fillna(0).astype(int).to_numpy()

    same_week_type = weekly.groupby(["route_type", "week_of"]).cost_per_tonne_km
    total, n = same_week_type.transform("sum"), same_week_type.transform("count")
    weekly["peer_avg"] = ((total - cpk) / (n - 1)).where(n > 1)  # excludes the route itself
    weekly["peer_routes"] = n - 1

    # Flags are decided on the same 1-decimal values the output shows.
    weekly["vs_own_pct"] = ((cpk / weekly.own_history_avg - 1) * 100).round(1)
    weekly["vs_peer_pct"] = ((cpk / weekly.peer_avg - 1) * 100).round(1)
    weekly["anomaly_own"] = weekly.vs_own_pct >= OWN_RISE_PCT
    weekly["anomaly_peer"] = (weekly.vs_own_pct > 0) & (weekly.vs_peer_pct >= PEER_RISE_PCT)
    weekly["is_anomaly"] = weekly.anomaly_own | weekly.anomaly_peer
    return add_episodes(weekly)


def add_episodes(weekly: pd.DataFrame) -> pd.DataFrame:
    """Group a route's flagged weeks into episodes (no more than EPISODE_GAP_WEEKS apart), so a slow creep
    reads as one story: 'nth flagged week since X, +Y% above the level before it began'."""
    weekly = weekly.assign(episode_start=pd.NaT, episode_week_no=0, vs_pre_episode_pct=float("nan"))
    for _, rows in weekly[weekly.is_anomaly].groupby("route"):
        start, prev, n = None, None, 0
        for i, r in rows.iterrows():
            if prev is None or (r.week_of - prev).days > 7 * EPISODE_GAP_WEEKS:
                start, n = r.week_of, 0
                baseline = r.own_history_avg  # the route's level in the 8 weeks before the episode
            n, prev = n + 1, r.week_of
            weekly.loc[i, ["episode_start", "episode_week_no", "vs_pre_episode_pct"]] = (
                start, n, round((r.cost_per_tonne_km / baseline - 1) * 100, 1))
    return weekly
