# =============================================================================
# STEP 0  —  COMPILE THE CHINA DATASET FROM THE RAW CFDID   (file: 00_compile.py)
# =============================================================================
# WHY THIS STEP EXISTS
#   Our first compilation (team file, 279 events) kept PARENT (L1) rainstorm
#   events. A parent event can span up to 11 provinces (e.g. 2020-009), so its
#   AP / CAA / DEL are national totals for a whole flood season. The paper's
#   Table 3 shows its sample does NOT contain such totals:
#     - the paper's maximum AP (1510) and maximum CAA (135.2) are BOTH event
#       2007-027 (Anhui) in CFDID, to the decimal;
#     - four larger multi-province parent events (2008-017, 2011-019,
#       2017-005, 2020-009) would exceed that maximum, so the paper must have
#       them split by province (CFDID stores those splits as L2 sub-events);
#     - the paper's minimum DEL is 1.011 (100m CNY), i.e. nothing below
#       CNY 100 million: a "major rainstorm" floor.
#   So we rebuild the sample with three rules:
#     R1  rainstorm floods only, start year 2006-2020 (paper Sec. 3);
#     R2  province-level records: an L1 event that CFDID splits into L2
#         sub-events is replaced by those L2 records (L3 = sub-sub-events
#         inside a province are dropped, their L2 already contains them);
#         an L1 event with no sub-events is kept as it is;
#     R3  "major": reported direct economic loss >= CNY 100 million (nominal).
#   Severity file = records that also report AP and CAA (all three needed).
#   We test these rules against the paper's Table 3 in 00b_compare.py.
#
# FREQUENCY FILE (paper Sec. 3.4)
#   Number of records per year meeting R1-R3 (AP/CAA may be missing: a major
#   flood still happened). CFDID starts in 1994, so we get 1994-2020
#   (27 years) instead of the paper's 1986-2020.
#
# OUTPUT: data/clean/events.csv, data/clean/annual_counts.csv,
#         data/clean/events_team279.csv (old compilation, for comparison)
# =============================================================================
import numpy as np
import pandas as pd
import config as cfg

raw = pd.read_csv(cfg.RAW_DIR / "China Flood Disaster Impact Database.csv", encoding="utf-8-sig")
raw = raw.rename(columns={
    "Event level": "level", "Event ID": "event_id", "Start year": "year",
    "Province": "provinces", "Disaster category": "category",
    "Affected population (thousand persons)": "AP_k",
    "Affected cropland area (thousand ha)": "CAA_k",
    "Direct economic losses (million CNY)": "DEL_nom_m",
    "Direct economic losses (2023 constant prices, million CNY)": "DEL_2023_m",
    "Data source": "source"})

# Parent id of each L2/L3 record = id without its last "-NN"
raw["parent_id"] = np.where(raw.level == "L1", None, raw.event_id.str.rsplit("-", n=1).str[0])
has_children = set(raw.loc[raw.level == "L2", "parent_id"])

# Units of the paper: 10k persons, 10k ha, 100m CNY (2020 prices)
raw["AP"] = raw.AP_k / 10
raw["CAA"] = raw.CAA_k / 10
raw["DEL"] = raw.DEL_2023_m / 100 / cfg.CPI_2023_OVER_2020
raw["DEL_nominal"] = raw.DEL_nom_m / 100
raw["n_provinces"] = raw.provinces.astype(str).str.count(",") + 1


def province_level(df):
    """R2: L1 events without sub-events + all L2 sub-events."""
    return df[((df.level == "L1") & ~df.event_id.isin(has_children)) | (df.level == "L2")]


rain = raw[raw.category == "rainstorm flood"]                         # R1
major_all_years = province_level(rain)
major_all_years = major_all_years[major_all_years.DEL_nominal >= 1.0]  # R3

# ---- Severity file: 2006-2020, all three indicators reported and positive ----
sev = major_all_years[(major_all_years.year >= 2006) & (major_all_years.year <= 2020)]
sev = sev.dropna(subset=cfg.INDICATORS)
sev = sev[(sev[cfg.INDICATORS] > 0).all(axis=1)]
keep = ["event_id", "level", "year", "provinces", "n_provinces", "source",
        "AP", "CAA", "DEL", "DEL_nominal"]
sev[keep].to_csv(cfg.CLEAN_DIR / "events.csv", index=False)

# ---- Frequency file: 1994-2020 counts of major province-level rainstorms ----
# 2021-2023 also computed: NOT used for fitting, only to check the forecast.
years = range(1994, 2024)
counts = (major_all_years.groupby("year").size()
          .reindex(years, fill_value=0).rename("n_major").reset_index())
counts["n_complete"] = sev.groupby("year").size().reindex(years, fill_value=0).values
counts.to_csv(cfg.CLEAN_DIR / "annual_counts.csv", index=False)

# ---- The team's original compilation, kept for comparison ----
team = raw[(raw.category == "rainstorm flood") & (raw.level == "L1") &
           raw.year.between(2006, 2020)].dropna(subset=cfg.INDICATORS)
team[keep].to_csv(cfg.CLEAN_DIR / "events_team279.csv", index=False)

print("================ STEP 0 REPORT ================")
print(f"Team compilation (L1 parents, no floor): {len(team)} events  (expected 279)")
print(f"New compilation (province-level, DEL >= CNY 100m): {len(sev)} events  (paper: 245)")
print(f"   of which L1 kept whole: {(sev.level == 'L1').sum()}, L2 province sub-events: {(sev.level == 'L2').sum()}")
print(f"   multi-province records still present: {(sev.n_provinces > 1).sum()}")
print(f"Paper fingerprint 2007-027 (AP 1510, CAA 135.2) present: {'2007-027' in set(sev.event_id)}")
c20 = counts[counts.year <= 2020]
print(f"Annual major-rainstorm counts 1994-2020: mean {c20.n_major.mean():.1f}, "
      f"min {c20.n_major.min()}, max {c20.n_major.max()}")
print(counts.to_string(index=False))
