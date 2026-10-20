"""Consultation follow-up loss: stale student profile vs. signed result.

Reads the issued CSV only. Prints the figures cited in README.md and writes
the same tables under output/.

Judgments encoded here (also stated in README.md):
- Exact duplicate rows are dropped before any count.
- consult_time carries +08:00; profile_synced_at is UTC. Both are converted
  to UTC before subtraction.
- A profile is stale when lag >= 12 minutes. The issued file has no lag of
  9, 10, or 11 minutes, so any cutoff in (8, 12] selects the same rows.
- Empty follow_up_result means the result was not returned. Those rows stay
  out of loss rates and are not counted as losses. A warning can still fire
  on them, because lag is known when the consultation starts.
- Loss is priced at 120 yuan and one profile-stale warning at 15 yuan,
  as given in the task brief. No other prices are introduced.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "issued" / "consultations.csv"
OUT_DIR = ROOT / "output"

LOSS_YUAN = 120
WARN_YUAN = 15
STALE_MINUTES = 12
EXCLUDED_BRANCH = "华中教学部"


def load_issued():
    raw = pd.read_csv(DATA_PATH)
    n_raw = len(raw)
    n_dup_rows = int(raw.duplicated().sum())
    df = raw.drop_duplicates().copy()
    if df["consultation_id"].duplicated().any():
        raise SystemExit("duplicate consultation_id rows are not identical; stop")

    df["consult_time"] = pd.to_datetime(df["consult_time"], utc=True)
    df["profile_synced_at"] = pd.to_datetime(df["profile_synced_at"], utc=True)
    lag = (df["consult_time"] - df["profile_synced_at"]).dt.total_seconds() / 60
    if ((lag < 0) | (lag > 60) | (lag % 1 != 0)).any():
        raise SystemExit("profile lag is outside 0-60 whole minutes; stop")
    df["lag_min"] = lag.astype(int)
    df["stale"] = df["lag_min"] >= STALE_MINUTES
    df["known"] = df["follow_up_result"].notna()
    df["loss"] = df["follow_up_result"].eq(0)
    df["day"] = df["consult_time"].dt.tz_convert("Asia/Shanghai").dt.date

    sample = df.loc[df["consultation_id"] == 1948].iloc[0]
    if sample["lag_min"] != 4:
        raise SystemExit("timezone check failed for consultation_id 1948")
    return df, n_raw, n_dup_rows


def segment_stats(frame):
    known = frame[frame["known"]]
    fresh = known[~known["stale"]]
    stale = known[known["stale"]]
    fresh_n = len(fresh)
    fresh_loss = int(fresh["loss"].sum())
    stale_n = len(stale)
    stale_loss = int(stale["loss"].sum())
    expected_if_fresh = stale_n * fresh_loss / fresh_n
    excess_loss = stale_loss - expected_if_fresh
    alert_n = int(frame["stale"].sum())
    warn_cost = alert_n * WARN_YUAN
    excess_yuan = excess_loss * LOSS_YUAN
    observed_loss_yuan = stale_loss * LOSS_YUAN
    return {
        "rows": len(frame),
        "known": len(known),
        "unknown": int((~frame["known"]).sum()),
        "fresh_n": fresh_n,
        "fresh_loss": fresh_loss,
        "fresh_rate": fresh_loss / fresh_n,
        "stale_n": stale_n,
        "stale_loss": stale_loss,
        "stale_rate": stale_loss / stale_n,
        "expected_if_fresh": expected_if_fresh,
        "excess_loss": excess_loss,
        "excess_yuan": excess_yuan,
        "alert_n": alert_n,
        "warn_cost": warn_cost,
        "net_if_gap_closes": excess_yuan - warn_cost,
        "observed_loss_yuan": observed_loss_yuan,
        "net_if_all_stale_losses_saved": observed_loss_yuan - warn_cost,
        "breakeven_losses": warn_cost / LOSS_YUAN,
    }


def rate_table(frame, columns):
    known = frame[frame["known"]]
    grouped = known.groupby(columns, observed=True)
    out = grouped.agg(n=("loss", "size"), loss=("loss", "sum"))
    out["loss_rate"] = out["loss"] / out["n"]
    return out.reset_index()


def print_segment(title, stats):
    print(f"\n== {title} ==")
    print(f"rows {stats['rows']}  known {stats['known']}  unknown {stats['unknown']}")
    print(
        f"fresh {stats['fresh_loss']}/{stats['fresh_n']} = {stats['fresh_rate']:.6f}"
    )
    print(
        f"stale {stats['stale_loss']}/{stats['stale_n']} = {stats['stale_rate']:.6f}"
    )
    print(
        f"alerts(incl unknown) {stats['alert_n']}  warn_cost {stats['warn_cost']}"
    )
    print(
        f"stale losses if fresh rate {stats['expected_if_fresh']:.4f}  "
        f"excess {stats['excess_loss']:.4f}  excess_yuan {stats['excess_yuan']:.2f}"
    )
    print(
        f"net if gap closes {stats['net_if_gap_closes']:.2f}  "
        f"breakeven losses {stats['breakeven_losses']:.4f}"
    )
    print(
        f"observed stale-loss yuan {stats['observed_loss_yuan']}  "
        f"net if every stale loss were saved {stats['net_if_all_stale_losses_saved']}"
    )


def print_no_dedup_sensitivity():
    """Same definitions, but keep the 57 copied rows. Money must not use this."""
    raw = pd.read_csv(DATA_PATH)
    raw["consult_time"] = pd.to_datetime(raw["consult_time"], utc=True)
    raw["profile_synced_at"] = pd.to_datetime(raw["profile_synced_at"], utc=True)
    raw["lag_min"] = (
        (raw["consult_time"] - raw["profile_synced_at"]).dt.total_seconds() / 60
    ).astype(int)
    raw["stale"] = raw["lag_min"] >= STALE_MINUTES
    known = raw[raw["follow_up_result"].notna()]
    fresh = known[~known["stale"]]
    stale = known[known["stale"]]
    print("\n== sensitivity: keep duplicate rows, known outcomes only ==")
    print(f"rows {len(raw)}  known {len(known)}  loss {int(known['follow_up_result'].eq(0).sum())}")
    print(
        f"fresh {int(fresh['follow_up_result'].eq(0).sum())}/{len(fresh)} "
        f"= {fresh['follow_up_result'].eq(0).mean():.6f}"
    )
    print(
        f"stale {int(stale['follow_up_result'].eq(0).sum())}/{len(stale)} "
        f"= {stale['follow_up_result'].eq(0).mean():.6f}"
    )


def main():
    df, n_raw, n_dup_rows = load_issued()
    known = df[df["known"]]
    rule = df[df["branch_office"] != EXCLUDED_BRANCH]
    huazhong = df[df["branch_office"] == EXCLUDED_BRANCH]

    print(f"raw_rows {n_raw}")
    print(f"exact_duplicate_rows {n_dup_rows}")
    print(f"dedup_rows {len(df)}")
    print(f"unique_consultation_id {df['consultation_id'].nunique()}")
    print(f"known {int(df['known'].sum())}  unknown {int((~df['known']).sum())}")
    print(f"loss {int(known['loss'].sum())}  loss_yuan {int(known['loss'].sum()) * LOSS_YUAN}")
    gap = ((df["lag_min"] >= 9) & (df["lag_min"] <= 11)).sum()
    print(f"lag_min min {df['lag_min'].min()} max {df['lag_min'].max()} gap_9_to_11 {int(gap)}")
    print(
        "window_shanghai "
        f"{df['day'].min()} .. {df['day'].max()}"
    )
    corr = known[["lag_min", "advisor_quota", "chat_round_count", "follow_up_result"]].corr()
    print("corr_lag_quota {:.6f}".format(corr.loc["lag_min", "advisor_quota"]))
    print("corr_lag_chat {:.6f}".format(corr.loc["lag_min", "chat_round_count"]))
    print("corr_lag_result {:.6f}".format(corr.loc["lag_min", "follow_up_result"]))
    print("corr_quota_result {:.6f}".format(corr.loc["advisor_quota", "follow_up_result"]))
    print("corr_chat_result {:.6f}".format(corr.loc["chat_round_count", "follow_up_result"]))

    all_stats = segment_stats(df)
    rule_stats = segment_stats(rule)
    hz_stats = segment_stats(huazhong)
    print_no_dedup_sensitivity()
    print_segment("all branches, deduped", all_stats)
    print_segment("warning rule: stale and branch != 华中教学部", rule_stats)
    print_segment("华中教学部 only", hz_stats)

    by_lag = rate_table(df, ["lag_min"])
    by_branch = rate_table(df, ["branch_office", "stale"])
    by_course = rate_table(df, ["course_category", "stale"])
    by_day = rate_table(rule, ["day", "stale"])

    print("\n== loss by branch x stale (known outcomes) ==")
    print(by_branch.to_string(index=False))
    print("\n== loss by day x stale, excluding 华中 (known outcomes) ==")
    print(by_day.to_string(index=False))

    OUT_DIR.mkdir(exist_ok=True)
    by_lag.to_csv(OUT_DIR / "loss_by_lag_minute.csv", index=False)
    by_branch.to_csv(OUT_DIR / "loss_by_branch.csv", index=False)
    by_course.to_csv(OUT_DIR / "loss_by_course.csv", index=False)
    by_day.to_csv(OUT_DIR / "loss_by_day_ex_huazhong.csv", index=False)

    summary = pd.DataFrame(
        [
            {"segment": "all", **all_stats},
            {"segment": "rule_ex_huazhong", **rule_stats},
            {"segment": "huazhong", **hz_stats},
        ]
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
