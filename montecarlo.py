#!/usr/bin/env python3
"""
Part 3 — Miyawaki Forest Monte Carlo uncertainty analysis
Group ID: 11_2

Run from this folder:
    python part3_uncertainty.py

Required packages:
    pandas, numpy, matplotlib, openpyxl

The script reads the three source workbooks included in this folder, validates
Group 11_2 records, runs Monte Carlo simulations for the generic Chave equation,
and writes CSV summaries and a single mean-with-95%-interval plot.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================== EDITABLE SETTINGS ==============================

GROUP_ID = "11_2"
N_SIMULATIONS = 25_000
RANDOM_SEED = 20261104

# Input uncertainty distributions from "Sources of uncertainty.xlsx".
DBH_CV = 0.025
HEIGHT_CV = 0.10
WOOD_DENSITY_CV = 0.10
CHAVE_RESIDUAL_LOG_SIGMA = 0.357

# The Chave residual is applied as exp(epsilon).
# No additional residual bias correction is applied.
CARBON_FRACTION = 0.47
CO2_PER_CARBON = 44.0 / 12.0

# Assignment Part 2 constants, used only to propagate Part 3 uncertainty
# into an optional annual per-hectare CO2-rate interval.
TREE_DENSITY_PER_M2 = 0.75
STAND_AGE_YEARS = 5.0
PROPAGATE_TO_ANNUAL_RATE = True

# Tree 550 is marked measurable, but the comment says it may be dead.
# It is included in the primary analysis because death is not confirmed.
# Set False if field/photo verification confirms that it was dead.
INCLUDE_POSSIBLY_DEAD_TREE_550 = True

# Chave model domain note from the lecture: calibration used stems >= 5 cm DBH.
# Keep smaller stems in the required analysis, but flag them as extrapolations.
CHAVE_MIN_DBH_CM_FOR_FLAG = 5.0

# File names (keep these source files beside this script).
OBSERVATIONS_FILE = "observations.xlsx"
ALLOMETRIC_FILE = "allometric_equations.xlsx"
UNCERTAINTY_FILE = "sources_of_uncertainty.xlsx"

OUTPUT_DIR = Path("part3_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================== DATA PREPARATION ==============================

def canonical_species(name):
    """Normalize the two known spelling mismatches in the supplied workbooks."""
    s = str(name).strip().lower()

    aliases = {
        "simarouba glauca": "simarauba glauca dc.",
        "swietenia mahagoni": "switenia mahagony (l.) jacq.",
    }

    return aliases.get(s, s)


def positive_normal(rng, mean, sd, n):
    """Draw from a normal distribution, rejecting non-positive values."""
    vals = rng.normal(mean, sd, n)
    bad = vals <= 0

    while np.any(bad):
        vals[bad] = rng.normal(mean, sd, int(bad.sum()))
        bad = vals <= 0

    return vals


def load_and_validate():

    obs = pd.read_excel(
        OBSERVATIONS_FILE,
        sheet_name="group data"
    )

    density = pd.read_excel(
        ALLOMETRIC_FILE,
        sheet_name="wood_density"
    )

    group = obs.loc[
        obs["GroupID"].astype(str) == GROUP_ID
    ].copy()

    if group.empty:
        raise ValueError(
            f"No observations found for GroupID {GROUP_ID}."
        )

    if group["TreeID"].duplicated().any():
        dupes = group.loc[
            group["TreeID"].duplicated(False),
            "TreeID"
        ].tolist()

        raise ValueError(
            f"Duplicate TreeIDs in this group's records: {dupes}"
        )

    # Exclude trees that do not exist or could not be measured.
    measured = group.loc[
        group["TreeExists"].eq(True)
        & group["CanBeMeasured"].eq(True)
    ].copy()

    excluded = group.loc[
        ~(
            group["TreeExists"].eq(True)
            & group["CanBeMeasured"].eq(True)
        )
    ].copy()

    excluded["Exclusion_reason"] = np.where(
        ~excluded["TreeExists"].eq(True),
        "Tree marked as not existing/found",
        "CanBeMeasured=False; no usable stem measurements"
    )

    if not INCLUDE_POSSIBLY_DEAD_TREE_550:

        excluded_550 = measured.loc[
            measured["TreeID"].eq(550)
        ].copy()

        if not excluded_550.empty:

            excluded_550["Exclusion_reason"] = (
                "Tree 550 excluded by user setting; possible dead status"
            )

            excluded = pd.concat(
                [excluded, excluded_550],
                ignore_index=True
            )

            measured = measured.loc[
                ~measured["TreeID"].eq(550)
            ].copy()

    # Build species density lookup from the supplied wood_density sheet.
    density = density.dropna(
        subset=["Species_name", "Wood_density(g/cm3)"]
    ).copy()

    density["species_key"] = density["Species_name"].map(
        canonical_species
    )

    density_map = dict(
        zip(
            density["species_key"],
            density["Wood_density(g/cm3)"]
        )
    )

    # The observations sheet uses shortened species names,
    # while the density workbook uses full botanical names.
    # Add first-two-word lookup keys.
    for _, drow in density.iterrows():

        full_key = str(drow["species_key"])
        short_key = " ".join(full_key.split()[:2])

        density_map[short_key] = float(
            drow["Wood_density(g/cm3)"]
        )

    stem_rows = []
    tree_rows = []

    for _, row in measured.sort_values("TreeID").iterrows():

        tid = int(row["TreeID"])
        species = str(row["TreeSpecies"]).strip()
        key = canonical_species(species)

        if key not in density_map:
            raise ValueError(
                f"Missing wood-density value for TreeID {tid}, "
                f"species '{species}'."
            )

        is_bamboo = (
            bool(row.get("IsBamboo", False))
            if pd.notna(row.get("IsBamboo", False))
            else False
        )

        culms = row.get("Culms", np.nan)

        culm_count = (
            int(culms)
            if is_bamboo and pd.notna(culms)
            else 1
        )

        if is_bamboo and culm_count < 1:
            raise ValueError(
                f"Invalid bamboo culm count for TreeID {tid}: {culms}"
            )

        comment = (
            ""
            if pd.isna(row.get("Any comment"))
            else str(row.get("Any comment"))
        )

        n_stems = (
            int(row["StemCount_N"])
            if pd.notna(row.get("StemCount_N"))
            else 0
        )

        if n_stems < 1:
            raise ValueError(
                f"TreeID {tid} has invalid StemCount_N={n_stems}."
            )

        tree_rows.append({
            "TreeID": tid,
            "Species": species,
            "Density_species_key": key,
            "Wood_density_mean_g_cm3": float(density_map[key]),
            "Is_bamboo_clump": is_bamboo,
            "Culms_count": culm_count,
            "Recorded_stem_count": n_stems,
            "Comment": comment,
        })

        # Every measured stem is handled separately.
        # Bamboo uses one representative culm, multiplied by culm count.
        for j in range(1, 6):

            cbh_col = f"CBH{j}(cm)"
            h_col = f"H{j}(m)"

            cbh = row.get(cbh_col, np.nan)
            height = row.get(h_col, np.nan)

            if pd.notna(cbh) or pd.notna(height):

                if pd.isna(cbh) or pd.isna(height):
                    raise ValueError(
                        f"TreeID {tid}, stem {j}: "
                        "CBH or height is missing."
                    )

                if float(cbh) <= 0 or float(height) <= 0:
                    raise ValueError(
                        f"TreeID {tid}, stem {j}: "
                        "non-positive CBH/height."
                    )

                stem_rows.append({
                    "TreeID": tid,
                    "Species": species,
                    "Density_species_key": key,
                    "Wood_density_mean_g_cm3": float(density_map[key]),
                    "Stem_number": j,
                    "CBH_cm": float(cbh),
                    "DBH_cm": float(cbh) / math.pi,
                    "Height_m": float(height),
                    "Is_bamboo_clump": is_bamboo,
                    "Culms_count": culm_count,
                    "Tree_comment": comment,
                })

        observed_stems = sum(
            pd.notna(row.get(f"CBH{j}(cm)", np.nan))
            and pd.notna(row.get(f"H{j}(m)", np.nan))
            for j in range(1, 6)
        )

        expected_stems = 1 if is_bamboo else n_stems

        if observed_stems != expected_stems:
            raise ValueError(
                f"TreeID {tid}: {observed_stems} complete "
                f"CBH/H stem records, but expected {expected_stems}. "
                "Check raw workbook."
            )

    stems = pd.DataFrame(stem_rows)
    trees = pd.DataFrame(tree_rows)

    if stems.empty:
        raise ValueError(
            "No valid stem measurements were found."
        )

    stems["Below_5cm_Chave_domain_flag"] = (
        stems["DBH_cm"] < CHAVE_MIN_DBH_CM_FOR_FLAG
    )

    stems["Data_note"] = ""

    stems.loc[
        stems["Below_5cm_Chave_domain_flag"],
        "Data_note"
    ] = (
        "DBH below 5 cm calibration threshold reported in lecture; "
        "retained as required, interpret cautiously"
    )

    stems.loc[
        stems["TreeID"].eq(543),
        "Data_note"
    ] = (
        stems.loc[
            stems["TreeID"].eq(543),
            "Data_note"
        ].astype(str)
        + "; only two measured stems included; "
        "third damaged stem was not measured"
    ).str.strip("; ")

    # Height and DBH must be numeric and positive.
    if (
        (stems["DBH_cm"] <= 0).any()
        or (stems["Height_m"] <= 0).any()
    ):
        raise ValueError(
            "Non-positive DBH or height detected."
        )

    return group, measured, excluded, trees, stems


# ============================== MONTE CARLO ENGINE ==============================

def simulate(stems, trees, n_sim, seed, enabled=None):
    """
    enabled controls one-at-a-time sensitivity runs.

    Wood density is sampled once per species per iteration and shared
    by all trees/stems of that species.
    """

    if enabled is None:
        enabled = {
            "dbh": True,
            "height": True,
            "density": True,
            "residual": True
        }

    rng = np.random.default_rng(seed)

    tree_ids = trees["TreeID"].astype(int).tolist()

    tree_index = {
        tid: i
        for i, tid in enumerate(tree_ids)
    }

    totals = np.zeros(
        (n_sim, len(tree_ids)),
        dtype=float
    )

    # Draw one density value per species per iteration.
    species_means = (
        trees[
            [
                "Density_species_key",
                "Wood_density_mean_g_cm3"
            ]
        ]
        .drop_duplicates()
        .set_index("Density_species_key")[
            "Wood_density_mean_g_cm3"
        ]
        .to_dict()
    )

    density_draws = {}

    if enabled["density"]:

        sigma = math.sqrt(
            math.log(1.0 + WOOD_DENSITY_CV**2)
        )

        for sp, mean_rho in species_means.items():

            # Preserve the supplied point value as arithmetic mean.
            mu = math.log(mean_rho) - 0.5 * sigma**2

            density_draws[sp] = rng.lognormal(
                mean=mu,
                sigma=sigma,
                size=n_sim
            )

    else:

        density_draws = {
            sp: np.full(n_sim, rho)
            for sp, rho in species_means.items()
        }

    for _, stem in stems.iterrows():

        tid = int(stem["TreeID"])
        sp = stem["Density_species_key"]

        dbh0 = float(stem["DBH_cm"])
        h0 = float(stem["Height_m"])

        dbh = (
            positive_normal(
                rng,
                dbh0,
                DBH_CV * dbh0,
                n_sim
            )
            if enabled["dbh"]
            else np.full(n_sim, dbh0)
        )

        height = (
            positive_normal(
                rng,
                h0,
                HEIGHT_CV * h0,
                n_sim
            )
            if enabled["height"]
            else np.full(n_sim, h0)
        )

        rho = density_draws[sp]

        epsilon = (
            rng.normal(
                0.0,
                CHAVE_RESIDUAL_LOG_SIGMA,
                n_sim
            )
            if enabled["residual"]
            else np.zeros(n_sim)
        )

        residual_multiplier = np.exp(epsilon)

        # Generic Chave equation:
        # rho in g/cm3, DBH in cm, H in m, AGB in kg.
        agb_one_stem = (
            0.0673
            * (rho * dbh**2 * height) ** 0.976
        )

        agb_one_stem = (
            agb_one_stem * residual_multiplier
        )

        # Bamboo: one representative culm multiplied by culm count.
        if bool(stem["Is_bamboo_clump"]):

            agb_one_stem = (
                agb_one_stem * int(stem["Culms_count"])
            )

        totals[:, tree_index[tid]] += agb_one_stem

    forest_agb = totals.sum(axis=1)

    tree_co2 = (
        totals
        * CARBON_FRACTION
        * CO2_PER_CARBON
    )

    forest_co2 = (
        forest_agb
        * CARBON_FRACTION
        * CO2_PER_CARBON
    )

    return {
        "tree_agb": totals,
        "forest_agb": forest_agb,
        "tree_co2": tree_co2,
        "forest_co2": forest_co2,
        "tree_ids": tree_ids,
    }


def summarize_array(x):

    return {
        "Mean": float(np.mean(x)),
        "Median": float(np.median(x)),
        "SD": float(np.std(x, ddof=1)),
        "P2_5": float(np.percentile(x, 2.5)),
        "P97_5": float(np.percentile(x, 97.5)),
    }


# ============================== SAVE RESULTS ==============================

def write_outputs(
    group,
    measured,
    excluded,
    trees,
    stems,
    sim,
    sensitivity
):

    ids = sim["tree_ids"]

    agb_rows = []

    for i, tid in enumerate(ids):

        vals = summarize_array(
            sim["tree_agb"][:, i]
        )

        co2_vals = summarize_array(
            sim["tree_co2"][:, i]
        )

        tr = trees.loc[
            trees["TreeID"].eq(tid)
        ].iloc[0]

        agb_rows.append({
            "TreeID": tid,
            "Species": tr["Species"],
            "Wood_density_point_g_cm3": (
                tr["Wood_density_mean_g_cm3"]
            ),
            "Culms_count": tr["Culms_count"],
            **{
                f"AGB_{k}_kg": v
                for k, v in vals.items()
            },
            **{
                f"CO2_{k}_kg": v
                for k, v in co2_vals.items()
            },
        })

    tree_summary = pd.DataFrame(agb_rows)

    tree_summary.to_csv(
        OUTPUT_DIR / "part3_tree_results.csv",
        index=False
    )

    forest_agb = summarize_array(
        sim["forest_agb"]
    )

    forest_co2 = summarize_array(
        sim["forest_co2"]
    )

    n_trees = len(ids)

    forest_rows = [
        {
            "Quantity": "Forest total above-ground biomass",
            "Unit": "kg dry biomass",
            **forest_agb
        },
        {
            "Quantity": "Forest total CO2 equivalent",
            "Unit": "kg CO2",
            **forest_co2
        },
    ]

    if PROPAGATE_TO_ANNUAL_RATE:

        # Assignment scaling:
        # sampled trees occupy n / 0.75 m2,
        # then convert to per hectare and divide by 5 years.
        occupied_area_m2 = (
            n_trees / TREE_DENSITY_PER_M2
        )

        annual_rate = (
            sim["forest_co2"]
            / occupied_area_m2
            * 10_000.0
            / STAND_AGE_YEARS
        )

        forest_rows.append({
            "Quantity": "Annual CO2 rate propagated from Part 3",
            "Unit": "kg CO2 ha-1 year-1",
            **summarize_array(annual_rate),
        })

    pd.DataFrame(forest_rows).to_csv(
        OUTPUT_DIR / "part3_forest_summary.csv",
        index=False
    )

    # Save every forest-level simulation draw.
    draw_df = pd.DataFrame({
        "Simulation": np.arange(
            1,
            len(sim["forest_agb"]) + 1
        ),
        "Forest_AGB_kg": sim["forest_agb"],
        "Forest_CO2_kg": sim["forest_co2"],
    })

    if PROPAGATE_TO_ANNUAL_RATE:
        draw_df["Annual_CO2_rate_kg_ha_year"] = annual_rate

    draw_df.to_csv(
        OUTPUT_DIR / "part3_forest_simulation_draws.csv",
        index=False
    )

    stems.to_csv(
        OUTPUT_DIR / "part3_cleaned_stem_data.csv",
        index=False
    )

    trees.to_csv(
        OUTPUT_DIR / "part3_tree_metadata.csv",
        index=False
    )

    excluded.to_csv(
        OUTPUT_DIR / "part3_excluded_records.csv",
        index=False
    )

    # Required plot: one mean point per TreeID with 95% interval whiskers.
    x = tree_summary["TreeID"].astype(str).tolist()

    means = tree_summary[
        "AGB_Mean_kg"
    ].to_numpy()

    lower = (
        means
        - tree_summary["AGB_P2_5_kg"].to_numpy()
    )

    upper = (
        tree_summary["AGB_P97_5_kg"].to_numpy()
        - means
    )

    fig, ax = plt.subplots(
        figsize=(11, 5.8)
    )

    ax.errorbar(
        np.arange(len(x)),
        means,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
        linewidth=1.2
    )

    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(
        x,
        rotation=45,
        ha="right"
    )

    ax.set_xlabel("Tree ID")
    ax.set_ylabel("Above-ground biomass (kg)")

    ax.set_title(
        f"Generic Chave AGB uncertainty by tree "
        f"({N_SIMULATIONS:,} Monte Carlo draws)"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "part3_mean_95_interval.png",
        dpi=220
    )

    plt.close(fig)

    # One-at-a-time source sensitivity.
    sens_rows = []

    for name, values in sensitivity.items():

        s = summarize_array(values)

        sens_rows.append({
            "Source_varied_alone": name,
            "Forest_AGB_SD_kg": s["SD"],
            "Forest_AGB_CV_percent": (
                100.0 * s["SD"] / s["Mean"]
                if s["Mean"]
                else np.nan
            ),
            "Forest_AGB_P2_5_kg": s["P2_5"],
            "Forest_AGB_P97_5_kg": s["P97_5"],
        })

    pd.DataFrame(sens_rows).sort_values(
        "Forest_AGB_SD_kg",
        ascending=False
    ).to_csv(
        OUTPUT_DIR / "part3_uncertainty_source_sensitivity.csv",
        index=False
    )

    # Tree interval width ranking.
    tree_summary["Interval_width_kg"] = (
        tree_summary["AGB_P97_5_kg"]
        - tree_summary["AGB_P2_5_kg"]
    )

    tree_summary[
        "Interval_width_as_percent_of_mean"
    ] = (
        100
        * tree_summary["Interval_width_kg"]
        / tree_summary["AGB_Mean_kg"]
    )

    tree_summary.sort_values(
        "Interval_width_kg",
        ascending=False
    ).to_csv(
        OUTPUT_DIR / "part3_tree_interval_width_ranking.csv",
        index=False
    )


# ============================== MAIN PROGRAM ==============================

def main():

    group, measured, excluded, trees, stems = (
        load_and_validate()
    )

    # Main simulation: all uncertainty sources enabled.
    main_sim = simulate(
        stems,
        trees,
        N_SIMULATIONS,
        RANDOM_SEED,
        enabled={
            "dbh": True,
            "height": True,
            "density": True,
            "residual": True
        }
    )

    # One-at-a-time sensitivity diagnostics.
    sensitivity = {}

    for label, enabled in [
        (
            "DBH only",
            {
                "dbh": True,
                "height": False,
                "density": False,
                "residual": False
            }
        ),
        (
            "Height only",
            {
                "dbh": False,
                "height": True,
                "density": False,
                "residual": False
            }
        ),
        (
            "Wood density only",
            {
                "dbh": False,
                "height": False,
                "density": True,
                "residual": False
            }
        ),
        (
            "Chave residual only",
            {
                "dbh": False,
                "height": False,
                "density": False,
                "residual": True
            }
        ),
    ]:

        s = simulate(
            stems,
            trees,
            N_SIMULATIONS,
            RANDOM_SEED + len(sensitivity) + 1,
            enabled
        )

        sensitivity[label] = s["forest_agb"]

    write_outputs(
        group,
        measured,
        excluded,
        trees,
        stems,
        main_sim,
        sensitivity
    )

    # Additional comparison for Tree 550.
    if (
        INCLUDE_POSSIBLY_DEAD_TREE_550
        and 550 in set(trees["TreeID"])
    ):

        alt_trees = trees.loc[
            trees["TreeID"].ne(550)
        ].copy()

        alt_stems = stems.loc[
            stems["TreeID"].ne(550)
        ].copy()

        alt = simulate(
            alt_stems,
            alt_trees,
            N_SIMULATIONS,
            RANDOM_SEED + 99,
            enabled={
                "dbh": True,
                "height": True,
                "density": True,
                "residual": True
            }
        )

        alt_summary = summarize_array(
            alt["forest_agb"]
        )

        main_summary = summarize_array(
            main_sim["forest_agb"]
        )

        pd.DataFrame([
            {
                "Scenario": (
                    "Primary: Tree 550 included (status uncertain)"
                ),
                **main_summary
            },
            {
                "Scenario": "Sensitivity: Tree 550 excluded",
                **alt_summary
            },
        ]).to_csv(
            OUTPUT_DIR / "part3_tree550_inclusion_sensitivity.csv",
            index=False
        )

    # Save run settings and methodology details.
    metadata = {
        "GroupID": GROUP_ID,
        "N_simulations": N_SIMULATIONS,
        "Random_seed": RANDOM_SEED,
        "Measured_tree_count_in_primary_run": len(trees),
        "Stem_measurement_count": len(stems),
        "Excluded_unmeasurable_count": (
            int(len(group) - len(measured))
        ),
        "Tree550_included": (
            INCLUDE_POSSIBLY_DEAD_TREE_550
        ),
        "DBH_CV": DBH_CV,
        "Height_CV": HEIGHT_CV,
        "Wood_density_lognormal_CV": (
            WOOD_DENSITY_CV
        ),
        "Chave_residual_log_sigma": (
            CHAVE_RESIDUAL_LOG_SIGMA
        ),
        "Carbon_fraction": CARBON_FRACTION,
        "Chave_bias_correction": (
            "None; lecture's approx. 5% overestimation tendency "
            "reported separately"
        ),
        "Equation": (
            "AGB = 0.0673 * (rho * DBH^2 * H)^0.976"
        ),
        "Wood_density_sampling": (
            "One lognormal draw per species per iteration, "
            "shared by all same-species trees"
        ),
        "Residual_sampling": (
            "Independent epsilon ~ Normal(0, 0.357^2) "
            "per measured stem per iteration; multiplier exp(epsilon)"
        ),
    }

    pd.Series(metadata).to_csv(
        OUTPUT_DIR / "part3_run_metadata.csv",
        header=["Value"]
    )

    print("Part 3 simulation completed.")

    print(
        f"Trees included: {len(trees)}; "
        f"stems/representative culms: {len(stems)}"
    )

    print(
        f"Unmeasurable records excluded: "
        f"{len(group) - len(measured)}"
    )

    print(
        f"Simulation draws: {N_SIMULATIONS:,}"
    )

    print(
        f"Outputs: {OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()