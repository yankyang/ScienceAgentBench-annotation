import os
import sys
import csv
import json
import random
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd


def apply_anti_contamination(world):

    random.seed(42)
    removed_idx = []
    if len(world) > 5:
        removed_idx = random.sample(list(world.index), 5)
        world = world.drop(removed_idx)
    kept_idx = list(world.index)
    return world, kept_idx, removed_idx


def main():
    task_name = "choropleths"
    out_dir = Path("./pred_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_file = out_dir / f"{task_name}_test.png"
    manifest_json = out_dir / f"{task_name}_manifest.json"
    manifest_csv = out_dir / f"{task_name}_manifest.csv"
    choro_manifest = out_dir / f"{task_name}_choro_manifest.csv"

    try:
        dataset_dir = Path("./benchmark/datasets/choropleths")
        shp_file = dataset_dir / "naturalearth_lowres.shp"
        if not shp_file.exists():
            raise FileNotFoundError(
                f"Dataset not found at {shp_file}. "
                "Please ensure naturalearth_lowres.* files are placed correctly."
            )

        world = gpd.read_file(shp_file)

        world, kept_idx, removed_idx = apply_anti_contamination(world)

        world["gdp_per_cap"] = world["gdp_md_est"] / world["pop_est"]
        world["gdp_per_cap"] = world["gdp_per_cap"].fillna(0)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        world.plot(
            column="gdp_per_cap",
            cmap="viridis",
            linewidth=0.8,
            ax=ax,
            edgecolor="0.8",
            legend=True,
            legend_kwds={
                "label": "GDP per Capita (estimated)",
                "orientation": "horizontal",
            },
        )
        ax.set_title("Choropleth Map of GDP per Capita", fontsize=14)
        plt.axis("off")
        plt.savefig(png_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        cols_used = ["iso_a3", "name", "pop_est", "gdp_md_est", "gdp_per_cap"]
        choro_data = []

        for idx, row in world.iterrows():
            entry = {
                "iso_a3": row.get("iso_a3", ""),
                "name": row.get("name", ""),
                "pop_est": row.get("pop_est", None),
                "gdp_md_est": row.get("gdp_md_est", None),
                "gdp_per_cap": row.get("gdp_per_cap", None),
                "kept": True,
            }
            choro_data.append(entry)
        gpd_all = gpd.read_file(shp_file)
        for idx in removed_idx:
            row = gpd_all.loc[idx]
            entry = {
                "iso_a3": row.get("iso_a3", ""),
                "name": row.get("name", ""),
                "pop_est": row.get("pop_est", None),
                "gdp_md_est": row.get("gdp_md_est", None),
                "gdp_per_cap": (
                    row.get("gdp_md_est", 0) / row.get("pop_est", 1)
                    if row.get("pop_est", 0) != 0
                    else 0
                ),
                "kept": False,
            }
            choro_data.append(entry)

        with open(choro_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols_used + ["kept"])
            writer.writeheader()
            writer.writerows(choro_data)

        manifest = {
            "task": task_name,
            "input_shapefile": str(shp_file.resolve()),
            "output_png": str(png_file.resolve()),
            "num_total": len(gpd.read_file(shp_file)),
            "num_kept": len(kept_idx),
            "num_removed": len(removed_idx),
            "removed_indices": removed_idx,
            "choropleths_manifest": str(choro_manifest.resolve()),
        }
        with open(manifest_json, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(manifest.keys())
            writer.writerow(manifest.values())

        print(f"[OK] Visualization saved to {png_file}")
        print(f"[OK] Choropleths manifest → {choro_manifest}")
        print(f"[OK] Summary manifest → {manifest_json} / {manifest_csv}")

    except Exception as e:
        err_file = out_dir / f"{task_name}_error.json"
        with open(err_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "traceback": traceback.format_exc(limit=200),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
