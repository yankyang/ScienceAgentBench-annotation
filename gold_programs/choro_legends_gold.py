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
    if len(world) > 5:
        drop_idx = random.sample(list(world.index), 5)
        world = world.drop(drop_idx)
    return world


def main():
    task_name = "choro_legends"
    out_dir = Path("./pred_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_file = out_dir / f"{task_name}_test.png"
    manifest_json = out_dir / f"{task_name}_manifest.json"
    manifest_csv = out_dir / f"{task_name}_manifest.csv"
    records_csv = out_dir / f"{task_name}_records.csv"

    try:
        dataset_dir = Path("./benchmark/datasets/choro_legends")
        shp_file = dataset_dir / "naturalearth_lowres.shp"
        if not shp_file.exists():
            raise FileNotFoundError(
                f"Dataset not found at {shp_file}. "
                "Please ensure naturalearth_lowres.* files are placed correctly."
            )

        world = gpd.read_file(shp_file)
        world = apply_anti_contamination(world)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        world.plot(
            column="pop_est",
            cmap="OrRd",
            linewidth=0.8,
            ax=ax,
            edgecolor="0.8",
            legend=True,
            legend_kwds={"label": "Population Estimate", "orientation": "horizontal"},
        )
        ax.set_title("Choropleth Map of Population Estimate", fontsize=14)
        plt.axis("off")

        plt.savefig(png_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        cols_to_save = ["name", "pop_est"]
        if all(col in world.columns for col in cols_to_save):
            world[cols_to_save].to_csv(records_csv, index=False)
        else:
            with open(records_csv, "w") as f:
                f.write("name,pop_est\n[missing columns]\n")

        manifest = {
            "task": task_name,
            "input_shapefile": str(shp_file.resolve()),
            "output_png": str(png_file.resolve()),
            "removed_rows": 5 if len(world) > 5 else 0,
            "record_csv": str(records_csv.resolve()),
        }
        with open(manifest_json, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(manifest.keys())
            writer.writerow(manifest.values())

        print(f"[OK] Visualization saved to {png_file}")
        print(f"[OK] Manifest → {manifest_json} / {manifest_csv}")
        print(f"[OK] Records CSV → {records_csv}")

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
