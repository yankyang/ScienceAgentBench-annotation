import os
import sys
import json
import csv
import random
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy
import cartopy.crs as ccrs


def apply_anti_contamination(world):
    random.seed(42)
    if len(world) > 5:
        drop_idx = random.sample(list(world.index), 5)
        world = world.drop(drop_idx)
    return world


def main():
    task_name = "cartopy_convert"
    out_dir = Path("./pred_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_file = out_dir / f"{task_name}_test.png"
    manifest_json = out_dir / f"{task_name}_manifest.json"
    manifest_csv = out_dir / f"{task_name}_manifest.csv"

    try:
        cartopy.config["data_dir"] = "./benchmark/datasets/cartopy_convert/"

        dataset_dir = Path("./benchmark/datasets/cartopy_convert")
        shp_file = dataset_dir / "naturalearth_lowres.shp"
        if not shp_file.exists():
            raise FileNotFoundError(f"Missing shapefile: {shp_file}")

        world = gpd.read_file(shp_file)
        world = apply_anti_contamination(world)

        fig, ax = plt.subplots(
            figsize=(10, 6),
            subplot_kw={"projection": ccrs.Robinson()},
        )
        ax.set_global()

        world.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            edgecolor="black",
            linewidth=0.5,
            facecolor="lightgray",
        )

        ax.set_title("World Map - Local Natural Earth (Robinson Projection)")
        plt.savefig(png_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        manifest = {
            "task": task_name,
            "input_shapefile": str(shp_file.resolve()),
            "output_png": str(png_file.resolve()),
            "removed_rows": 5 if len(world) > 5 else 0,
            "cartopy_data_dir": cartopy.config["data_dir"],
        }

        with open(manifest_json, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(manifest.keys())
            writer.writerow(manifest.values())

        print(f"[OK] Saved map → {png_file}")
        print(f"[OK] Manifest → {manifest_json} / {manifest_csv}")

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
