import os
import sys
import json
import traceback
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def apply_anti_contamination(world):

    if len(world) <= 5:
        return world, []

    world_sorted = world.sort_values("iso_a3", ascending=True)
    removed_iso = list(world_sorted["iso_a3"].head(5))
    world_kept = world[~world["iso_a3"].isin(removed_iso)]
    return world_kept, removed_iso


def main():
    task_name = "inset_map"
    out_dir = Path("./pred_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{task_name}_test.png"
    removed_file = out_dir / f"{task_name}_removed_ids.txt"
    manifest_file = out_dir / f"{task_name}_manifest.json"

    try:
        dataset_dir = Path("./benchmark/datasets/inset_map")
        shp_file = dataset_dir / "naturalearth_lowres.shp"
        if not shp_file.exists():
            raise FileNotFoundError(
                f"Dataset not found at {shp_file}. "
                "Please place naturalearth_lowres.* in benchmark/datasets/inset_map/"
            )

        world = gpd.read_file(shp_file)

        world_filtered, removed_iso = apply_anti_contamination(world)

        fig, ax = plt.subplots(figsize=(10, 6))
        world_filtered.boundary.plot(ax=ax, linewidth=0.5, edgecolor="black")
        world_filtered.plot(ax=ax, facecolor="lightgrey")

        ax.set_title("World map with inset region", fontsize=12)
        ax.set_axis_off()

        inset_region_name = "China"
        inset_region = world_filtered[world_filtered["name"] == inset_region_name]
        if inset_region.empty:
            raise ValueError(f"Inset region not found: {inset_region_name}")

        axins = inset_axes(ax, width="30%", height="30%", loc="lower left", borderpad=2)
        world_filtered.boundary.plot(ax=axins, linewidth=0.5, edgecolor="black")
        world_filtered.plot(ax=axins, facecolor="lightgrey")
        inset_region.plot(ax=axins, facecolor="green", edgecolor="black")

        axins.set_xlim(inset_region.total_bounds[0] - 5, inset_region.total_bounds[2] + 5)
        axins.set_ylim(inset_region.total_bounds[1] - 5, inset_region.total_bounds[3] + 5)
        axins.set_title(f"Inset: {inset_region_name}", fontsize=8)
        axins.set_axis_off()

        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        with open(removed_file, "w", encoding="utf-8") as f:
            for rid in removed_iso:
                f.write(rid + "\n")

        plotted_countries = list(world_filtered["name"].dropna().unique())
        manifest = {
            "task": task_name,
            "input_shapefile": str(shp_file.resolve()),
            "output_png": str(out_file.resolve()),
            "inset_region": inset_region_name,
            "removed_iso_a3": removed_iso,
            "num_removed": len(removed_iso),
            "num_kept": len(world_filtered),
            "plotted_countries": plotted_countries[:10],  
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"[OK] Inset map visualization saved to {out_file}")
        print(f"[OK] Removed IDs → {removed_file}")
        print(f"[OK] Manifest → {manifest_file}")

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
