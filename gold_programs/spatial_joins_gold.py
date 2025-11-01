import sys
import traceback
import random
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point

def main():
    task_name = "spatial_joins"
    out_dir = Path("./pred_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{task_name}_result.csv"

    try:
        dataset_dir = Path("./benchmark/datasets/spatial_joins/")
        shp_file = dataset_dir / "nybb.shp"
        if not shp_file.exists():
            raise FileNotFoundError(f"Dataset not found: {shp_file}")

        polydf = gpd.read_file(shp_file)

        b = [int(x) for x in polydf.total_bounds]
        N = 8
        pointdf = gpd.GeoDataFrame(
            [
                {"geometry": Point(x, y), "value1": x + y, "value2": x - y}
                for x, y in zip(
                    range(b[0], b[2], int((b[2] - b[0]) / N)),
                    range(b[1], b[3], int((b[3] - b[1]) / N)),
                )
            ],
            crs=polydf.crs,
        )

        try:
            join_df = gpd.sjoin(pointdf, polydf, how="left", predicate="within")
        except TypeError:
            join_df = gpd.sjoin(pointdf, polydf, how="left", op="within")

        removed_examples = join_df.sample(n=min(5, len(join_df)), random_state=42)
        remaining_df = join_df.drop(removed_examples.index)

        anti_file = Path("./benchmark/eval_programs/gold_results/") / f"{task_name}_removed.csv"
        anti_file.parent.mkdir(parents=True, exist_ok=True)
        removed_examples.to_csv(anti_file, index=False)

        remaining_df.to_csv(out_file, index=False)
        print(f"Saved result to {out_file.resolve()}")

    except Exception as e:
        error_file = out_dir / f"{task_name}_error.log"
        with open(error_file, "w") as f:
            traceback.print_exc(file=f)
        print(f"Error occurred: {e}. Traceback saved to {error_file.resolve()}")

if __name__ == "__main__":
    main()
