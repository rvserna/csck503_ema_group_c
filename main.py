import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import matplotlib.pyplot as plt
from pathlib import Path

import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

RAW_DATA_PATH = Path(r'./raw_data/')

class BreakOutError(Exception):
    pass

class GridEmissions:
    def __init__(self):
        grid_shape_path = RAW_DATA_PATH / "supporting_data/grid/LAEI2019_Grid.shp"
        self.df = gpd.read_file(grid_shape_path, columns=["Grid_ID_19"])
        self.df["grid_id"] = self.df["Grid_ID_19"].astype(int)
        self.df.drop(columns="Grid_ID_19", inplace=True)
    
    def _close_multiline(self, multiline, tolerance):
        multiline_list = list(multiline.geoms)
        start_again = True
        while start_again:
            try:
                start_again = False
                length = len(multiline_list)
                if length == 1:
                    return multiline_list[0]
                for i in range(0, length):
                    for j in range(i+1, length):
                        line1 = multiline_list[i]
                        line2 = multiline_list[j]

                        start1, end1 = line1.coords[0], line1.coords[-1]
                        start2, end2 = line2.coords[0], line2.coords[-1]

                        best_fit = None
                        line_length = tolerance
                        potentials = [
                                        shapely.geometry.LineString([end1, start2]),
                                        shapely.geometry.LineString([end1, end2]),
                                        shapely.geometry.LineString([start1, start2]),
                                        shapely.geometry.LineString([start1, end2]),
                                    ]
                        
                        for potential in potentials:
                            if (potential.length < line_length):
                                best_fit = potential
                                line_length = potential.length
                        if best_fit is not None:
                            del multiline_list[i]
                            del multiline_list[j-1]
                            multiline_list.append(shapely.ops.linemerge((line1, line2, best_fit)))
                            start_again = True
                            raise BreakOutError
            except BreakOutError:
                pass
        return multiline

    def _process_links_groupby(self, gdf, retain_fields=[], non_contiguity_tolerance=200):

        # Retain the original gdf's CRS
        crs = gdf.crs

        # Explode multi-line geometries into individual entries
        exploded_gdf = gdf.explode(index_parts=True)

        # Merge the lines using shapely's linemerge
        merged_geom = shapely.ops.linemerge(exploded_gdf.geometry.tolist())

        if isinstance(merged_geom, shapely.geometry.MultiLineString):
            logger.warning("Found non-contiguity in LinkID %s.", gdf.name)
            if non_contiguity_tolerance:
                merged_geom = self._close_multiline(merged_geom, non_contiguity_tolerance)
                if not isinstance(merged_geom, shapely.geometry.MultiLineString):
                    logging.info("Fixed contiguity.")
                else:
                    logging.warning("Failed to fix contiguity.")
        # Create a new GeoDataFrame from the merged geometry, retaining the CRS
        result = gpd.GeoDataFrame({"geometry": [merged_geom]}, crs=crs)

        # Append any fields that are required from the original gdf
        for field in retain_fields:
            result[field] = gdf[field].iloc[0]

        # Return the finalised gdf for this group.
        return result

    def _find_total_intersecting_length_by_grid(self, df2, new_column):
        df2 = df2.to_crs(self.df.crs)
        overlap = gpd.overlay(df1=self.df, df2=df2, how="intersection", keep_geom_type=False)
        overlap[new_column] = overlap.geometry.length

        result = overlap.groupby('grid_id')[new_column].sum().reset_index()
        result['grid_id'] = result['grid_id'].astype(object)

        self.df = self.df.merge(result, on='grid_id', how='left')
        self.df[new_column] = self.df[new_column].fillna(0.0).astype(float).apply(lambda x: round(x, 6))
    
    def _find_total_intersecting_agg_by_grid(self, df2, agg):
        df2 = df2.to_crs(self.df.crs)
        overlap = gpd.overlay(df1=self.df, df2=df2, how="intersection", keep_geom_type=False)
        aggregated_data = overlap.groupby("grid_id").agg(**agg).reset_index()

        self.df = self.df.merge(aggregated_data, how='left', on='grid_id')
        for column in agg.keys():
            self.df[column] = self.df[column].fillna(0.0).astype(float).apply(lambda x: round(x, 6))
    
    def _add_link_data(self, link_type, rel_path, link_id_col, retain_fields=[]):
        file_path = RAW_DATA_PATH / rel_path
        gdf = gpd.read_file(file_path, columns=None)
        print(gdf.iloc[0])
        print(gdf.iloc[0].geometry.length)

        # Preprocess gdf
        # gdf contains multiple entries for same link so need to merge rows by link_id_col
        # resulting in a new df containing one entry per link (with the resulting lines merged via shapely).
        compiled_gdf = gdf.groupby(link_id_col).apply(self._process_links_groupby, include_groups=False, retain_fields=retain_fields).reset_index(drop=True)

        # Add total length of links per grid
        self._find_total_intersecting_length_by_grid(compiled_gdf, f"total_{link_type}_length")

        # Now find a list of stations (start and/or end points on the each link)
        # Extract start and end points
        start_points = compiled_gdf.geometry.apply(lambda x: shapely.geometry.Point(x.coords[0]))  # Start point
        end_points = compiled_gdf.geometry.apply(lambda x: shapely.geometry.Point(x.coords[-1])) # End point

        # Combine start and end points into a single series
        all_points = pd.concat([start_points, end_points])
        all_points_counts = all_points.value_counts()

        all_points_gdf = gpd.GeoDataFrame(geometry=all_points)
        all_points_gdf[f"total_{link_type}_link_terminations"] = all_points_gdf.geometry.map(all_points_counts)

        all_points_gdf = all_points_gdf.drop_duplicates(subset="geometry").reset_index(drop=True)

        agg_by_link = {f"total_{link_type}_stations": ("geometry", "size"),
                       f"total_{link_type}_link_terminations": (f"total_{link_type}_link_terminations", "sum")}

        self._find_total_intersecting_agg_by_grid(all_points_gdf, agg_by_link)
        # Add count of "stations" (either end of the line) to the main df per grid

    def add_emissions_data(self):
        emissions_summary_path = RAW_DATA_PATH / "emissions_summary"
        in_scope_dfs = [self.df]
        for path in emissions_summary_path.rglob("*-all-sources.shp"):
            logger.debug(f"Loading file %s", path)
            # TODO: Consider using Excel instead as geo-overhead is uneeded (already in grid).
            emissions_df = gpd.read_file(path,
                                        columns=("pollutant", "all_2019"))
            # TODO: Probably more sensible to regex this from path
            pollutant = emissions_df.iloc[0]["pollutant"]
            emissions_df.rename(columns={"all_2019": f"total_pollutant_{pollutant}_2019"},
                                inplace=True)

            in_scope_dfs.append(emissions_df.drop(columns=["pollutant", "geometry"]))
        self.df = pd.concat(in_scope_dfs, axis=1)

    def add_rail_data(self):
        self._add_link_data("rail", "supporting_data/rail/LAEI_Rail_Network_Link_ExactCut.shp", "LinkID", retain_fields=["DESCRIPT"])

    def add_shipping_data(self):
        self._add_link_data("shipping", "supporting_data/shipping/LAEI_PassengerShipping_Network_Link.shp", "LINK_ID")
    
    def add_road_data(self):
        shape_file_path = RAW_DATA_PATH / "supporting_data/road/shape/laei-2019-major-roads-final-unique-toids-flows-speeds-osgb.shp"
        gdf = gpd.read_file(shape_file_path, columns=["TOID"])

        # Excel contains more data, but we need spatial data to geographically merge into our 1km grid...
        # We can merge them by index.
        excel_file_path = RAW_DATA_PATH / "supporting_data/road/excel/laei-2019-major-roads-vkm-flows-speeds.xlsx"
        df = pd.read_excel(excel_file_path)

        gdf = gdf.merge(df, left_index=True, right_index=True)

        print(gdf[gdf["TOID_x"] != gdf["TOID_y"]])

        print(gdf.iloc[121])
        print(gdf.iloc[121].geometry.length)
        self._find_total_intersecting_length_by_grid(gdf, "total_road_length")

        print([col for col in gdf.columns.values if col.startswith("F_")])
        agg_by_link = {f"total_road_count_by_toid": ("geometry", "size"),
                       f"average_road_speed_kph": (f"Sp_kph", "mean")}
        self._find_total_intersecting_agg_by_grid(gdf, agg_by_link)

if __name__ == "__main__":
    grid = GridEmissions()
    #grid.add_emissions_data()
    #grid.add_rail_data()
    #grid.add_shipping_data()
    grid.add_road_data()
    #print(grid.df[(grid.df["total_rail_stations"] != 0.0) & (grid.df["total_shipping_stations"] != 0.0)])
    print(grid.df[(grid.df["total_road_length"] != 0.0)])