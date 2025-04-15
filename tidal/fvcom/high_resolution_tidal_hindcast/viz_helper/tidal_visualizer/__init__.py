# __init__ module - Part of tidal_visualizer package

# Import core components
from .core import TidalVisualizerBase
from .visualization import TidalVisualizer
from .hotspots import find_tidal_energy_hotspots
from .config import (
    get_location_names,
    get_areas_of_interest,
    get_location_info,
    get_area_of_interest_info,
    add_hotspots_to_config,
)
from .io import (
    find_dataset_file,
    load_dataset,
    get_variable_data,
    export_hotspots_to_csv,
    import_hotspots_from_csv,
)
from .utils import (
    haversine_distance,
    format_location_display,
    format_area_of_interest_display,
    calculate_bounds_center,
    calculate_power_statistics,
)

# Import new functionality
from .time_series import create_time_series_visualization, plot_time_series_at_point
from .elevation_profiles import (
    create_elevation_profile,
    add_profile_line_to_map,
    create_cross_section_with_flow,
)
from .context_layers import ContextLayerManager, AVAILABLE_BASEMAPS
from .coordinates import CoordinateManager
from .vector_visualization import (
    add_flow_vectors,
    add_scaled_flow_vectors,
    add_streamlines,
    create_composite_visualization,
    create_particle_animation,
)
