"""
Unified variable registry for all tidal hindcast quantities of interest.
"""

VARIABLE_REGISTRY = {
    # =========================================================================
    # Water column speed variables
    # =========================================================================
    "mean_current_speed": {
        "display_name": "Mean Current Speed",
        "column_name": "vap_water_column_mean_sea_water_speed",
        "units": "m/s",
        "included_on_atlas": True,
        # Documentation (from generate_atlas_variable_docs.py)
        "one_liner": "Annual average of depth-averaged current speed",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#mean-current-speed",
        "complete_description": (
            "Mean Current Speed is the annual average of the depth-averaged current velocity "
            "magnitude, representing the characteristic flow speed at each grid location under "
            "free-stream (undisturbed) conditions. This metric is intended for IEC 62600-201 "
            "Stage 1 reconnaissance-level analysis to identify areas with tidal current resources. "
            "Calculated as the temporal mean of depth-averaged |U| where U = \u221a(u\u00b2 + v\u00b2).\n\n"
            "Mean Current Speed is computed by first averaging velocity magnitudes across all "
            "10 sigma layers at each timestep, then averaging over the full hindcast year. "
            "Current speed is the vector magnitude of eastward (u) and northward (v) "
            "velocity components: U = \u221a(u\u00b2 + v\u00b2). Tidal currents flow slower near the seafloor due "
            "to friction and faster in the upper water column. Depth-averaging provides a "
            "representative value for the entire water column.\n\n"
            "Engineering applications include initial site screening, comparing relative site "
            "potential across regions, and Stage 1 IEC 62600-201 tidal energy resource "
            "characterization."
        ),
        "references": ["iec_62600_201", "fvcom"],
        # Scientific/engineering context
        "physical_meaning": "Yearly average of depth averaged current speed",
        "intended_usage": "Site screening and turbine selection for power generation",
        "intended_usage_detail": (
            "Primary metric for identifying viable tidal energy sites. Used to estimate "
            "annual energy production (AEP), compare site potential across regions, determine "
            "minimum viable current speeds for commercial deployment (typically >1.5 m/s), "
            "and select appropriate turbine technology. Critical for feasibility studies and "
            "initial resource assessments."
        ),
        "equation": r"$\overline{\overline{U}} = U_{\text{average}} = \text{mean}\left(\left[\text{mean}(U_{1,t}, ..., U_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$",
        "equation_variables": [
            "U_{i,t} = \u221a(u\u00b2 + v\u00b2) \u2014 velocity magnitude at sigma layer i at time t (m/s)",
            "N_\u03c3 = 10 sigma layers (terrain-following vertical layers dividing the water column "
            "into equal-thickness fractions from surface to seafloor)",
            "T = 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)",
        ],
    },
    "p95_current_speed": {
        "display_name": "95th Percentile Current Speed",
        "column_name": "vap_water_column_95th_percentile_sea_water_speed",
        "units": "m/s",
        "included_on_atlas": True,
        # Documentation
        "one_liner": "95th percentile of yearly maximum current speed (all depths)",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#95th-percentile-current-speed",
        "complete_description": (
            "95th Percentile Current Speed is the current speed exceeded only 5% of the time, "
            "computed from the depth-maximum (highest value across sigma layers) at each timestep. "
            "This metric characterizes extreme flow conditions relevant to structural loading and "
            "device survivability. Used for preliminary assessment of extreme current conditions in "
            "support of Stage 2 feasibility studies per IEC 62600-201. "
            "Calculated as P\u2089\u2085(max_\u03c3(U)) where U = \u221a(u\u00b2 + v\u00b2).\n\n"
            "At each timestep, the maximum speed across all 10 sigma layers is identified "
            "(the depth-maximum), then the 95th percentile of this time series is computed. "
            "The depth-maximum is used (rather than depth-average) because structural and "
            "mechanical systems must withstand peak loads that can occur at any depth in the "
            "water column. This metric characterizes expected high-flow conditions for "
            "survivability design, while excluding rare extreme events.\n\n"
            "Engineering applications include structural loading calculations, blade and "
            "support structure design, and fatigue analysis. This metric helps size "
            "components to withstand expected high-flow conditions."
        ),
        "references": ["iec_62600_201", "fvcom"],
        # Scientific/engineering context
        "physical_meaning": "95th percentile of yearly depth maximum current speed",
        "intended_usage": "Generator sizing and power electronics design",
        "intended_usage_detail": (
            "Critical for sizing electrical generation components. Used to determine "
            "maximum generator output capacity, size power electronics and converters for "
            "peak electrical loads, design control systems for extreme speed conditions, "
            "and set cut-out speeds for generator protection. Essential for electrical "
            "system certification, grid connection requirements, and ensuring generators "
            "can handle maximum rotational speeds without damage."
        ),
        "equation": r"$U_{95} = \text{percentile}(95, \left[\max(U_{1,t}, ..., U_{N_{\sigma},t}) \text{ for } t=1,...,T\right])$",
        "equation_variables": [
            "U_{i,t} = \u221a(u\u00b2 + v\u00b2) \u2014 velocity magnitude at sigma layer i at time t (m/s)",
            "max_\u03c3 \u2014 maximum value across all 10 sigma layers at each timestep",
            "P\u2089\u2085 \u2014 95th percentile operator over the full time series",
            "N_\u03c3 = 10 sigma layers",
            "T = 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)",
        ],
    },
    "p99_current_speed": {
        "display_name": "99th Percentile Current Speed",
        "column_name": "vap_water_column_99th_percentile_sea_water_speed",
        "units": "m/s",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "99th percentile of yearly depth maximum current speed",
        "intended_usage": "Extreme event analysis and safety system design",
        "intended_usage_detail": (
            "Used for extreme event planning and safety margin calculations. Critical for "
            "designing emergency shutdown systems, setting absolute operational limits, and "
            "ensuring equipment can survive rare but intense current events. Important for "
            "insurance risk assessments, environmental impact studies, and regulatory "
            "compliance for extreme conditions."
        ),
        "equation": r"$U_{99} = \text{percentile}(99, \left[\max(U_{1,t}, ..., U_{N_{\sigma},t}) \text{ for } t=1,...,T\right])$",
        "equation_variables": [
            r"$U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$ are velocity magnitudes at uniformly distributed sigma level $i$ at volume centers at time $t$ (m/s)",
            r"$N_{\sigma} = 10$ levels",
            r"$T = 1$ year",
        ],
    },
    "max_current_speed": {
        "display_name": "Maximum Current Speed",
        "column_name": "vap_water_column_max_sea_water_speed",
        "units": "m/s",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "Absolute maximum depth-max current speed observed over the year",
        "intended_usage": "Ultimate load design and survival analysis",
        "intended_usage_detail": (
            "Defines the absolute worst-case current speed for ultimate load calculations. "
            "Essential for structural survival analysis, determining maximum possible loads "
            "on turbines and support structures, and designing fail-safe mechanisms. Used "
            "for regulatory compliance demonstrating equipment can survive maximum observed "
            "conditions."
        ),
        "equation": r"$U_{\max} = \max\left(\left[\max(U_{1,t}, ..., U_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$",
        "equation_variables": [
            r"$U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$ are velocity magnitudes at uniformly distributed sigma level $i$ at volume centers at time $t$ (m/s)",
            r"$N_{\sigma} = 10$ levels",
            r"$T = 1$ year",
        ],
    },
    # =========================================================================
    # Power density variables
    # =========================================================================
    "mean_power_density": {
        "display_name": "Mean Power Density",
        "column_name": "vap_water_column_mean_sea_water_power_density",
        "units": "W/m\u00b2",
        "included_on_atlas": True,
        # Documentation
        "one_liner": "Annual average of depth-averaged kinetic energy flux",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#mean-power-density",
        "complete_description": (
            "Mean Power Density is the annual average of the kinetic energy flux per unit area, "
            "representing the theoretical power available for extraction from the undisturbed "
            "tidal flow. The cubic relationship with velocity (P = \u00bd\u03c1U\u00b3) makes this metric highly "
            "sensitive to current speed variations. Used for Stage 1 resource characterization "
            "and site ranking to indicate theoretical resource magnitude.\n\n"
            "Power density is computed at each sigma layer using the cube of the current speed, "
            "then depth-averaged and temporally averaged over the full hindcast year. The cubic "
            "relationship with velocity means small increases in current speed yield large "
            "increases in available power\u2014doubling the speed increases power density by a "
            "factor of eight.\n\n"
            "Engineering applications include comparing relative energy availability between "
            "sites and initial economic feasibility screening."
        ),
        "references": ["iec_62600_201", "fvcom"],
        # Scientific/engineering context
        "physical_meaning": "Yearly average of depth averaged power density (kinetic energy flux)",
        "intended_usage": "Resource quantification and economic feasibility analysis",
        "intended_usage_detail": (
            "Direct measure of extractable energy resource for economic analysis. Used to "
            "calculate theoretical power output, estimate capacity factors for project "
            "financing, compare energy density between sites, and determine optimal turbine "
            "spacing in arrays. Essential for LCOE calculations, investor presentations, "
            "and grid integration planning. Minimum thresholds (typically >300 W/m\u00b2) define "
            "commercial viability."
        ),
        "equation": r"$\overline{\overline{P}} = P_{\text{average}} = \text{mean}\left(\left[\text{mean}(P_{1,t}, ..., P_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$",
        "equation_variables": [
            "P_{i,t} = \u00bd\u03c1U\u00b3_{i,t} \u2014 power density at sigma layer i at time t (W/m\u00b2)",
            "\u03c1 = 1025 kg/m\u00b3 (nominal seawater density; actual varies with temperature and salinity)",
            "U_{i,t} = \u221a(u\u00b2 + v\u00b2) \u2014 velocity magnitude (m/s)",
            "N_\u03c3 = 10 sigma layers",
            "T = 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)",
        ],
    },
    "p95_power_density": {
        "display_name": "95th Percentile Power Density",
        "column_name": "vap_water_column_95th_percentile_sea_water_power_density",
        "units": "W/m\u00b2",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "95th percentile of the yearly maximum of depth averaged power density (kinetic energy flux)",
        "intended_usage": "Structural design loads and extreme loading conditions",
        "intended_usage_detail": (
            "Essential for structural engineering and extreme load analysis. Used to "
            "determine maximum design loads for turbine blades, drive trains, support "
            "structures, and foundation systems. Critical for fatigue analysis, ultimate "
            "load calculations, and ensuring structural integrity during extreme tidal "
            "events. Defines design margins for mooring systems, tower structures, and "
            "emergency braking systems. Required for structural certification and insurance "
            "assessments."
        ),
        "equation": r"$P_{95} = \text{percentile}(95, \left[\max(P_{1,t}, ..., P_{N_{\sigma},t}) \text{ for } t=1,...,T\right])$",
        "equation_variables": [
            r"$P_{i,t} = \frac{1}{2} \rho U_{i,t}^3$ with $\rho = 1025$ kg/m³",
            r"$U_{i,t}$ are velocity magnitudes at uniformly distributed sigma level $i$ at volume centers at time $t$",
            r"$N_{\sigma} = 10$ levels",
            r"$T = 1$ year",
        ],
    },
    # =========================================================================
    # Depth variables
    # =========================================================================
    "mean_water_depth": {
        "display_name": "Mean Depth",
        "column_name": "vap_sea_floor_depth",
        "units": "m (below NAVD88)",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "Yearly average distance from water surface to the sea floor",
        "intended_usage": "Installation planning and foundation design",
        "intended_usage_detail": (
            "Fundamental constraint for deployment strategy and cost estimation. Used to "
            "determine installation vessel requirements, foundation type selection (gravity, "
            "pile, suction caisson), and deployment method feasibility. Critical for cost "
            "modeling (deeper = more expensive), accessibility planning for maintenance "
            "operations, and environmental impact assessments. Optimal depths typically "
            "20-50m for current technology, with deeper sites requiring specialized "
            "equipment and higher costs."
        ),
        "equation": r"$\overline{d} = d_{\text{average}} = \text{mean}\left(\left[(h + \zeta_t) \text{ for } t=1,...,T\right]\right)$",
        "equation_variables": [
            r"$h$ is bathymetry below NAVD88 (m)",
            r"$\zeta_t$ is sea surface elevation above NAVD88 at time $t$ (m)",
            r"$T = 1$ year",
        ],
    },
    "min_water_depth": {
        "display_name": "Minimum Water Depth",
        "column_name": "vap_water_column_height_min",
        "units": "m",
        "included_on_atlas": True,
        # Documentation
        "one_liner": "Minimum water depth (during 1 year model run)",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#minimum-water-depth",
        "complete_description": (
            "Minimum Water Depth is the lowest water depth (surface to seafloor) "
            "observed at each grid location over the hindcast year, typically occurring during "
            "extreme low tide conditions. This metric defines the minimum vertical clearance "
            "available for device deployment and is critical for assessing depth constraints. "
            "Used in Stage 2 feasibility studies for turbine placement and collision avoidance. "
            "Calculated as d_min = min_t(h + \u03b6_t) where h is bathymetry and \u03b6 is sea surface elevation.\n\n"
            "Minimum water depth typically occurs during spring tides when tidal range is maximized. "
            "Total water depth is calculated as the sum of bathymetry depth (h) and sea surface "
            "elevation (\u03b6). Values are adjusted from NAVD88 to MSL reference.\n\n"
            "The difference between maximum and minimum water depth approximates the "
            "tidal range at each location.\n\n"
            "Engineering applications include assessing turbine clearance requirements and "
            "identifying areas where shallow water may limit device deployment."
        ),
        "references": ["iec_62600_201", "fvcom"],
        # Scientific/engineering context
        "physical_meaning": "Minimum water depth observed over the year (shallowest, typically at low tide)",
        "intended_usage": "Minimum clearance and grounding risk assessment",
        "intended_usage_detail": (
            "Critical for determining minimum water depth available for turbine deployment. "
            "Used to ensure adequate clearance between turbine blades and seafloor, assess "
            "grounding risk during extreme low tides, and determine deployment feasibility "
            "in shallow areas. Essential for safety planning and operational constraints."
        ),
        "equation": r"$d_{\min} = \min\left(\left[(h + \zeta_t) \text{ for } t=1,...,T\right]\right)$",
        "equation_variables": [
            "h \u2014 bathymetry depth (m)",
            "\u03b6_t \u2014 sea surface elevation at time t (m)",
            "T = 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)",
        ],
    },
    "max_water_depth": {
        "display_name": "Maximum Water Depth",
        "column_name": "vap_water_column_height_max",
        "units": "m",
        "included_on_atlas": True,
        # Documentation
        "one_liner": "Maximum water depth (during 1 year model run)",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#maximum-water-depth",
        "complete_description": (
            "Maximum Water Depth is the greatest water depth (surface to seafloor) "
            "observed at each grid location over the hindcast year, typically occurring during "
            "extreme high tide conditions. This metric represents the upper bound of water depth "
            "variability at each location. Used in Stage 2 feasibility studies for mooring system "
            "design, cable routing, and understanding the full operating depth envelope. "
            "Calculated as d_max = max_t(h + \u03b6_t) where h is bathymetry and \u03b6 is sea surface elevation.\n\n"
            "Maximum water depth typically occurs during spring tides when tidal range is maximized. "
            "Water depth is calculated as the sum of bathymetry depth (h) and sea surface "
            "elevation (\u03b6). Values are adjusted from NAVD88 to MSL reference.\n\n"
            "The difference between maximum and minimum water depth approximates the "
            "tidal range at each location.\n\n"
            "Engineering applications include mooring system design considerations and "
            "understanding the full range of water depths at a site."
        ),
        "references": ["iec_62600_201", "fvcom"],
        # Scientific/engineering context
        "physical_meaning": "Maximum water depth observed over the year (deepest, typically at high tide)",
        "intended_usage": "Maximum mooring loads and installation planning",
        "intended_usage_detail": (
            "Defines maximum water depth for mooring system design and installation vessel "
            "requirements. Used to determine maximum mooring line lengths, assess tidal "
            "range impacts on operations, and plan for extreme high tide conditions. "
            "Important for cable routing and connection system design."
        ),
        "equation": r"$d_{\max} = \max\left(\left[(h + \zeta_t) \text{ for } t=1,...,T\right]\right)$",
        "equation_variables": [
            "h \u2014 bathymetry depth (m)",
            "\u03b6_t \u2014 sea surface elevation at time t (m)",
            "T = 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)",
        ],
    },
    # =========================================================================
    # Surface layer variables
    # =========================================================================
    "surface_mean_speed": {
        "display_name": "Surface Layer Mean Speed",
        "column_name": "vap_surface_layer_mean_sea_water_speed",
        "units": "m/s",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "Yearly average current speed at the surface layer (sigma_level_1)",
        "intended_usage": "Surface current assessment for navigation and floating devices",
        "intended_usage_detail": (
            "Characterizes current conditions at the water surface for floating tidal devices "
            "and navigation safety. Used to assess surface current impacts on vessel "
            "operations, floating platform stability, and cable/mooring system surface loads. "
            "Important for operations planning and environmental flow characterization."
        ),
        "equation": r"$\overline{U}_{\text{surface}} = \text{mean}\left(\left[U_{1,t} \text{ for } t=1,...,T\right]\right)$",
        "equation_variables": [
            r"$U_{1,t} = \sqrt{u_{1,t}^2 + v_{1,t}^2}$ is velocity magnitude at sigma_level_1 (surface) at time $t$ (m/s)",
            r"$T = 1$ year",
        ],
    },
    "surface_p95_speed": {
        "display_name": "Surface Layer 95th Percentile Speed",
        "column_name": "vap_surface_layer_95th_percentile_sea_water_speed",
        "units": "m/s",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "95th percentile of surface layer current speed over the year",
        "intended_usage": "Surface current design loads for floating systems",
        "intended_usage_detail": (
            "Design metric for floating tidal energy systems and surface infrastructure. "
            "Used to size mooring systems for floating platforms, design surface buoys and "
            "markers, and assess extreme surface current loads on cables and connectors."
        ),
        "equation": r"$U_{\text{surface},95} = \text{percentile}(95, \left[U_{1,t} \text{ for } t=1,...,T\right])$",
        "equation_variables": [
            r"$U_{1,t} = \sqrt{u_{1,t}^2 + v_{1,t}^2}$ is velocity magnitude at sigma_level_1 (surface) at time $t$ (m/s)",
            r"$T = 1$ year",
        ],
    },
    "surface_p99_speed": {
        "display_name": "Surface Layer 99th Percentile Speed",
        "column_name": "vap_surface_layer_99th_percentile_sea_water_speed",
        "units": "m/s",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "99th percentile of surface layer current speed over the year",
        "intended_usage": "Extreme surface current events for safety systems",
        "intended_usage_detail": (
            "Characterizes extreme surface current conditions for safety system design. "
            "Used for emergency response planning, setting operational limits for surface "
            "vessels and floating devices, and designing safety margins for surface "
            "infrastructure."
        ),
        "equation": r"$U_{\text{surface},99} = \text{percentile}(99, \left[U_{1,t} \text{ for } t=1,...,T\right])$",
        "equation_variables": [
            r"$U_{1,t} = \sqrt{u_{1,t}^2 + v_{1,t}^2}$ is velocity magnitude at sigma_level_1 (surface) at time $t$ (m/s)",
            r"$T = 1$ year",
        ],
    },
    "surface_max_speed": {
        "display_name": "Surface Layer Maximum Speed",
        "column_name": "vap_surface_layer_max_sea_water_speed",
        "units": "m/s",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "Absolute maximum surface layer current speed observed over the year",
        "intended_usage": "Ultimate surface current loads for survival analysis",
        "intended_usage_detail": (
            "Defines worst-case surface current conditions for ultimate load analysis. "
            "Essential for survival design of floating systems, determining maximum loads "
            "on surface infrastructure, and regulatory compliance for extreme surface "
            "conditions."
        ),
        "equation": r"$U_{\text{surface},\max} = \max\left(\left[U_{1,t} \text{ for } t=1,...,T\right]\right)$",
        "equation_variables": [
            r"$U_{1,t} = \sqrt{u_{1,t}^2 + v_{1,t}^2}$ is velocity magnitude at sigma_level_1 (surface) at time $t$ (m/s)",
            r"$T = 1$ year",
        ],
    },
    # =========================================================================
    # Grid resolution
    # =========================================================================
    "grid_resolution": {
        "display_name": "Grid Resolution",
        "column_name": "vap_grid_resolution",
        "units": "m",
        "included_on_atlas": True,
        # Documentation
        "one_liner": "Average edge length of triangular model grid cells",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#grid-resolution",
        "complete_description": (
            "Grid Resolution is the average edge length of the unstructured triangular model "
            "grid cells, indicating the spatial scale at which tidal currents are resolved by the "
            "FVCOM hydrodynamic model. Essential model metadata for assessing spatial uncertainty "
            "and determining appropriate applications. IEC 62600-201 requires <500 m for Stage 1 "
            "reconnaissance and <50 m for Stage 2 feasibility assessments. "
            "Calculated as R = \u2153(d\u2081 + d\u2082 + d\u2083) where d are triangle edge lengths.\n\n"
            "Grid Resolution is calculated as the average of the three edge lengths "
            "for each triangular grid cell, based on the original unstructured mesh defined "
            "by the model developers at Pacific Northwest National Laboratory. "
            "The unstructured triangular mesh allows variable resolution, with finer grids in "
            "areas of interest (channels, straits) and coarser grids in open water.\n\n"
            "Per IEC 62600-201 tidal energy resource assessment standards:\n"
            "\u2022 Stage 1 feasibility (reconnaissance-level) assessments require grid resolution < 500 m\n"
            "\u2022 Stage 2 (layout design) assessments require grid resolution < 50 m\n\n"
            "Engineering applications include assessing model fidelity and determining "
            "appropriate applications for the data."
        ),
        "references": ["iec_62600_201", "fvcom"],
        # Scientific/engineering context
        "physical_meaning": "Average edge length of triangular finite volume elements",
        "intended_usage": "Model accuracy assessment and validation",
        "intended_usage_detail": (
            "Indicates the spatial scale at which model results are resolved. Finer "
            "resolution (smaller values) provides more detailed results but requires "
            "greater computational resources. Used to assess model fidelity, determine "
            "appropriate applications for the data, and understand spatial limitations of "
            "the model output. Critical for validating model results against observations "
            "and determining if resolution is adequate for specific engineering "
            "applications. Per IEC 62600-201 standards: Stage 1 assessments require "
            "< 500 m resolution, while Stage 2 detailed studies require < 50 m resolution "
            "for areas of interest."
        ),
        "equation": r"$\text{Grid Resolution} = \frac{1}{3}(d_1 + d_2 + d_3)$",
        "equation_variables": [
            "d\u2081, d\u2082, d\u2083 \u2014 geodesic distances between triangle vertices (m)",
        ],
    },
    # =========================================================================
    # Direction variables
    # =========================================================================
    "ebb_direction": {
        "display_name": "Sea Water Ebb Tide To Direction",
        "column_name": "vap_sea_water_ebb_to_direction_sigma_level_3",
        "units": "m/s",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "",
        "intended_usage": "",
        "intended_usage_detail": "",
        "equation": "",
        "equation_variables": [],
    },
    "flood_direction": {
        "display_name": "Sea Water Flood Tide To Direction",
        "column_name": "vap_sea_water_flood_to_direction_sigma_level_1",
        "units": "m/s",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "",
        "intended_usage": "",
        "intended_usage_detail": "",
        "equation": "",
        "equation_variables": [],
    },
    # =========================================================================
    # Sea surface elevation variables
    # =========================================================================
    "mean_surface_elevation": {
        "display_name": "Mean Sea Surface Elevation",
        "column_name": "vap_surface_elevation",
        "units": "m",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "",
        "intended_usage": "",
        "intended_usage_detail": "",
        "equation": "",
        "equation_variables": [],
    },
    "tidal_range": {
        "display_name": "Range of Sea Surface Elevation",
        "column_name": "vap_tidal_range",
        "units": "m",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "",
        "intended_usage": "",
        "intended_usage_detail": "",
        "equation": "",
        "equation_variables": [],
    },
    # =========================================================================
    # Tidal period variables
    # =========================================================================
    "min_tidal_period": {
        "display_name": "Shortest Tidal Period",
        "column_name": "vap_min_tidal_period",
        "units": "hours",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "",
        "intended_usage": "",
        "intended_usage_detail": "",
        "equation": "",
        "equation_variables": [],
    },
    "max_tidal_period": {
        "display_name": "Longest Tidal Period",
        "column_name": "vap_max_tidal_period",
        "units": "hours",
        "included_on_atlas": False,
        "one_liner": "",
        "documentation_url": "",
        "complete_description": "",
        "references": [],
        "physical_meaning": "",
        "intended_usage": "",
        "intended_usage_detail": "",
        "equation": "",
        "equation_variables": [],
    },
}
