"""
Unified variable definitions for all tidal hindcast documentation
"""

import re

from src.gis_colors_registry import GIS_COLORS_REGISTRY

DOCUMENTATION_REGISTRY = {
    "data_availability": {
        "href": "https://mhkdr.openei.org/submissions/632",
        "full_text": "High Resolution Tidal Hindcast Data Repository",
        "short_text": "Dataset Repository",
        "keyword": "DATA_CITATION",
    },
    "data_access": {
        "href": "https://data.openei.org/s3_viewer?bucket=marine-energy-data&prefix=us-tidal%2F",
        "full_text": "High Resolution Tidal Hindcast Data Access on OpenEI",
        "short_text": "Dataset Repository",
        "keyword": "DATA_ACCESS",
    },
    "dataset_documentation": {
        "href": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/",
        "full_text": "Tidal Hindcast Dataset Documentation",
        "short_text": "Dataset Documentation",
        "keyword": "DOCUMENTATION",
    },
    "pnnl_team": {
        "href": "https://www.pnnl.gov/projects/ocean-dynamics-modeling",
        "full_text": "Pacific Northwest National Laboratory Ocean Dynamics and Modeling Group",
        "short_text": "PNNL Ocean Dynamics and Modeling Group",
        "keyword": "PNNL",
    },
    "nlr_team": {
        "href": "https://www.nlr.gov/water/resource-characterization",
        "full_text": "National Laboratory of the Rockies Marine Energy Resource Characterization Team",
        "short_text": "NLR Resource Characterization Team",
        "keyword": "NLR",
    },
    "doe": {
        "href": "https://www.energy.gov/",
        "full_text": "U.S. Department of Energy",
        "short_text": "DOE",
        "keyword": "DOE",
    },
    "wpto": {
        "href": "https://www.energy.gov/eere/water/water-power-technologies-office",
        "full_text": "Water Power Technologies Office",
        "short_text": "WPTO",
        "keyword": "WPTO",
    },
    "contact_email": {
        "href": "mailto:marineresource@nlr.gov",
        "full_text": "marineresource@nlr.gov",
        "short_text": "marineresource@nlr.gov",
        "keyword": "CONTACT_EMAIL",
    },
}

VARIABLE_REGISTRY = {
    # =========================================================================
    # Water column speed variables
    # =========================================================================
    "mean_current_speed": {
        "display_name": "Mean Current Speed",
        "column_name": "vap_water_column_mean_sea_water_speed",
        "units": "m/s",
        "long_name": "Mean Current Speed (depth-averaged)",
        "one_liner": "Annual average of depth-averaged current speed",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#mean-current-speed",
        "complete_description": (
            "annual average of the depth-averaged current velocity "
            "magnitude, representing the characteristic flow speed at each grid location under "
            "free-stream (undisturbed) conditions. This metric is intended for IEC 62600-201 "
            "Stage 1 reconnaissance-level analysis to identify areas with tidal current resources."
            "Engineering applications include initial site screening, comparing relative site "
            "potential across regions, and Stage 1 IEC 62600-201 tidal energy resource "
            "characterization."
        ),
        "references": [],
        # Scientific/engineering context
        "physical_meaning": "Yearly average of depth averaged current speed",
        "intended_usage": "Site screening and turbine selection for power generation",
        "intended_usage_detail": (
            "Primary metric for identifying viable tidal energy sites. Used to estimate "
            "annual energy production (AEP), compare site potential across regions, determine "
            "expected average viable current speeds for commercial deployment, "
            "and select appropriate turbine technology",
        ),
        "equation": r"$\overline{\overline{U}} = U_{\text{average}} = \text{mean}\left(\left[\text{mean}(U_{1,t}, ..., U_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$",
        "equation_variables": [
            r"$U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$, velocity magnitude at sigma layer $i$ at time $t$ $[\text{m/s}]$",
            r"$N_{\sigma} = 10$, sigma layers (terrain-following vertical layers dividing the water column into equal-thickness fractions from surface to seafloor)",
            r"$T$, 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)",
        ],
    },
    "p95_current_speed": {
        "display_name": "95th Percentile Current Speed",
        "column_name": "vap_water_column_95th_percentile_sea_water_speed",
        "units": "m/s",
        "long_name": "95th Percentile Current Speed",
        # Documentation
        "one_liner": "Estimated extreme current speed, outlier-tolerant and comparable across sites for reconnaissance-level assessment",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#95th-percentile-current-speed",
        "complete_description": (
            "95th Percentile Current Speed provides a robust, outlier-tolerant estimate of "
            "extreme current conditions at each grid location, intended for consistent "
            "cross-site comparison during reconnaissance-level resource characterization. "
            "Unlike the absolute maximum, this statistic is resistant to isolated numerical "
            "artifacts and transient model effects, making it a more reliable and reproducible "
            "basis for comparing extreme flow conditions across sites.\n\n"
            "This value is derived from a numerical hydrodynamic model and represents "
            "modeled conditions only. It should not be interpreted as a measured or "
            "ground-truth observation. Site-specific validation against in-situ measurements "
            "is recommended before use in detailed engineering design.\n\n"
            "Engineering applications include preliminary structural loading assessment, "
            "blade and support structure design screening, and initial fatigue analysis."
        ),
        "references": [],
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
            r"$U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$, velocity magnitude at sigma layer $i$ at time $t$ $[\text{m/s}]$",
            r"$\max_{\sigma}$, maximum value across all 10 sigma layers at each timestep",
            r"$P_{95}$, 95th percentile operator over the full time series",
            r"$N_{\sigma} = 10$, sigma layers",
            r"$T$, 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)",
        ],
    },
    "p99_current_speed": {
        "display_name": "99th Percentile Current Speed",
        "column_name": "vap_water_column_99th_percentile_sea_water_speed",
        "units": "m/s",
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
            r"$U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$, velocity magnitude at sigma level $i$ at time $t$ $[\text{m/s}]$",
            r"$N_{\sigma} = 10$, sigma layers",
            r"$T$, 1 year of hindcast data",
        ],
    },
    "max_current_speed": {
        "display_name": "Maximum Current Speed",
        "column_name": "vap_water_column_max_sea_water_speed",
        "units": "m/s",
        "long_name": "Maximum Current Speed",
        "one_liner": "Absolute maximum current speed observed over the hindcast year",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#maximum-current-speed",
        "complete_description": (
            "Maximum Current Speed is the absolute highest current speed observed at any "
            "depth and any time during the hindcast year. This metric defines the worst-case "
            "flow condition from the numerical model."
            "Important caveat: Because this is a single extreme value from a numerical model, "
            "it may be influenced by model artifacts or transient numerical effects. For "
            "engineering design loads, the 95th Percentile Current Speed is generally "
            "preferred as a more robust and statistically representative metric for extreme "
            "conditions. The maximum value is provided as an upper bound reference but "
            "should be used with caution for design purposes."
        ),
        "references": [],
        "physical_meaning": "Absolute maximum depth-max current speed observed over the year",
        "intended_usage": "Upper bound reference for extreme conditions",
        "intended_usage_detail": (
            "Provides an absolute upper bound on current speed from the model. While useful "
            "as a reference, this single extreme value may reflect model artifacts rather than "
            "physical reality. For structural design loads and survival analysis, the 95th "
            "percentile speed is generally preferred as a more robust metric. The maximum "
            "speed is most useful for quick screening of absolute worst-case conditions and "
            "as a sanity check against the percentile-based statistics."
        ),
        "equation": r"$U_{\max} = \max\left(\left[\max(U_{1,t}, ..., U_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$",
        "equation_variables": [
            r"$U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$, velocity magnitude at sigma level $i$ at time $t$ $[\text{m/s}]$",
            r"$N_{\sigma} = 10$, sigma layers",
            r"$T$, 1 year of hindcast data",
        ],
    },
    # =========================================================================
    # Power density variables
    # =========================================================================
    "mean_power_density": {
        "display_name": "Mean Power Density",
        "column_name": "vap_water_column_mean_sea_water_power_density",
        "units": "W/m\u00b2",
        "long_name": "Mean Power Density (depth-averaged)",
        # Documentation
        "one_liner": "Annual average of depth-averaged kinetic energy flux",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#mean-power-density",
        "complete_description": (
            "Mean Power Density is the annual average of the kinetic energy flux per unit area, "
            "representing the theoretical power available for extraction from the undisturbed "
            "tidal flow. The cubic relationship with velocity makes this metric highly "
            "sensitive to current speed variations. Used for Stage 1 resource characterization "
            "and site ranking to indicate theoretical resource magnitude."
            "Engineering applications include comparing relative energy availability between "
            "sites and initial economic feasibility screening."
        ),
        "references": ["hass_2011_assessment"],
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
            r"$P_{i,t} = \frac{1}{2} \rho U_{i,t}^3$, power density at sigma layer $i$ at time $t$ $[\text{W/m}^2]$",
            r"$\rho = 1025$, nominal seawater density (actual varies with temperature and salinity) $[\text{kg/m}^3]$",
            r"$U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$, velocity magnitude $[\text{m/s}]$",
            r"$N_{\sigma} = 10$, sigma layers",
            r"$T$, 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)",
        ],
    },
    "p95_power_density": {
        "display_name": "95th Percentile Power Density",
        "column_name": "vap_water_column_95th_percentile_sea_water_power_density",
        "units": "W/m\u00b2",
        "long_name": "95th Percentile Power Density",
        "one_liner": "Estimated extreme power density, outlier-tolerant and comparable across sites for reconnaissance-level assessment",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#95th-percentile-power-density",
        "complete_description": (
            "95th Percentile Power Density provides a robust, outlier-tolerant estimate of "
            "extreme power density conditions at each grid location, intended for consistent "
            "cross-site comparison during reconnaissance-level resource characterization. "
            "Unlike the absolute maximum, this statistic is resistant to isolated numerical "
            "artifacts and transient model effects. Due to the cubic relationship between "
            "velocity and power density, extreme values are particularly sensitive to model "
            "artifacts, making the 95th percentile a more reliable and reproducible basis "
            "for comparing extreme energy flux across sites.\n\n"
            "This value is derived from a numerical hydrodynamic model and represents "
            "modeled conditions only. It should not be interpreted as a measured or "
            "ground-truth observation. Site-specific validation against in-situ measurements "
            "is recommended before use in detailed engineering design.\n\n"
            "Engineering applications include preliminary extreme load assessment, "
            "power electronics sizing, and initial design margin estimation."
        ),
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
            r"$P_{i,t} = \frac{1}{2} \rho U_{i,t}^3$, power density with $\rho = 1025$ $[\text{kg/m}^3]$",
            r"$U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$, velocity magnitude at sigma level $i$ at time $t$ $[\text{m/s}]$",
            r"$N_{\sigma} = 10$, sigma layers",
            r"$T$, 1 year of hindcast data",
        ],
    },
    # =========================================================================
    # Depth variables
    # =========================================================================
    "mean_water_depth": {
        "display_name": "Sea Floor Depth",
        "column_name": "vap_sea_floor_depth",
        "units": "m (below NAVD88)",
        "long_name": "Model Sea Floor Depth from NAVD88",
        "one_liner": "Model bathymetry depth below NAVD88 vertical datum",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#sea-floor-depth",
        "complete_description": (
            "Sea Floor Depth is the model bathymetry representing the distance from the "
            "NAVD88 vertical datum to the seafloor at each grid cell center. This is a "
            "fundamental site characterization parameter required by IEC 62600-201 for all "
            "assessment stages. Bathymetry determines deployment feasibility, foundation "
            "type, and installation methodology."
            "The bathymetry values are from the FVCOM model grid, which was developed by "
            "Pacific Northwest National Laboratory using the best available bathymetric "
            "survey data for each region. Values are referenced to the NAVD88 vertical "
            "datum."
            "Engineering applications include deployment feasibility screening (current "
            "tidal turbine technology typically requires 20-50m depth), foundation type "
            "selection, installation vessel requirements, and cost modeling."
        ),
        "references": ["iec_62600_201"],
        "physical_meaning": "Model bathymetry depth below NAVD88 vertical datum",
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
            r"$h$, bathymetry below NAVD88 $[\text{m}]$",
            r"$\zeta_t$, sea surface elevation above NAVD88 at time $t$ $[\text{m}]$",
            r"$T$, 1 year of hindcast data",
        ],
    },
    "min_water_depth": {
        "display_name": "Minimum Water Depth",
        "column_name": "vap_water_column_height_min",
        "units": "m",
        "long_name": "Minimum Water Depth",
        # Documentation
        "one_liner": "Minimum water depth (during 1 year model run)",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#minimum-water-depth",
        "complete_description": (
            "Minimum Water Depth is the lowest water depth (surface to seafloor) "
            "observed at each grid location over the hindcast year, typically occurring during "
            "extreme low tide conditions. This metric defines the minimum vertical clearance "
            "available for device deployment and is critical for assessing depth constraints. "
            "Used in Stage 2 feasibility studies for turbine placement and collision avoidance."
            "The difference between maximum and minimum water depth approximates the "
            "tidal range at each location."
            "Engineering applications include assessing turbine clearance requirements and "
            "identifying areas where shallow water may limit device deployment."
        ),
        "references": [],
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
            r"$h$, bathymetry below NAVD88 $[\text{m}]$",
            r"$\zeta_t$, sea surface elevation above NAVD88 at time $t$ $[\text{m}]$",
            r"$T$, 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)",
        ],
    },
    "max_water_depth": {
        "display_name": "Maximum Water Depth",
        "column_name": "vap_water_column_height_max",
        "units": "m",
        "long_name": "Maximum Water Depth",
        # Documentation
        "one_liner": "Maximum water depth (during 1 year model run)",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#maximum-water-depth",
        "complete_description": (
            "Maximum Water Depth is the greatest water depth (surface to seafloor) "
            "observed at each grid location over the hindcast year, typically occurring during "
            "extreme high tide conditions. This metric represents the upper bound of water depth "
            "variability at each location. Used in Stage 2 feasibility studies for mooring system "
            "design, cable routing, and understanding the full operating depth envelope."
            "The difference between maximum and minimum water depth approximates the "
            "tidal range at each location."
            "Engineering applications include mooring system design considerations and "
            "understanding the full range of water depths at a site."
        ),
        "references": [],
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
            r"$h$, bathymetry below NAVD88 $[\text{m}]$",
            r"$\zeta_t$, sea surface elevation above NAVD88 at time $t$ $[\text{m}]$",
            r"$T$, 1 year of hindcast data (hourly for Alaska locations, half-hourly for others)",
        ],
    },
    # =========================================================================
    # Surface layer variables
    # =========================================================================
    "surface_mean_speed": {
        "display_name": "Surface Layer Mean Speed",
        "column_name": "vap_surface_layer_mean_sea_water_speed",
        "units": "m/s",
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
            r"$U_{1,t} = \sqrt{u_{1,t}^2 + v_{1,t}^2}$, velocity magnitude at sigma level 1 (surface) at time $t$ $[\text{m/s}]$",
            r"$T$, 1 year of hindcast data",
        ],
    },
    "surface_p95_speed": {
        "display_name": "Surface Layer 95th Percentile Speed",
        "column_name": "vap_surface_layer_95th_percentile_sea_water_speed",
        "units": "m/s",
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
            r"$U_{1,t} = \sqrt{u_{1,t}^2 + v_{1,t}^2}$, velocity magnitude at sigma level 1 (surface) at time $t$ $[\text{m/s}]$",
            r"$T$, 1 year of hindcast data",
        ],
    },
    "surface_p99_speed": {
        "display_name": "Surface Layer 99th Percentile Speed",
        "column_name": "vap_surface_layer_99th_percentile_sea_water_speed",
        "units": "m/s",
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
            r"$U_{1,t} = \sqrt{u_{1,t}^2 + v_{1,t}^2}$, velocity magnitude at sigma level 1 (surface) at time $t$ $[\text{m/s}]$",
            r"$T$, 1 year of hindcast data",
        ],
    },
    "surface_max_speed": {
        "display_name": "Surface Layer Maximum Speed",
        "column_name": "vap_surface_layer_max_sea_water_speed",
        "units": "m/s",
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
            r"$U_{1,t} = \sqrt{u_{1,t}^2 + v_{1,t}^2}$, velocity magnitude at sigma level 1 (surface) at time $t$ $[\text{m/s}]$",
            r"$T$, 1 year of hindcast data",
        ],
    },
    # =========================================================================
    # Grid resolution
    # =========================================================================
    "grid_resolution": {
        "display_name": "Grid Resolution",
        "column_name": "vap_grid_resolution",
        "units": "m",
        "long_name": "Grid Resolution",
        # Documentation
        "one_liner": "Average edge length of triangular model grid cells",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#grid-resolution",
        "complete_description": (
            "Grid Resolution is the average edge length of the unstructured triangular model "
            "grid cells, indicating the spatial scale at which tidal currents are resolved by the "
            "FVCOM hydrodynamic model. Essential model metadata for assessing spatial uncertainty "
            "and determining appropriate applications. IEC 62600-201 requires <500 m for Stage 1 "
            "reconnaissance and <50 m for Stage 2 feasibility assessments."
            "The unstructured triangular mesh allows variable resolution, with finer grids in "
            "areas of interest (channels, straits) and coarser grids in open water."
            "Per IEC 62600-201 tidal energy resource assessment standards:"
            "- Stage 1 feasibility (reconnaissance-level) assessments require grid resolution < 500 m"
            "- Stage 2 (layout design) assessments require grid resolution < 50 m"
            "Engineering applications include assessing model fidelity and determining "
            "appropriate applications for the data."
        ),
        "references": ["iec_62600_201"],
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
            r"$d_1, d_2, d_3$, geodesic distances between triangle vertices $[\text{m}]$",
        ],
    },
    # =========================================================================
    # Direction variables
    # =========================================================================
    "ebb_direction": {
        "display_name": "Sea Water Ebb Tide To Direction",
        "column_name": "vap_sea_water_ebb_to_direction_sigma_level_3",
        "units": "m/s",
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
        "column_name": "vap_sea_surface_elevation_mean",
        "units": "m (offset from NAVD88)",
        "long_name": "Mean Sea Surface Elevation (model MSL)",
        "one_liner": "Average sea surface elevation representing the model mean sea level offset from NAVD88",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#mean-sea-surface-elevation",
        "complete_description": (
            "Mean Sea Surface Elevation is the time-averaged sea surface height at each "
            "grid location, representing the model's mean sea level (MSL) as an offset "
            "from the NAVD88 vertical datum. This value provides the datum reference "
            "context for interpreting all other elevation and depth variables."
            "The difference between model MSL and NAVD88 varies by location due to "
            "geoid-ellipsoid separation, ocean dynamic topography, and regional sea level "
            "variations. This offset is essential for converting between the model's "
            "internal reference frame and standard geodetic datums."
            "Included on the atlas as a datum reference for coastal engineers and "
            "oceanographers to verify data consistency and convert between reference frames."
        ),
        "references": [],
        "physical_meaning": "Time-averaged sea surface elevation (model MSL offset from NAVD88)",
        "intended_usage": "Vertical datum reference and data quality verification",
        "intended_usage_detail": (
            "Provides context for interpreting elevation and depth values in the dataset. "
            "Used by coastal engineers to verify data consistency against known tidal "
            "datums, convert between reference frames, and check that model results are "
            "physically reasonable for the region."
        ),
        "equation": r"$\overline{\zeta} = \text{mean}(\zeta_t)$ for $t = 1, ..., T$",
        "equation_variables": [
            r"$\zeta_t$, sea surface elevation above NAVD88 at time $t$ $[\text{m}]$",
            r"$T$, 1 year of hindcast data",
        ],
    },
    "tidal_range": {
        "display_name": "Tidal Range",
        "column_name": "vap_tidal_range",
        "units": "m",
        "long_name": "Tidal Range (Max - Min Sea Surface Elevation)",
        "one_liner": "Difference between maximum and minimum sea surface elevation over the hindcast year",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#tidal-range",
        "complete_description": (
            "Tidal Range is the difference between the maximum and minimum sea surface "
            "elevation observed at each grid location over the hindcast year. This metric "
            "characterizes the tidal regime and vertical water level variability. "
            "IEC 62600-201 requires tidal range description as part of Stage 1 site "
            "characterization."
            "Tidal range is classified as microtidal (<2m), mesotidal (2-4m), or macrotidal "
            "(>4m). Larger tidal ranges indicate stronger tidal forcing, which generally "
            "correlates with stronger tidal currents but also creates greater challenges "
            "for device deployment due to water level variability."
            "Engineering applications include mooring system design (must accommodate full "
            "range of water levels), cable routing, and understanding the relationship "
            "between water level and current speed at a site."
        ),
        "references": ["iec_62600_201"],
        "physical_meaning": "Maximum minus minimum sea surface elevation over the hindcast year",
        "intended_usage": "Tidal regime classification and mooring design",
        "intended_usage_detail": (
            "Characterizes the vertical tidal variability at each location. Used to classify "
            "the tidal regime (micro/meso/macrotidal), assess water level impacts on device "
            "operations, design mooring systems that accommodate full tidal excursion, and "
            "plan cable routing to handle depth changes. Important context for understanding "
            "the relationship between tidal forcing and current speed."
        ),
        "equation": r"$R = \zeta_{\max} - \zeta_{\min} = \max(\zeta_t) - \min(\zeta_t)$",
        "equation_variables": [
            r"$\zeta_t$, sea surface elevation at time $t$ $[\text{m}]$",
            r"$\zeta_{\max}$, maximum sea surface elevation over the year $[\text{m}]$",
            r"$\zeta_{\min}$, minimum sea surface elevation over the year $[\text{m}]$",
            r"$T$, 1 year of hindcast data",
        ],
    },
    # =========================================================================
    # Tidal period variables
    # =========================================================================
    "min_tidal_period": {
        "display_name": "Shortest Tidal Period",
        "column_name": "vap_min_tidal_period",
        "units": "hours",
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
    # Maximum power density
    # =========================================================================
    "max_power_density": {
        "display_name": "Maximum Power Density",
        "column_name": "vap_water_column_max_sea_water_power_density",
        "units": "W/m\u00b2",
        "long_name": "Maximum Power Density",
        "one_liner": "Absolute maximum depth-averaged power density observed over the hindcast year",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#maximum-power-density",
        "complete_description": (
            "Maximum Power Density is the absolute highest kinetic energy flux per unit "
            "area observed at any time during the hindcast year. Due to the cubic "
            "relationship between velocity and power, the maximum power density "
            "is highly sensitive to extreme current speed events."
            "Important caveat: Because power density scales with the cube of velocity, "
            "any model artifacts or numerical transients in the maximum speed are amplified "
            "in this metric. The mean power density and 95th percentile speed are generally "
            "more robust metrics for resource characterization. The maximum power density "
            "is provided as an upper bound reference and sanity check."
        ),
        "references": [],
        "physical_meaning": "Absolute maximum depth-averaged power density over the year",
        "intended_usage": "Upper bound reference for peak resource conditions",
        "intended_usage_detail": (
            "Provides the absolute upper bound on power density from the model. Due to "
            "the cubic velocity-power relationship, this value is sensitive to model "
            "artifacts. Mean power density is preferred for resource quantification and "
            "economic analysis. The maximum is useful as a reference for understanding "
            "peak conditions and as a sanity check."
        ),
        "equation": r"$P_{\max} = \max\left(\left[\text{mean}(P_{1,t}, ..., P_{N_{\sigma},t}) \text{ for } t=1,...,T\right]\right)$",
        "equation_variables": [
            r"$P_{i,t} = \frac{1}{2} \rho U_{i,t}^3$, power density with $\rho = 1025$ $[\text{kg/m}^3]$",
            r"$U_{i,t} = \sqrt{u_{i,t}^2 + v_{i,t}^2}$, velocity magnitude $[\text{m/s}]$",
            r"$N_{\sigma} = 10$, sigma layers",
            r"$T$, 1 year of hindcast data",
        ],
    },
    # =========================================================================
    # Average tidal period
    # =========================================================================
    "average_tidal_period": {
        "display_name": "Average Tidal Period",
        "column_name": "vap_average_tidal_period",
        "units": "hours",
        "long_name": "Average Tidal Period",
        "one_liner": "Mean period between successive high tides over the hindcast year",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#average-tidal-period",
        "complete_description": (
            "Average Tidal Period is the mean time between successive high tide peaks "
            "at each grid location, characterizing the dominant tidal frequency. This "
            "metric indicates whether the tidal regime is semi-diurnal (~12.4 hours), "
            "diurnal (~24.8 hours), or mixed."
            "The tidal period directly determines the power generation cycle length and "
            "is relevant for grid integration planning, energy storage sizing, and "
            "understanding resource intermittency patterns. Semi-diurnal sites produce "
            "four slack-to-peak cycles per day, while diurnal sites produce two."
            "Engineering applications include energy production scheduling, grid "
            "integration analysis, and energy storage system sizing."
        ),
        "references": [],
        "physical_meaning": "Mean time between successive high tides",
        "intended_usage": "Tidal regime classification and energy scheduling",
        "intended_usage_detail": (
            "Classifies the tidal regime (semi-diurnal ~12.4h, diurnal ~24.8h, or mixed) "
            "to inform energy production scheduling and grid integration planning. "
            "Semi-diurnal sites provide more frequent generation cycles. Important for "
            "energy storage sizing and understanding power intermittency patterns."
        ),
        "equation": r"$\overline{T}_{\text{tide}} = \text{mean}(T_1, T_2, ..., T_N)$",
        "equation_variables": [
            r"$T_i$, time between successive high tide peaks $i$ and $i+1$ $[\text{hours}]$",
            r"$N$, number of tidal cycles in the hindcast year",
        ],
    },
    # =========================================================================
    # Distance to shore
    # =========================================================================
    "distance_to_shore": {
        "display_name": "Distance to Shore",
        "column_name": "vap_distance_to_shore",
        "units": "NM",
        "long_name": "Distance to Shore",
        "one_liner": "Geodesic distance from grid cell center to nearest shoreline",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#distance-to-shore",
        "complete_description": (
            "Distance to Shore is the geodesic distance from each grid cell center to "
            "the nearest shoreline point, calculated using the GSHHG (Global Self-consistent "
            "Hierarchical High-resolution Geography) shoreline database. Reported in "
            "nautical miles (NM)."
            "Distance to shore is a practical siting constraint that affects cable cost, "
            "grid connection feasibility, and operations and maintenance logistics. "
            "NREL site screening methodology uses a threshold of <20 km (~10.8 NM) to "
            "nearest transmission infrastructure."
            "Engineering applications include cable routing cost estimation, grid "
            "connection planning, and O&M logistics assessment."
        ),
        "references": [],
        "physical_meaning": "Geodesic distance from grid cell center to nearest shoreline",
        "intended_usage": "Cable cost estimation and O&M logistics",
        "intended_usage_detail": (
            "Key practical siting constraint for cost estimation. Longer distances to "
            "shore increase subsea cable costs, reduce grid connection feasibility, "
            "and increase transit time for maintenance vessels. Used in LCOE calculations "
            "and logistics planning."
        ),
        "equation": r"$d = \text{haversine}(\text{cell center}, \text{nearest shoreline point})$",
        "equation_variables": [
            r"$d$, geodesic distance calculated using GSHHG shoreline database $[\text{NM}]$",
            r"$1 \text{ NM} = 1.852 \text{ km}$",
        ],
    },
    # =========================================================================
    # Sea surface elevation extremes (sanity check variables)
    # =========================================================================
    "sea_surface_elevation_high_tide_max": {
        "display_name": "Maximum Sea Surface Elevation at High Tide",
        "column_name": "vap_sea_surface_elevation_high_tide_max",
        "units": "m (relative to model MSL)",
        "long_name": "Max Sea Surface Elevation at High Tide",
        "one_liner": "Highest sea surface elevation observed during high tide over the hindcast year",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#max-sea-surface-elevation-at-high-tide",
        "complete_description": (
            "Maximum Sea Surface Elevation at High Tide is the highest sea surface "
            "elevation value observed during high tide conditions over the hindcast year, "
            "relative to the model's mean sea level. This typically occurs during spring "
            "tides when astronomical tidal forcing is maximized."
            "This metric serves as a sanity check for data quality, allowing coastal "
            "engineers and oceanographers to verify that modeled extreme water levels "
            "are physically reasonable for the region. Together with the low tide minimum, "
            "it provides confidence bounds on the vertical water level envelope."
        ),
        "references": [],
        "physical_meaning": "Highest sea surface elevation at high tide over the year",
        "intended_usage": "Data quality verification and extreme water level reference",
        "intended_usage_detail": (
            "Sanity check metric for coastal engineers to verify model output consistency. "
            "The maximum high tide elevation should be physically reasonable for the "
            "region and consistent with known tidal characteristics. Also useful for "
            "understanding the upper bound of water level variability."
        ),
        "equation": r"$\zeta_{\text{HT,max}} = \max(\zeta_t | t \in \text{high tide peaks})$",
        "equation_variables": [
            r"$\zeta_t$, sea surface elevation relative to model MSL at time $t$ $[\text{m}]$",
            r"High tide peaks identified from the sea surface elevation time series",
        ],
    },
    "sea_surface_elevation_low_tide_min": {
        "display_name": "Minimum Sea Surface Elevation at Low Tide",
        "column_name": "vap_surface_elevation_low_tide_min",
        "units": "m (relative to model MSL)",
        "long_name": "Min Sea Surface Elevation at Low Tide",
        "one_liner": "Lowest sea surface elevation observed during low tide over the hindcast year",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#min-sea-surface-elevation-at-low-tide",
        "complete_description": (
            "Minimum Sea Surface Elevation at Low Tide is the lowest sea surface "
            "elevation value observed during low tide conditions over the hindcast year, "
            "relative to the model's mean sea level. This typically occurs during spring "
            "tides when astronomical tidal forcing is maximized."
            "This metric serves as a sanity check for data quality, allowing coastal "
            "engineers and oceanographers to verify that modeled extreme low water levels "
            "are physically reasonable for the region. Together with the high tide maximum, "
            "it provides confidence bounds on the vertical water level envelope."
        ),
        "references": [],
        "physical_meaning": "Lowest sea surface elevation at low tide over the year",
        "intended_usage": "Data quality verification and extreme water level reference",
        "intended_usage_detail": (
            "Sanity check metric for coastal engineers to verify model output consistency. "
            "The minimum low tide elevation should be physically reasonable for the "
            "region and consistent with known tidal characteristics. Also useful for "
            "understanding the lower bound of water level variability and assessing "
            "minimum clearance conditions."
        ),
        "equation": r"$\zeta_{\text{LT,min}} = \min(\zeta_t | t \in \text{low tide troughs})$",
        "equation_variables": [
            r"$\zeta_t$, sea surface elevation relative to model MSL at time $t$ $[\text{m}]$",
            r"Low tide troughs identified from the sea surface elevation time series",
        ],
    },
    # =========================================================================
    # Structural / identity columns (included on atlas for data access)
    # =========================================================================
    "face_id": {
        "display_name": "Face ID",
        "column_name": "face_id",
        "units": "",
        "long_name": "Face ID",
        "one_liner": "Location specific unique integer identifier for each triangular grid element",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#face_id",
    },
    "center_latitude": {
        "display_name": "Center Latitude",
        "column_name": "lat_center",
        "units": "degrees_north",
        "long_name": "Center Latitude",
        "one_liner": "Latitude of the triangular element centroid (WGS84)",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#center_latitude",
    },
    "center_longitude": {
        "display_name": "Center Longitude",
        "column_name": "lon_center",
        "units": "degrees_east",
        "long_name": "Center Longitude",
        "one_liner": "Longitude of the triangular element centroid (WGS84)",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#center_longitude",
    },
    "full_year_s3_uri": {
        "display_name": "S3 URI for Full Year Time Series Data",
        "column_name": "full_year_data_s3_uri",
        "units": "",
        "long_name": "S3 URI for Full Year Time Series Data",
        "one_liner": "direct link (S3 URI) to download the one-year hindcast time series (parquet) for this location. Includes speed, direction, for 10 uniform sigma levels at half-hourly (lower 48) or hourly (Alaska) intervals.",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#full_year_s3_uri",
    },
    "full_year_https_url": {
        "display_name": "HTTPS URL for Full Year Time Series Data",
        "column_name": "full_year_data_https_url",
        "units": "",
        "long_name": "HTTPS URL for Full Year Time Series Data",
        "one_liner": "direct link (HTTPS)  to download the one-year hindcast time series (parquet) for this location. Includes speed, direction, for 10 uniform sigma levels at half-hourly (lower 48) or hourly (Alaska) intervals",
        "documentation_url": "https://natlabrockies.github.io/Marine_Energy_Resource_Characterization/tidal-hindcast/#full_year_https_url",
    },
}

POLYGON_COLUMNS = [
    "element_corner_1_lat",
    "element_corner_1_lon",
    "element_corner_2_lat",
    "element_corner_2_lon",
    "element_corner_3_lat",
    "element_corner_3_lon",
]

# Atlas columns: display names and metadata live in VARIABLE_REGISTRY (long_name, units)
ATLAS_COLUMNS = [
    *POLYGON_COLUMNS,
    # ── Core Resource Metrics (IEC 62600-201 Stage 1) ──
    "vap_water_column_mean_sea_water_speed",
    "vap_water_column_mean_sea_water_power_density",
    "vap_water_column_95th_percentile_sea_water_speed",
    "vap_water_column_95th_percentile_sea_water_power_density",
    # "vap_water_column_max_sea_water_speed",
    # "vap_water_column_max_sea_water_power_density",
    # ── Bathymetry & Depth ──
    "vap_water_column_height_min",
    "vap_water_column_height_max",
    "vap_tidal_range",
    # "vap_sea_floor_depth",
    "vap_distance_to_shore",
    # ── Sea Surface Elevation & Tidal Characteristics ──
    "vap_sea_surface_elevation_high_tide_max",
    "vap_surface_elevation_low_tide_min",
    # "vap_sea_surface_elevation_mean",
    # ── Site Context ──
    "vap_grid_resolution",
    # ── Location Identity ──
    "face_id",
    "lat_center",
    "lon_center",
    # ── Data Access ──
    "full_year_data_https_url",
    "full_year_data_s3_uri",
]

# Derive included_on_atlas from ATLAS_COLUMNS (single source of truth)
_atlas_column_set_all = set(ATLAS_COLUMNS)
for _var_entry in VARIABLE_REGISTRY.values():
    _var_entry["included_on_atlas"] = _var_entry["column_name"] in _atlas_column_set_all


dataset_info = (
    "Source: <DATA_CITATION>, funded by <DOE> <WPTO>. "
    "Modeled by <PNNL>; standardized and released by <NLR>. "
    "See <DOCUMENTATION> for methodology, citations, and full dataset access. "
    "Contact <CONTACT_EMAIL> with questions."
)


def _html_link(text, href):
    if href.startswith("mailto:"):
        return f'<a href="{href}">{text}</a>'
    return f'<a href="{href}" target="_blank" rel="noopener noreferrer">{text}</a>'


_LINK_FORMATTERS = {
    "text": lambda text, href: text,
    "markdown": lambda text, href: f"[{text}]({href})",
    "html": _html_link,
}


def _render(
    text_spec, keyword_spec, variable_spec, link_fmt, keyword_text_field="full_text"
):
    """Render a template by substituting keyword and variable placeholders.

    Keywords are rendered using link_fmt(display_text, href). Variable entries
    are substituted as plain text. Raises ValueError on unresolved placeholders.
    """
    keyword_lookup = {entry["keyword"]: entry for entry in keyword_spec.values()}
    result = text_spec
    for key, entry in keyword_lookup.items():
        result = result.replace(
            f"<{key}>", link_fmt(entry[keyword_text_field], entry["href"])
        )
    # Keys whose values are prose descriptions that should be lowercased mid-sentence
    _lowercase_mid_sentence = {"one_liner", "complete_description"}
    for key, value in variable_spec.items():
        placeholder = f"<{key.upper()}>"
        val_str = str(value)
        if placeholder not in result:
            continue
        if key in _lowercase_mid_sentence:
            parts = result.split(placeholder)
            result = parts[0]
            for part in parts[1:]:
                preceding = result.rstrip()
                if not preceding or preceding[-1] in ".!?\n":
                    result += val_str + part
                else:
                    # Lowercase first char unless the first word is all-caps
                    # (acronym like "HTTPS", "URI") which must stay as-is.
                    first_word = val_str.split()[0] if val_str else ""
                    if first_word.isupper() and len(first_word) > 1:
                        result += val_str + part
                    else:
                        result += val_str[0].lower() + val_str[1:] + part
        else:
            result = result.replace(placeholder, val_str)
    unresolved = re.findall(r"<[A-Z_]+>", result)
    if unresolved:
        raise ValueError(f"Unresolved placeholders: {unresolved}")
    return result


def _render_all_formats(text_spec, keyword_spec, variable_spec):
    """Render a template to text, markdown, and HTML."""
    return {
        fmt: _render(text_spec, keyword_spec, variable_spec, link_fmt)
        for fmt, link_fmt in _LINK_FORMATTERS.items()
    }


# This goes with each variable on the atlas page
# Concise, points the user to the right place
def atlas_variable_spec(variable_spec, keyword_spec):
    units_part = " [<UNITS>]" if variable_spec.get("units") else ""
    first_sentence = (
        f"<DISPLAY_NAME>{units_part} is the <ONE_LINER>. "
        "For more detail, see <VARIABLE_LINK>."
    )
    rest = dataset_info
    display_lower = variable_spec["display_name"].lower()
    augmented_keywords = dict(keyword_spec)
    augmented_keywords["_variable_link"] = {
        "href": variable_spec["documentation_url"],
        "full_text": f"complete {display_lower} documentation",
        "short_text": f"complete {display_lower} documentation",
        "keyword": "VARIABLE_LINK",
    }
    # Text and markdown: single paragraph
    flat_spec = first_sentence + " " + rest
    result = {
        "display_name": variable_spec["display_name"],
        "text": _render(
            flat_spec, augmented_keywords, variable_spec, _LINK_FORMATTERS["text"]
        ),
        "markdown": _render(
            flat_spec, augmented_keywords, variable_spec, _LINK_FORMATTERS["markdown"]
        ),
    }
    # HTML: wrap in <div> with first sentence in its own <p> with <br/>
    html_first = _render(
        first_sentence, augmented_keywords, variable_spec, _LINK_FORMATTERS["html"]
    )
    html_rest = _render(
        rest, augmented_keywords, variable_spec, _LINK_FORMATTERS["html"]
    )
    result["html"] = f"<div><p>{html_first}<br/></p><p>{html_rest}</p></div>"
    return result


def documentation_variable_spec(variable_spec, keyword_spec):
    units_part = " [<UNITS>]" if variable_spec.get("units") else ""
    text_spec = (
        f"<DISPLAY_NAME>{units_part} is the <COMPLETE_DESCRIPTION>. " + dataset_info
    )
    result = {
        "display_name": variable_spec["display_name"],
        **_render_all_formats(text_spec, keyword_spec, variable_spec),
    }
    if "equation" in variable_spec:
        result["equation"] = variable_spec["equation"]
    if "equation_variables" in variable_spec:
        result["equation_variables"] = variable_spec["equation_variables"]
    return result


atlas_variable_specification = {}
documentation_variable_specification = {}

# Only generate specs for non-polygon columns on the atlas
_atlas_column_set = set(ATLAS_COLUMNS) - set(POLYGON_COLUMNS)

# Build set of column_names that have color styling

_colored_layer_columns = {
    VARIABLE_REGISTRY[key]["column_name"]
    for key in GIS_COLORS_REGISTRY
    if key in VARIABLE_REGISTRY
}

for _var_key, _var_entry in VARIABLE_REGISTRY.items():
    _col_name = _var_entry["column_name"]
    if _col_name not in _atlas_column_set:
        continue
    if "documentation_url" not in _var_entry:
        continue
    spec = atlas_variable_spec(_var_entry, DOCUMENTATION_REGISTRY)
    spec["show_as_layer_with_color_spec"] = _col_name in _colored_layer_columns
    atlas_variable_specification[_col_name] = spec
    if "complete_description" in _var_entry:
        documentation_variable_specification[_col_name] = documentation_variable_spec(
            _var_entry, DOCUMENTATION_REGISTRY
        )
