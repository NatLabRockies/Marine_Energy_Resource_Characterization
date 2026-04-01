# Sigma Layers

## What Are Sigma Layers?

A sigma layer is a terrain-following vertical coordinate system where the water column is divided into layers that conform to both the seafloor (bottom) and the time-varying sea surface (top). Sigma ($\sigma$) expresses vertical position as a proportion of total water depth, ranging from $\sigma = 0$ at the surface to $\sigma = -1$ at the seafloor. Layers stretch and compress dynamically as water depth varies with tides.

## Vertical Structure

The FVCOM model uses **10 uniformly spaced sigma layers**. Model variables (velocity, etc.) are computed at each layer center. The physical depth of any sigma value is:

$$
\text{depth} = -D \times \sigma
$$

where $D = h + \zeta$ (bathymetry + sea surface elevation).

<figure markdown="span">
  ![Sigma coordinate system over a tidal cycle](../../assets/sigma_timeseries.png){ width="100%" }
  <figcaption>Sigma coordinate system over a 4-day tidal cycle at Cook Inlet, Alaska. Top panel shows sigma level boundaries (solid lines) and layer centers (dashed lines) in elevation relative to NAVD88. The seafloor remains fixed while the sea surface oscillates with tides, causing sigma layers to expand during high tide and compress during low tide. Middle and bottom panels show the corresponding total water depth and individual layer height variations.</figcaption>
</figure>

<figure markdown="span">
  ![Sigma layer comparison at low and high tide](../../assets/sigma_layer_comparison.png){ width="100%" }
  <figcaption>Sigma levels and layer centers at low tide (left) and high tide (right) for Cook Inlet, Alaska. The 10 sigma layers maintain their proportional positions within the water column while physical depths change with tidal elevation. Layer centers (dots) indicate where model variables are computed.</figcaption>
</figure>

## Implications for Data Users

While sigma coordinates provide consistent data structure across varying bathymetry and tidal conditions, they require interpolation to extract data at fixed absolute depths. Queries such as "average current speed at 8 m depth" require calculating which sigma layers correspond to that depth at each timestep.

## Depth-Averaging

A **depth-averaged** value is computed by averaging a quantity across all vertical layers (sigma layers) at a given horizontal location and time. Depth-averaging produces a single representative value for the entire water column, useful for characterizing overall flow conditions while accounting for vertical velocity profiles.

All summary variables in this dataset (mean current speed, power density, etc.) use depth-averaged values unless otherwise noted.

## Depth-Maximum

The **depth-maximum** is the maximum value of a quantity across all vertical layers at a given horizontal location and time. The depth-maximum captures the peak value occurring anywhere in the water column, which is relevant for structural loading calculations where components must withstand the highest forces at any depth.

The 95th percentile variables in this dataset use depth-maximum values to capture extreme conditions.


--8<-- "docs/tidal/high_resolution_hindcast/_cite-widget.md"
