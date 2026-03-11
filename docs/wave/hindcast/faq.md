# FAQ

## What time period does the dataset cover?

The wave hindcast covers 32 years from 1979 to 2010 at 3-hour intervals. The project team intends to extend this to 2020, pending DOE support.

## What spatial resolution is available?

Resolution varies from approximately 200 meters in shallow coastal waters to ~10 km in deep offshore waters, using SWAN's unstructured grid capability.

## Which regions are currently available?

West Coast, Atlantic, and Hawaii are available. Alaska, Gulf of Mexico, Puerto Rico, U.S. Virgin Islands, and U.S. Pacific Island Territories are planned.

## How do I access virtual buoy data?

Virtual buoy hourly data is available at:

```
s3://wpto-pds-US_wave/v1.0.0/virtual_buoy/${domain}/
```

Or via HSDS at `/nlr/US_wave/virtual_buoy/`.

## What does the scale factor mean?

Data is stored as scaled integers. To get physical values:

```python
physical_value = dataset[...] / dataset.attrs['scale_factor']
```

The `rex` library handles this automatically.
