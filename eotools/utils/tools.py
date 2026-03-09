import xarray as xr


def deduplicate_dims(ds: xr.Dataset, mapping: dict[str, list[str]]) -> xr.Dataset:
    """Deduplicate dimensions with the same name by renaming them to unique names."""
    # Ensure no overlap between keys and any values
    all_keys = set(mapping.keys())
    all_values = set()
    for values in mapping.values():
        all_values.update(values)
    overlapping = all_keys & all_values
    if overlapping:
        raise ValueError(f"Keys overlap with values: {overlapping}")
    
    new_ds = ds.copy()
    for var_name in new_ds.data_vars:
        i = 0
        current_dims = ds[var_name].dims
        new_dims = []
        for d in current_dims:
            if d in mapping:
                new_dims.append(mapping[d][i])
                i += 1
            else:
                new_dims.append(d)
        
        # recreate the DataArray
        da = new_ds[var_name]
        new_da = xr.DataArray(da.values, dims=new_dims, attrs=da.attrs, name=da.name)
        new_ds[var_name] = new_da

    coords = {k: v.values for (k, v) in ds.coords.items() if k not in mapping}
    for k, list_v in mapping.items():
        for v in list_v:
            coords[v] = ds.coords[k].values
    
    return new_ds.assign_coords(coords).drop(mapping)