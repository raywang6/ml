import polars as pl
from datetime import datetime, timedelta
from typing import Iterable, Dict

def get_files_in_namecard_dirs(root_dir):
    from pathlib import Path
    import joblib
    root = Path(root_dir)
    files = []
    for path in root.rglob('*'):
        if path.is_file() and path.name.endswith('_namecard'):
            files.append(path)
    res = []
    for ipath in files:
        rdict = joblib.load(ipath)
        rdict['symbol'] = ipath.parent.name
        rdict['mname'] = ipath.name.split('_namecard')[0]
        res.append(rdict)
    return pl.DataFrame(res)


def flatten(_xs, /):
    for x in _xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def generate_features_from_config(
    data: pl.DataFrame,
    dtname: str,
    config: Dict
    ) -> pl.DataFrame:
    return data.sort(dtname).with_columns(
            flatten(
                (
                        expr().alias(fname) for fname, expr in config.items()
                )
            )
        )




def generate_training_dates(start, end):
    # Helper function to parse and truncate a date
    def truncate_to_first_day(date):
        if isinstance(date, str):
            dt = datetime.strptime(date, "%Y-%m-%d")
        else:
            dt = date
        return dt.replace(day=1)
    # Parse and truncate start and end dates
    start_trunc = truncate_to_first_day(start)
    end_trunc = truncate_to_first_day(end)
    # Check if start is after end after truncation
    if start_trunc > end_trunc:
        return pl.Series(dtype=pl.Datetime(time_unit = 'ns', time_zone = 'UTC'))
    # Generate the date range
    date_series = pl.date_range(
        start=start_trunc,
        end=end_trunc,
        interval="1mo",
        closed="both",
        eager=True,
        time_unit = 'ns', time_zone = 'UTC'
    )
    return date_series
