import pandas as pd

def calculate_data_shape(x):
    return x.shape

def take_columns(x):
    return x.columns

def calculate_target_ratio(x, target_name):
    return round(x[target_name].mean(), 2)

def calculate_data_dtypes(x):
    return [x.dtypes[x.dtypes != 'object'].count(),
            x.dtypes[x.dtypes == 'object'].count()]

def calculate_cheap_apartment(x):
    return round((x.price_doc <= 10 ** 6).sum())

def calculate_squad_in_cheap_apartment(x):
    return round(x[x.price_doc <= 10 ** 6].full_sq.mean())

def calculate_mean_price_in_new_housing(x):
    return round(x[(x.num_room == 3) & (x.build_year >= 2010)].price_doc.mean())

def calculate_mean_squared_by_num_rooms(x):
    return round(x.groupby('num_room').full_sq.mean(), 2)

def calculate_squared_stats_by_material(x):
    return round(x.groupby('material').price_doc.agg(['max', 'min']).rename(columns={"min": "amin", "max": "amax"}), 2)

def calculate_crosstab(x):
    return round(x.pivot_table('price_doc', index=['sub_area'], columns=['product_type'], aggfunc='mean').fillna(0), 2)