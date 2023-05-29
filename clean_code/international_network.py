import numpy as np
import pandas as pd
import concurrent
from tqdm import tqdm

def icio_international(icio_path, recipe_df, countries, country_sector_names):

    icio_df = pd.read_csv(icio_path, index_col=0)

    # recombine mexico
    mex_df = icio_df[icio_df.index.str.contains("MEX")]
    mx1_df = icio_df[icio_df.index.str.contains("MX1")]
    mx2_df = icio_df[icio_df.index.str.contains("MX2")]
    for i, row in mex_df.iterrows():
        if not row.name in ["MEX_TAXSUB"]:
            sec = row.name.split("_")[1]
            
            assert (mx1_df.loc[f"MX1_{sec}"].index == row.index).all()
            assert (mx2_df.loc[f"MX2_{sec}"].index == row.index).all()

            icio_df.loc[row.name] += mx1_df.loc[f"MX1_{sec}"]
            icio_df.loc[row.name] += mx2_df.loc[f"MX2_{sec}"]

    # recombine china
    chn_df = icio_df[icio_df.index.str.contains("CHN")]
    cn1_df = icio_df[icio_df.index.str.contains("CN1")]
    cn2_df = icio_df[icio_df.index.str.contains("CN2")]
    for i, row in chn_df.iterrows():
        if not row.name in ["CHN_TAXSUB"]:
            sec = row.name.split("_")[1]
            
            assert (cn1_df.loc[f"CN1_{sec}"].index == row.index).all()
            assert (cn2_df.loc[f"CN2_{sec}"].index == row.index).all()

            icio_df.loc[row.name] += cn1_df.loc[f"CN1_{sec}"]
            icio_df.loc[row.name] += cn2_df.loc[f"CN2_{sec}"]

    gfcf_cols = icio_df.columns[icio_df.columns.map(lambda x: "GFCF" in x)]
    gfcf_df = icio_df[gfcf_cols]
    
    country_sectors = len(country_sector_names)
    n_sec = len(recipe_df)

    international_flows = np.zeros((len(countries), len(countries)))
    international_matrix = np.zeros((country_sectors, country_sectors))
    for i, i_cou in enumerate(countries):
        for j, j_cou in enumerate(countries):

            rows = gfcf_df.index.where(gfcf_df.index.map(lambda x: i_cou in x and not "TAXSUB" in x)).dropna()
            cols = gfcf_df.columns.where(gfcf_cols.map(lambda x: j_cou in x and not "TAXSUB" in x)).dropna()

            international_flows[i,j] = gfcf_df.loc[rows, cols].sum(axis=0,skipna=True).sum(axis=0,skipna=True)

            international_matrix[i*n_sec:(i+1)*n_sec, j*n_sec:(j+1)*n_sec] = recipe_df.values * international_flows[i,j]
    
    return international_flows, international_matrix

def get_countries(icio_folder, oecd_bea):

    icio_df = pd.read_csv(f"{icio_folder}/ICIO2021_2000.csv")
    gfcf_cols = icio_df.columns[icio_df.columns.map(lambda x: "GFCF" in x)]
    countries = gfcf_cols.map(lambda x: x.split("_")[0])
    country_sector_names = []

    for cou in countries:
        for sec in list(oecd_bea.keys()):
            country_sector_names.append(cou + "_" + sec)
    
    return countries, country_sector_names

def icio_international_series(icio_folder, oecd_bea, oecd_recipe, years):

    temp_recipe = pd.DataFrame(oecd_recipe[0,:,:])
    temp_recipe.columns = list(oecd_bea.keys())
    temp_recipe.index = list(oecd_bea.keys())

    icio_df = pd.read_csv(f"{icio_folder}/ICIO2021_{years[0]}.csv")
    gfcf_cols = icio_df.columns[icio_df.columns.map(lambda x: "GFCF" in x)]
    countries = gfcf_cols.map(lambda x: x.split("_")[0])
    country_sector_names = []

    for cou in countries:
        for sec in temp_recipe.columns:
            country_sector_names.append(cou + "_" + sec)

    n_country_sectors = len(list(oecd_bea.keys())) * len(countries)
    international_flow_series = np.zeros((len(years), len(countries), len(countries)))
    international_matrix_series = np.zeros((len(years), n_country_sectors, n_country_sectors))
    for t in tqdm(range(len(years))):
        temp_recipe = pd.DataFrame(oecd_recipe[t,:,:])
        temp_recipe.columns = list(oecd_bea.keys())
        temp_recipe.index = list(oecd_bea.keys())
        international_flow_series[t,:,:], international_matrix_series[t,:,:] = \
            icio_international(f"{icio_folder}/ICIO2021_{years[t]}.csv", temp_recipe, countries, country_sector_names)
        
    return international_flow_series, international_matrix_series

def iter_prop_fit(df, row_margin, col_margin, iters):

    for iter in range(iters):

        next_S1 = df.copy()
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                next_S1.iloc[i,j] = row_margin[i] * df.iloc[i,j] / (df.sum(axis=1, skipna=True)[i] + 1e-20)

        next_S2 = next_S1.copy()
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                next_S2.iloc[i,j] = col_margin[j] * next_S1.iloc[i,j] / (next_S1.sum(axis=0,skipna=True)[j] + 1e-20)

        df = next_S2.copy()
        
    return next_S2

def inner_loop(args):

    i, i_cou, recipe_df, prod_df, expend_df, countries = args

    results = []

    for j, j_cou in enumerate(countries):
        rows = prod_df.index.where(prod_df.index.map(lambda x: i_cou in x and not "TAXSUB" in x)).dropna()
        cols = prod_df.columns.where(prod_df.columns.map(lambda x: j_cou in x and not "TAXSUB" in x)).dropna()

        temp_recipe = recipe_df.copy()
        temp_recipe.index = recipe_df.index.map(lambda x: i_cou + "_" + x)

        marginal_production = prod_df.loc[rows, cols]
        marginal_production = temp_recipe.join(marginal_production).iloc[:,-1]
        marginal_production = marginal_production.fillna(0)

        marginal_expenditure = expend_df.loc[expend_df["cou"].map(lambda x: j_cou in x ), "val"]
        marginal_expenditure = temp_recipe.T.join(marginal_expenditure).iloc[:,-1]
        # assume 0 if missing

        marginal_expenditure = marginal_expenditure.fillna(0)
        
        marginal_expenditure = marginal_expenditure * marginal_production.sum() / marginal_expenditure.sum()

        temp_mat = iter_prop_fit(recipe_df, list(marginal_production), list(marginal_expenditure), 10)

        results.append(temp_mat)

    return results


def iterative_fitting(icio_path, recipe_df, expend_df, countries, country_sector_names):

    icio_df = pd.read_csv(icio_path, index_col=0)

    gfcf_cols = icio_df.columns[icio_df.columns.map(lambda x: "GFCF" in x)]
    gfcf_df = icio_df[gfcf_cols]

    international_matrix = np.zeros((len(country_sector_names), len(country_sector_names)))

    tasks = [(i, i_cou, recipe_df, gfcf_df, expend_df, countries) for i, i_cou in enumerate(countries)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(inner_loop, tasks))


    for i, i_cou in enumerate(countries):
        for j, j_cou in enumerate(countries):
            international_matrix[i*len(recipe_df):(i+1)*len(recipe_df), j*len(recipe_df):(j+1)*len(recipe_df)] = results[i][j].values

    return international_matrix

def gfcf_international(icio_path, recipe_df, gfcf_df, countries, country_sector_names):

    icio_df = pd.read_csv(icio_path, index_col=0)

    prod_cols = icio_df.columns[icio_df.columns.map(lambda x: "GFCF" in x)]
    # remove any non oecd countries since they don't have data on expenditure
    countries = list(prod_cols.map(lambda x: x.split("_")[0])) #[:38]
    country_sector_names = []

    for cou in countries:
        for sec in recipe_df.columns:
            country_sector_names.append(cou + "_" + sec)

    matrix = iterative_fitting(icio_path, recipe_df, gfcf_df, countries)

    matrix = pd.DataFrame(matrix)
    # international_matrix_df2 = pd.DataFrame(np.log(international_matrix2))
    matrix.index = country_sector_names
    matrix.columns = country_sector_names 

    matrix = matrix / matrix.sum()
    matrix = np.log(matrix)

    return matrix


def transform_gfcf(gfcf_path, exchange_path, activity_sector_map): 

    expend_df = pd.read_csv(gfcf_path)
    exchange_rates = pd.read_csv(exchange_path, index_col=0)

    expend_df2 = expend_df[["LOCATION", "ACTIVITY", "TIME", "Value"]][expend_df["TIME"] == 2014]
    expend_df2.reset_index(inplace=True, drop=True)

    for i, row in expend_df2.iterrows():
        expend_df2.iloc[i, 3] = row["Value"] / exchange_rates.loc[row["LOCATION"], "2014"]

    rows = []

    for country in expend_df2["LOCATION"].unique():
        for key, vals in activity_sector_map.items():

            vals = expend_df2.loc[
                (expend_df2["LOCATION"] == country)
                & (expend_df2["ACTIVITY"].map(lambda x: x in vals)), 
                "Value"
            ]

            if len(vals) == 0:
                val = np.nan
            else:
                val = np.sum(vals)

            rows.append({
                "cou": country,
                "sec": key,
                "val": val
            })

    expend_df3 = pd.DataFrame(rows)
    expend_df3.index = expend_df3["sec"]

    return expend_df3

def get_sector_activity_map(path):

    sector_activity = pd.read_csv(path, index_col=0)

    sector_activity_map = {s:[] for s in sector_activity["sector"].unique()}
    for i, row in sector_activity.iterrows():
        sector_activity_map[row["sector"]].append(row["activity"])

    return sector_activity_map
