import concurrent
import numpy as np

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

        temp_mat = iter_prop_fit(recipe_df, list(marginal_production), list(marginal_expenditure), 20)

        results.append(temp_mat)

    return results


def iterative_fitting(recipe_df, prod_df, expend_df, countries):

    country_sectors = len(countries) * len(recipe_df)
    country_sector_names = []

    for cou in countries:
        for sec in recipe_df.columns:
            country_sector_names.append(cou + "_" + sec)

    international_matrix = np.zeros((country_sectors, country_sectors))

    tasks = [(i, i_cou, recipe_df, prod_df, expend_df, countries) for i, i_cou in enumerate(countries)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(inner_loop, tasks))


    for i, i_cou in enumerate(countries):
        for j, j_cou in enumerate(countries):
            international_matrix[i*len(recipe_df):(i+1)*len(recipe_df), j*len(recipe_df):(j+1)*len(recipe_df)] = results[i][j].values

    return international_matrix