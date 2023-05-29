import pandas as pd
import warnings
import re
import numpy as np 
import matplotlib.pyplot as plt

def extract_clean_expenditure(path):

    expend_df = pd.read_excel(path, sheet_name='Datasets', index_col=0)
    expend_df = expend_df.apply(lambda x: [i.replace(",","") if type(i) is str else i for i in x])
    expend_df = expend_df.astype(float)
    # gets part of index that identifies industry
    expend_df["industry"] = expend_df.index.map(lambda x: x[3:7])
    # gets part of index that indentifies asset
    expend_df["asset"] = expend_df.index.map(lambda x: x[8:-2])
    # this are total columns
    expend_df = expend_df[expend_df["asset"].map(lambda x: x not in ["EQ00", "ST00"])]

    return expend_df

def read_bridge(path, year):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        equip_df = pd.read_excel(path, sheet_name=year, header=4)
        equip_df.columns = ["nipa", "peq_name", "code", "description","prod_val","trans_costs", "wholesale", "retail", "buy_val", "year"]
        equip_df = equip_df[~(equip_df["code"] == "Used")]
        equip_df["code"] = equip_df["code"].map(lambda x: re.sub(r"[a-zA-Z]", "", str(x)))
        equip_df["code"] = equip_df["code"].map(lambda x: x + "0" if len(x) == 3 else x)

        equip_df["code"] = equip_df["code"].map(lambda x: x.replace("1","M") if x == "3361" else x)
        equip_df["code"] = equip_df["code"].map(lambda x: x.replace("4","O") if x == "3364" else x)
        equip_df["code"] = equip_df["code"].map(lambda x: x.replace("0","T") if x == "3130" else x)

        equip_df = equip_df[~(equip_df["code"] == "Used")]
        equip_df = equip_df[["nipa", "code", "prod_val", "trans_costs","wholesale","retail"]]
        
    return equip_df

def get_nipa_split(path, years):

    # create map for getting asset production weights
    nipa_split = {}
    for y in years:

        equip_df = read_bridge(path, y)

        yearly = {}
        for n in equip_df["nipa"].unique():
            # default split includes marginal costs
            temp = {
                "4810": 0,
                "4820": 0,
                "4830": 0,
                "4840": 0,
                "4850": 0,
                "487S": 0,
                "4930": 0,
                "4200": 0,
                "44RT": 0
            }
            for i, row in equip_df[equip_df["nipa"] == n].iterrows():

                # marginal values
                tot_transport = equip_df[equip_df["nipa"] == n]["trans_costs"].sum()
                tot_wholesale = equip_df[equip_df["nipa"] == n]["wholesale"].sum()
                tot_retail = equip_df[equip_df["nipa"] == n]["retail"].sum()

                # produced value
                tot_prod = equip_df[equip_df["nipa"] == n]["prod_val"].sum()

                # weighted production
                total = tot_transport + tot_wholesale + tot_retail + tot_prod
                temp[row["code"]] = row["prod_val"] / total

                # add transportation margins
                for t in ["4810", "4820", "4830", "4840", "4850", "487S", "4930"]:
                    temp[t] = temp[t] + row["trans_costs"] / total / 7
                
                # add wholesale margins
                temp["4200"] = temp["4200"] + row["wholesale"] / total

                # add retail margins
                temp["44RT"] = temp["44RT"] + row["retail"] / total

            yearly[n] = temp
        nipa_split[y] = yearly

    return nipa_split

def get_equip_expenditure(expend_df, asset_nipa, years):

    # get just equipment expenditures
    equip_expend_df = expend_df[expend_df["asset"].map(lambda x: x[0] == "E")].copy()
    # remove software products
    equip_expend_df = equip_expend_df[equip_expend_df["asset"].map(lambda x: not x in ["ENS1","ENS2","ENS3"])]
    equip_expend_df["nipa"] = equip_expend_df["asset"].map(lambda x: asset_nipa[x])

    # industry by nipa asset category
    equip_expend_df2 = equip_expend_df[years + ["industry", "nipa"]]
    equip_expend_df2 = equip_expend_df2.groupby(["industry", "nipa"]).agg("sum")

    return equip_expend_df2

def create_nonres_equipment_network(equip_expend_df, nipa_split, sectors, years):

    n_sec = len(sectors)
    equip_matrix = np.zeros((len(years), n_sec, n_sec))

    for i in range(n_sec):
        for j in range(n_sec):
            for t in range(len(years)):
                temp = 0
                for a in nipa_split[years[t]].keys():
                    try:
                        temp = temp + nipa_split[years[t]][a][sectors[i]] * equip_expend_df.loc[(sectors[j],a), years[t]]
                    except:
                        continue
                    
                equip_matrix[t,i,j] = temp

    return equip_matrix

def create_nonres_structure_network(expend_df, years, sectors): 

    n_secs = len(sectors)

    structure_df = expend_df[expend_df["asset"].map(lambda x: x[0] == "S")].copy()
    structure_df["asset"] = structure_df["asset"].map(lambda x: "Mining" if x in ["SM02", "SM01"] else "Other")
    structure_df = structure_df.groupby(["industry", "asset"]).agg("sum")
    structure_matrix = np.zeros((len(years), n_secs, n_secs))

    for i in range(n_secs):
        for j in range(n_secs):
            for t in range(len(years)):

                if sectors[i] == "2130":
                    structure_matrix[t,i,j] = structure_df.loc[(sectors[j],"Mining"), years[t]]
                elif sectors[i] == "2300":
                    structure_matrix[t,i,j] = structure_df.loc[(sectors[j],"Other"), years[t]]

    return structure_matrix

def get_ipp_expenditure(expend_df, ipp_map, years, sectors):

    n_secs = len(sectors)

    ipp_df = expend_df[expend_df["asset"].map(lambda x: x in ipp_map.keys())].copy()
    ipp_df["asset"] = ipp_df["asset"].map(lambda x: ipp_map[x])
    ipp_df = ipp_df.groupby(["industry", "asset"]).agg("sum")
    
    return ipp_df

def get_software_margin_weights(path, ipp_df, years, sectors):

    n_secs = len(sectors)

    ipp_matrix = np.zeros((len(years), n_secs, n_secs))
    software_expenditures = ipp_df[ipp_df.index.map(lambda x: x[0] == "5110")].sum(axis=0, skipna=True).loc[["2007", "2012"]]

    software_cols = ["541511"]

    wholesale_rows = [423100,423400,423600,423800,"423A00",424200,424400,424700,"424A00",425000,"4200ID"]
    retail_rows = [441000, 445000, 452000, 444000, 446000, 447000, 448000, 454000, "4B0000"]
    air_transport_rows = [481000]
    rail_transport_rows = [482000]
    water_transport_rows = [483000]
    truck_transport_rows = [484000]
    ground_transport_rows = [485000]
    pipe_transport_rows = [486000]
    other_transport_rows = ["48A000", 492000]
    storage_rows = [493000]

    ["4810", "4820", "4830", "4840", "4850", "487S", "4930"]

    wholesale_margin = 0
    retail_margin = 0
    air_transport_margin = 0
    rail_transport_margin = 0
    water_transport_margin = 0
    truck_transport_margin = 0
    ground_transport_margin = 0
    pipe_transport_margin = 0
    other_transport_margin = 0
    storage_margin = 0

    for y in ["2007", "2012"]:
        
        df = pd.read_excel(path, sheet_name=y, header=5, index_col=0)

        wholesale_margin += df.loc[wholesale_rows, software_cols].sum().sum() / software_expenditures.loc[y] / 2
        retail_margin += df.loc[retail_rows, software_cols].sum().sum() / software_expenditures.loc[y] / 2
        air_transport_margin += df.loc[air_transport_rows, software_cols].sum().sum() / software_expenditures.loc[y] / 2
        rail_transport_margin += df.loc[rail_transport_rows, software_cols].sum().sum() / software_expenditures.loc[y] / 2
        water_transport_margin += df.loc[water_transport_rows, software_cols].sum().sum() / software_expenditures.loc[y] / 2
        truck_transport_margin += df.loc[truck_transport_rows, software_cols].sum().sum() / software_expenditures.loc[y] / 2
        ground_transport_margin += df.loc[ground_transport_rows, software_cols].sum().sum() / software_expenditures.loc[y] / 2
        pipe_transport_margin += df.loc[pipe_transport_rows, software_cols].sum().sum() / software_expenditures.loc[y] / 2
        other_transport_margin += df.loc[other_transport_rows, software_cols].sum().sum() / software_expenditures.loc[y] / 2
        storage_margin += df.loc[storage_rows, software_cols].sum().sum() / software_expenditures.loc[y] / 2

    software_margin_weight = wholesale_margin + retail_margin + air_transport_margin + \
                            water_transport_margin + truck_transport_margin + ground_transport_margin + \
                            pipe_transport_margin + other_transport_margin + storage_margin
    
    return {
        "wholesale": wholesale_margin,
        "retail": retail_margin,
        "air_transport": air_transport_margin,
        "rail_transport": rail_transport_margin,
        "water_transport": water_transport_margin,
        "truck_transport": truck_transport_margin,
        "ground_transport": ground_transport_margin,
        "pipe_transport": pipe_transport_margin,
        "other_transport": other_transport_margin,
        "storage": storage_margin,
        "software": software_margin_weight
    }

def get_artistic_split(use_path, make_path):

    artistic_split = 0
    artistic = ["711A00"]
    for y in ["2007", "2012"]:

        make_df = pd.read_excel(make_path, sheet_name=y, header=5, index_col=0)
        use_df = pd.read_excel(use_path, sheet_name=y, header=5, index_col=0)

        total_production = make_df.loc[artistic, artistic].sum().sum()
        total_expenditure = use_df.loc[artistic[0]][1:].sum(skipna=True)

        artistic_split += total_production / total_expenditure / 2

    return artistic_split

def create_ipp_network(ipp_df, margins, artistic_split, years, sectors):

    n_secs = len(sectors)

    ipp_matrix = np.zeros((len(years), n_secs, n_secs))

    for i in range(n_secs):
        for j in range(n_secs):
            for t in range(len(years)):
                try:
                    # split marginal payments
                    if sectors[i] == "5110":
                        # this will result in a keyerror when sector j doesn't consume any software from 5110
                        total_software_expenditure = ipp_df.loc[(sectors[j],sectors[i]), years[t]]

                        # software from publishing industry
                        ipp_matrix[t,i,j] += (1 - margins["software"]) * total_software_expenditure
                        # wholesale margins
                        ipp_matrix[t,sectors.index("4200"),j] = margins["wholesale"] * total_software_expenditure
                        # retail margins
                        ipp_matrix[t,sectors.index("44RT"),j] = margins["retail"] * total_software_expenditure
                        # air transport margins
                        ipp_matrix[t,sectors.index("4810"),j] = margins["air_transport"] * total_software_expenditure
                        # air transport margins
                        ipp_matrix[t,sectors.index("4820"),j] = margins["rail_transport"] * total_software_expenditure
                        # water transport margins
                        ipp_matrix[t,sectors.index("4830"),j] = margins["water_transport"] * total_software_expenditure
                        # truck transport margins
                        ipp_matrix[t,sectors.index("4840"),j] = margins["truck_transport"] * total_software_expenditure
                        # ground transport margins
                        ipp_matrix[t,sectors.index("4850"),j] = margins["ground_transport"] * total_software_expenditure
                        # pipe transport margins
                        ipp_matrix[t,sectors.index("4860"),j] = margins["pipe_transport"] * total_software_expenditure
                        # other transport margins
                        ipp_matrix[t,sectors.index("487S"),j] = margins["other_transport"] * total_software_expenditure
                        # storage margins
                        ipp_matrix[t,sectors.index("4930"),j] = margins["storage"] * total_software_expenditure
                    elif sectors[i] == "711A":
                        ipp_matrix[t,i,j] = artistic_split * ipp_df.loc[(sectors[j],sectors[i]), years[t]]
                        ipp_matrix[t,sectors.index("5110"),j] += (1 - artistic_split) * ipp_df.loc[(sectors[j],sectors[i]), years[t]]
                    else:
                        ipp_matrix[t,i,j] = ipp_df.loc[(sectors[j],sectors[i]), years[t]]
                except KeyError as e:
                    continue

    return ipp_matrix

def get_resid_production_split(io_path, bridge_path, years):

    construction_rows = ["23"]
    real_estate_rows = ["HS", "ORE"]
    rental_rows = ["532RL"]
    fed_rows = ["521CI"]
    investments_rows = ["523"]
    insurance_rows = ["524"]
    funds_rows = ["525"]
    legal_rows = ["5411"]
    computer_rows = ["5415"]
    misc_prof_rows = ["5412OP"]

    # assumed % of margin payments that are due to new production
    new_margin_scale = .132

    construction_output = {}
    real_estate_output = {}
    rental_output = {}
    fed_output = {}
    investments_output = {}
    insurance_output = {}
    funds_output = {}
    legal_output = {}
    computer_output = {}
    misc_prof_output = {}

    resid_equip_output = {}
    warnings.simplefilter("ignore")
    for y in years: 

        housing_data = pd.read_excel(io_path, sheet_name=y, header=5, index_col=0).loc[:,"HS"]
        housing_data = housing_data.map(lambda x: 0 if x in ["...", np.nan] else x)
        
        construction_output[y] = housing_data.loc[construction_rows].sum()
        real_estate_output[y] = housing_data.loc[real_estate_rows].sum() * new_margin_scale
        rental_output[y] = housing_data.loc[rental_rows].sum() * new_margin_scale
        fed_output[y] = housing_data.loc[fed_rows].sum() * new_margin_scale
        investments_output[y] = housing_data.loc[investments_rows].sum() * new_margin_scale
        insurance_output[y] = housing_data.loc[insurance_rows].sum() * new_margin_scale
        funds_output[y] = housing_data.loc[funds_rows].sum() * new_margin_scale
        legal_output[y] = housing_data.loc[legal_rows].sum() * new_margin_scale
        computer_output[y] = housing_data.loc[computer_rows].sum() * new_margin_scale
        misc_prof_output[y] = housing_data.loc[misc_prof_rows].sum() * new_margin_scale
        
        bridge_df = pd.read_excel(bridge_path, sheet_name=y, header=4)
        bridge_df.columns = ["nipa", "peq_name", "code", "description","prod_val","trans_costs", "wholesale", "retail", "buy_val", "year"]
        resid_equip_output[y] = bridge_df[bridge_df["nipa"] == 34]['buy_val'].sum()

    return {
        "construction": construction_output,
        "real_estate": real_estate_output,
        "rental": rental_output,
        "fed": fed_output,
        "investments": investments_output,
        "insurance": insurance_output,
        "funds": funds_output,
        "legal": legal_output,
        "computer": computer_output,
        "misc_prof": misc_prof_output,
        "resid_equip": resid_equip_output
    }

def create_resid_network(outputs, nipa_split, years, sectors):

    n_secs = len(sectors)

    resid_matrix = np.zeros((len(years), n_secs, n_secs))

    real_estate = list(sectors).index("5310")

    for t in range(len(years)):

        # 34 is the nipa code for residential equipment
        resid_equip_split = nipa_split[years[t]][34].keys()
        
        for k in resid_equip_split:
            resid_matrix[t,list(sectors).index(k),real_estate] = outputs["resid_equip"][years[t]] * nipa_split[years[t]][34][k]

        resid_matrix[t, list(sectors).index("2300"), real_estate] = outputs["construction"][years[t]]
        resid_matrix[t, list(sectors).index("5310"), real_estate] = outputs["real_estate"][years[t]]
        resid_matrix[t, list(sectors).index("5320"), real_estate] = outputs["rental"][years[t]]
        resid_matrix[t, list(sectors).index("5210"), real_estate] = outputs["fed"][years[t]]
        resid_matrix[t, list(sectors).index("5230"), real_estate] = outputs["investments"][years[t]]
        resid_matrix[t, list(sectors).index("5240"), real_estate] = outputs["insurance"][years[t]]
        resid_matrix[t, list(sectors).index("5250"), real_estate] = outputs["funds"][years[t]]
        resid_matrix[t, list(sectors).index("5411"), real_estate] = outputs["legal"][years[t]]
        resid_matrix[t, list(sectors).index("5415"), real_estate] = outputs["computer"][years[t]]
        resid_matrix[t, list(sectors).index("5412"), real_estate] = outputs["misc_prof"][years[t]]

    return resid_matrix


def get_total_expend_weights(expend_df, years):

    tot_exp_df = expend_df[:-1].groupby("industry").agg("sum")

    bea_total_exp = {}
    for y in years: 
        bea_total_exp[y] = {}
        for sec, val in tot_exp_df[y].items():
            bea_total_exp[y][sec] = val

    return bea_total_exp

def get_chemical_split(icio_folder, years):

    chemical_splits = {}
    for y in years:
        icio_df = pd.read_csv(f"{icio_folder}/ICIO2021_{y}.csv", index_col=0)
        usa_cols = icio_df.columns[icio_df.columns.map(lambda x: x.split("_")[0] == "USA")]
        usa_chems = icio_df.loc[["USA_20", "USA_21"], usa_cols].sum(axis=1)
        chem_ratio = usa_chems[0] / usa_chems.sum()
        chemical_splits[y] = chem_ratio

    return chemical_splits

def create_oecd_matrix(bea_matrix, oecd_bea, bea_total_exp, chemical_split):

    ind = []
    for key, val in oecd_bea.items():
        ind = ind + val


    industries = list(bea_matrix.columns)
    sectors = list(oecd_bea.keys())

    # row summing matrix
    num_splits = {}
    R = np.zeros((len(sectors), len(industries)))
    for sec_i, sec in enumerate(sectors):
        for ind_i, ind in enumerate(industries):
            splits = np.sum([1 for k in oecd_bea.keys() if ind in oecd_bea[k]])
            R[sec_i,ind_i] = 1 / splits if ind in oecd_bea[sec] else 0

    # column weighted averaging matrix
    C = np.zeros((len(industries), len(sectors)))
    for ind_i, ind in enumerate(industries):
        for sec_i, sec in enumerate(sectors):
            C[ind_i, sec_i] = bea_total_exp[ind] if ind in oecd_bea[sec] else 0

    # C = normalize(C, axis=0, norm='l1')

    res = np.dot(np.dot(R, np.array(bea_matrix)), C)

    # column-wise normalization
    # res = normalize(res, axis = 0, norm='l1')

    res = res / res.sum()

    res = pd.DataFrame(res)
    res.index = sectors
    res.columns = sectors

    # fix chemical split
    res.loc[:, "20"] = res.loc[:, "20"] * chemical_split
    res.loc[:, "21"] = res.loc[:, "21"] * (1 - chemical_split)

    res = res[np.sort(sectors)]
    res.sort_index(inplace=True)

    return res

def create_oecd_matrix_series(bea_matrix, oecd_bea, bea_total_exp, chemical_split, years, sectors):

    recipe_series = np.zeros((len(years), 41, 41))
    for t in range(len(years)):
        temp_bea = pd.DataFrame(bea_matrix[t,:,:])
        temp_bea.columns = sectors
        temp_bea.index = sectors
        recipe_series[t,:,:] = create_oecd_matrix(temp_bea, oecd_bea, bea_total_exp[years[t]], chemical_split[years[t]])

    return recipe_series

def plot_top_flows(norm_bea, years, sectors):

    for i in range(len(sectors)):
        for j in range(len(sectors)):
            series = norm_bea[:, i, j]
            if series[-1] > series[0] + .01:
                plt.plot(years,series, label=f"{sectors[i]} to {sectors[j]}")

    plt.title("> 1% increase of total investment flows")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()

def plot_bot_flows(norm_bea, years, sectors):

    for i in range(len(sectors)):
        for j in range(len(sectors)):
            series = norm_bea[:, i, j]
            if series[-1] < series[0] - .01:
                plt.plot(years,series, label=f"{sectors[i]} to {sectors[j]}")

    plt.title("> 1% decrease of total investment flows")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()

def plot_top_prod(norm_bea, years, sectors):

    for i in range(len(sectors)):
        series = norm_bea[:,i,:].sum(axis = 1)
        if series[-1] > series[0] + .01:
            plt.plot(years, series, label = sectors[i])

    plt.title("> 1% increase of total production")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()

def plot_bot_prod(norm_bea, years, sectors):

    for i in range(len(sectors)):
        series = norm_bea[:,i,:].sum(axis = 1)
        if series[-1] < series[0] - .01:
            plt.plot(years, series, label = sectors[i])

    plt.title("> 1% decrease of total production")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()

def plot_top_expend(norm_bea, years, sectors):

    for i in range(len(sectors)):
        series = norm_bea[:,:,i].sum(axis = 1)
        if series[-1] > series[0] + .01:
            plt.plot(years, series, label = sectors[i])

    plt.title("> 1% increase of total expenditure")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()

def plot_bot_expend(norm_bea, years, sectors):

    for i in range(len(sectors)):
        series = norm_bea[:,:,i].sum(axis = 1)
        if series[-1] < series[0] - .01:
            plt.plot(years, series, label = sectors[i])

    plt.title("> 1% decrease of total expenditure")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()

