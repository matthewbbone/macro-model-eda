import numpy as np
import pandas as pd

def extract_bea_naics_map(path): 

    readme = pd.read_excel(path, header=14, converters={'BEA CODE': str, '2012 NAICS Codes': str})
    readme = readme[['BEA CODE', '2012 NAICS Codes']][readme['BEA CODE'].map(lambda x: not (type(x) is float or x == '--------') )]
    bea_naics_map = {}
    for i, row in readme.iterrows():
        naics = row[1].split(',')
        for n in naics:
            if '-' in naics[0]:
                temp = naics[0].split('-')
                naics = [str(x) for x in np.arange(int(temp[0]),int(temp[0][:-1] + temp[1]) + 1)]
        bea_naics_map[row[0]] = [n.strip() for n in naics]

    return bea_naics_map

def format_crosswalk(df, bea_naics):

    df["isic"] = df["oecd_code"].map(lambda x: x.split("T"))
    df["naics"] = df["bea_code"].map(lambda x: [str(i) for i in bea_naics[x]] if not type(x) is float else np.nan)

    return df

def crosswalk_map(df):

    oecd_codes = df["oecd_code"].unique()
    oecd_bea = {o: [] for o in oecd_codes}
    for i, row in df.iterrows():
        oecd_bea[row["oecd_code"]].append(row["bea_code"])

    return oecd_bea

def score_match(isic, naics, map_df):

    if np.nan in [isic, naics]: return 0

    naics_digits = len(naics[0])

    isic_mask = map_df["ISIC4Code"].map(lambda x: x[0:2] in isic)
    naics_mask = map_df["NAICS2012Code"].map(lambda x: type(x) is str and x[0:naics_digits] in naics)

    union = map_df[naics_mask | isic_mask]

    intersection = map_df[ naics_mask & isic_mask ]

    return len(intersection) / len(union)

def score_crosswalk(crosswalk_df, map_df):
    scores = []
    for i, row in crosswalk_df.iterrows():
        score = score_match(row["isic"], row["naics"], map_df)
        scores.append(score)
    
    return scores