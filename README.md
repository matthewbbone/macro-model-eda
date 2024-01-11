### Macro Model Investment Network

This repository contains the code for creating the international inter-industry investment network for a global macroeconomic model. The intention was to create a matrix, similar to the [ICIO](https://www.oecd.org/sti/ind/inter-country-input-output-tables.htm) international input-output tables but for investment flows. We did this by reimplementing the methodology from [VomLehn & Winberry](https://academic.oup.com/qje/article-abstract/137/1/387/6281043) using US data for 2000-2018 and converting from BEA sectors to the sectors in the OECD ICIO. We then combine this inter-industry investment network with the capital formation data in the ICIO tables to create an approximated international inter-industry investment network.

File Descriptions:
1. investment_matrix/interInvestNetwork.ipynb: This walks through the creation of the network from BEA + ICIO data to the final product
2. investment_matrix/bea_oecd_mapping.py: Utility functions for mapping between BEA and OECD data and evaluating the mapping
3. investment_matrix/international_network.py: Utility functions for cleaning and combining US data with ICIO data
4. investment_matrix/investment_recipe.py: Utility functions for reimplementing [VomLehn & Winberry](https://academic.oup.com/qje/article-abstract/137/1/387/6281043)

References:

Vom Lehn, Christian, and Thomas Winberry. "The investment network, sectoral comovement, and the changing US business cycle." The Quarterly Journal of Economics 137.1 (2022): 387-433.