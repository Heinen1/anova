import patsy
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# load data into dataframe
df = pd.read_csv('gage_rr_data.csv', sep=';', index_col=0)

# melt dataframe to that every row correspond to a response
# rename columns
df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=df.columns)
df_melt['variable'] = 40 * [1] + 40 * [2] + 40 * [3]
df_melt.rename(columns={"index":"part",
                        "variable":"operator",
                        "value":"response"}, inplace=True)

# fit OLS
# Use the R-style formula's
f = 'response ~ C(part) + C(operator) + C(part):C(operator)'
f = 'response ~ (part) + (operator) + (part):(operator)'
model = ols(f, data=df_melt).fit()

y, X = patsy.dmatrices(f, df_melt, return_type='dataframe')
X = sm.add_constant(X)
print(sm.OLS(y, X).fit().summary())

# print summary for OLS model
print(model.summary())

# perform the anova
anova_table = sm.stats.anova_lm(model, typ=2)

# print anova
print(anova_table)

# for a Gage R&R the F-statistic is calculated differently compared
# to standard ANOVA,
