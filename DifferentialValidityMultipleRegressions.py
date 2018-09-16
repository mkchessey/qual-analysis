import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import statsmodels.api as sm
from scipy.stats import ttest_ind, fisher_exact, norm
from math import sqrt

def regress_plot(y,x,xlim,ylim,xlab,ylab):
    model = sm.OLS(y, sm.add_constant(x)).fit()
    print(model.summary())
    print('\n\n')
    plt.figure()
    plt.scatter(x[x.columns[0]], y, edgecolors='black', label='Male')
    plt.scatter(x[x.columns[0]][x.Gender == 1], y[x.Gender == 1], edgecolors='black', label='Female')
    #plt.plot(x[x.columns[0]], model.params[0] + x[x.columns[0]]*model.params[1], label='Male fit')
    #plt.plot(x[x.columns[0]],model.params[0] + x[x.columns[0]] * model.params[1] + model.params[2],label='Female fit')
    #plt.plot(x[x.columns[0]],model.params[0] + x[x.columns[0]]*(model.params[1]+model.params[14]) + model.params[2], label='Female fit')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='best')
    plt.show()

def regress_plot_cohort(y,x,xlab,ylab):
    model = sm.OLS(y, sm.add_constant(x)).fit()
    print(model.summary())
    print('\n\n')
    plt.figure()
    plt.scatter(x, y, edgecolors='black')
    plt.plot(x, model.params[0] + x*model.params[1])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()

def regress_plot_eth(y, x, xlim, ylim, xlab, ylab):
    model = sm.OLS(y, sm.add_constant(x)).fit()
    print(model.summary())
    print('\n\n')
    plt.figure()
    plt.scatter(x[x.columns[0]], y, edgecolors='black', label='White')
    plt.scatter(x[x.columns[0]][x.Ethnicity == 1], y[x.Ethnicity == 1], edgecolors='black', label='Non-white')
    plt.plot(x[x.columns[0]], model.params[0] + x[x.columns[0]] * model.params[1], label='White fit')
    plt.plot(x[x.columns[0]], model.params[0] + x[x.columns[0]] * model.params[1] + model.params[2],label='Non-white fit')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='best')
    plt.show()

def regress_plot_cit(y, x, xlim, ylim, xlab, ylab):
    model = sm.OLS(y, sm.add_constant(x)).fit()
    print(model.summary())
    print('\n\n')
    plt.figure()
    plt.scatter(x[x.columns[0]], y, edgecolors='black', label='U.S.')
    plt.scatter(x[x.columns[0]][x.Citizen == 1], y[x.Citizen == 1], edgecolors='black', label='Non-U.S.')
    plt.plot(x[x.columns[0]], model.params[0] + x[x.columns[0]] * model.params[1], label='U.S. fit')
    plt.plot(x[x.columns[0]], model.params[0] + x[x.columns[0]] * model.params[1] + model.params[2],label='Non-U.S. fit')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='best')
    plt.show()

def regress_plot_deg(y, x, xlim, ylim, xlab, ylab):
    model = sm.OLS(y, sm.add_constant(x)).fit()
    print(model.summary())
    print('\n\n')
    plt.figure()
    plt.scatter(x[x.columns[0]], y, edgecolors='black', label='Earned PHD')
    plt.scatter(x[x.columns[0]][x['Degree Status'] == 1], y[x['Degree Status'] == 1], edgecolors='black', label='No PHD')
    plt.plot(x[x.columns[0]], model.params[0] + x[x.columns[0]] * model.params[1], label='PHD fit')
    plt.plot(x[x.columns[0]], model.params[0] + x[x.columns[0]] * model.params[1] + model.params[2],label='No PHD fit')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='best')
    plt.show()

# Read the data file
df = pd.read_excel(open('FinalGradFile20170429.xls','rb'), sheetname='Records')

# Remove master's degrees
df = df[df['Degree']=='PHD']
# ms_mask = df.apply(lambda x: x.Degree == 'MS' and ((df['Student Number'] == x['Student Number']) & (df.Degree == 'PHD')).any(), axis = 1)
# df = df[~ms_mask]

# Remove white space from before and after strings in data (Citizen, Degree, Degree Status, Gender, Ethnicity, Prelim Letter)
df.Citizen = df.Citizen.str.strip()
df.Degree = df.Degree.str.strip()
df['Degree Status'] = df['Degree Status'].str.strip()
df.Gender = df.Gender.str.strip()
df.Ethnicity = df.Ethnicity.str.strip()
df['Prelim Letter'] = df['Prelim Letter'].str.strip()

# Check odds ratio for male and female having 'Degree Status' == DA for PHD
# FYes = len(df[(df.Gender.str.strip() == 'F') & (df['Degree Status'].str.strip()=='DA')]['Degree Status'])
# FNo = len(df[(df.Gender.str.strip() == 'F') & ~(df['Degree Status'].str.strip()=='DA')]['Degree Status'])
# MYes = len(df[(df.Gender.str.strip() == 'M') & (df['Degree Status'].str.strip()=='DA')]['Degree Status'])
# MNo = len(df[(df.Gender.str.strip() == 'M') & ~(df['Degree Status'].str.strip()=='DA')]['Degree Status'])
# ContingencyTable = np.array([[FYes,FNo], [MYes,MNo]])
# print(fisher_exact(ContingencyTable,alternative='less'))
# print(fisher_exact(ContingencyTable,alternative='greater'))
# print(fisher_exact(ContingencyTable))
# # print(ContingencyTable)
# print(FYes, FNo, MYes, MNo)
# OddsRatio = (FYes/FNo) / (MYes/MNo)
# OR_SE = sqrt(1/FYes + 1/FNo + 1/MYes + 1/MNo)
# ConfidenceInterval = [OddsRatio - 1.96*OR_SE, OddsRatio + 1.96*OR_SE]
# print(OddsRatio, OR_SE, ConfidenceInterval)

# # Check averages for male and for female on 'Qtrs from admit to prelim pass'
# data = df[df['Qtrs from admit to prelim pass'].apply(np.isreal) & ~df['Qtrs from admit to prelim pass'].isnull()]# & (~(df['Qtrs from admit to prelim pass'] == 0))]
# #data = data[data['Qtrs from admit to prelim pass'] <= 12]
# Fave = data[data.Gender.str.strip() == 'F']['Qtrs from admit to prelim pass']
# Mave = data[data.Gender.str.strip() == 'M']['Qtrs from admit to prelim pass']
# print(Fave.mean(), Mave.mean())
# print(ttest_ind(Fave, Mave, equal_var=False))
# plt.figure()
# data[data.Gender.str.strip() == 'M']['Qtrs from admit to prelim pass'].hist()
# data[data.Gender.str.strip() == 'F']['Qtrs from admit to prelim pass'].hist()
# plt.show()

# Data summary information
print()

# Filter rows by existence of prelim data
#   1) Get rid of rows with no prelim data
#   2) Get rid of rows with all 0's on first questions, and then second 10 questions
#   3) Get rid of rows that passed the prelim before their first year 'Qtrs from admit to prelim pass'==0
cols = ['Prob%d' % (i + 1) for i in range(20)]

df = df.dropna(how = 'all', subset = cols)
df = df[~(df[cols[:10]] == 0).all(axis = 1)]
df = df[~(df[cols[10:]] == 0).all(axis = 1)]
# df = df[~(df['Qtrs from admit to prelim pass']==0)]

# Create extra column to store total Prelim score by adding up individual item scores
df['Total Score'] = df[cols].sum(axis = 1)

# Convert GRE scores (SUB, A, Q, V) from percentiles to zscores
# Percentiles are ordinal scale but we need interval scale to run a regression that makes sense
df.loc[df['GRE SUB']==0.0,'GRE SUB'] = 0.05
df['GRE V'] = norm.ppf(df['GRE V'])
df['GRE Q'] = norm.ppf(df['GRE Q'])
df['GRE A'] = norm.ppf(df['GRE A'])
df['GRE SUB'] = norm.ppf(df['GRE SUB'])

# Get rid of string values in Qtrs from admit to prelim pass
#df.loc[isinstance(df['Qtrs from admit to prelim pass'], str),'Qtrs from admit to prelim pass'] = ''

# Create extra columns to calculate time DIFFERENCE between prelim and advance to candidacy and between prelim and Degree
df['Time diff AC prelim'] = df['Qtrs from Admit to AC'].astype(float) - df['Qtrs from admit to prelim pass'].astype(float)
df['Time diff Degree prelim'] = df['Qtrs from Admit to Degree'].astype(float) - df['Qtrs from admit to prelim pass'].astype(float)
df['Time diff Degree AC'] = df['Qtrs from Admit to Degree'].astype(float) - df['Qtrs from Admit to AC'].astype(float)
#print(df['Time diff Degree prelim'])

# Separate prelim by form (year of administration) using code 'Letter of 1st prelim'
year_codes = sorted(set(df['Prelim Letter']))

# Create DUMMY VARIABLES for forms of exam. Need one fewer dummy variables than number of exams
df['formB'] = (df['Prelim Letter']=='B').astype(int)
df['formC'] = (df['Prelim Letter']=='C').astype(int)
df['formD'] = (df['Prelim Letter']=='D').astype(int)
df['formE'] = (df['Prelim Letter']=='E').astype(int)
df['formF'] = (df['Prelim Letter']=='F').astype(int)
df['formG'] = (df['Prelim Letter']=='G').astype(int)
df['formH'] = (df['Prelim Letter']=='H').astype(int)
df['formI'] = (df['Prelim Letter']=='I').astype(int)
df['formJ'] = (df['Prelim Letter']=='J').astype(int)
df['formK'] = (df['Prelim Letter']=='K').astype(int)
df['formM'] = (df['Prelim Letter']=='M').astype(int)

# for col in ['B', 'C', 'D']:
#     df['form'+ col] = (df['Prelim Letter'] == col).astype(int)

# Calculate z scores within year for Preliminary Exam Total Scores
year_mean = {year: df[df['Prelim Letter'] == year]['Total Score'].mean() for year in year_codes}
year_std = {year: df[df['Prelim Letter'] == year]['Total Score'].std(ddof=0) for year in year_codes}

df['zscore'] = df.apply((lambda x: (x['Total Score'] - year_mean[x['Prelim Letter']]) / year_std[x['Prelim Letter']]), axis = 1)

# Make plots to see if prelim raw score has to do with characteristics of the cohort
# create new dataframe called cohortdf
# have one row in cohortdf for each year ('Prelim Letter') of the exam
# have one column in cohortdf for each variable's average value per year:
#   'Total Score', 'Und GPA', 'Cum GPA', 'GRE SUB', 'GRE Q', 'GRE V', 'GRE A'
# grouped = df.groupby(by = 'Prelim Letter')
# cohortdf = grouped.mean()[['Total Score', 'Und GPA', 'Cum GPA', 'GRE SUB', 'GRE Q', 'GRE V', 'GRE A']]
# model = regress_plot_cohort(cohortdf['Total Score'], cohortdf['Und GPA'], 'Average Undergraduate GPA','Average Prelim raw score')
# model = regress_plot_cohort(cohortdf['Total Score'], cohortdf['Cum GPA'], 'Average Graduate GPA','Average Prelim raw score')
# model = regress_plot_cohort(cohortdf['Total Score'], cohortdf['GRE SUB'], 'Average Physics GRE z-score','Average Prelim raw score')
# model = regress_plot_cohort(cohortdf['Total Score'], cohortdf['GRE Q'], 'Average GRE quantitative z-score','Average Prelim raw score')
# model = regress_plot_cohort(cohortdf['Total Score'], cohortdf['GRE V'], 'Average GRE verbal z-score','Average Prelim raw score')
# model = regress_plot_cohort(cohortdf['Total Score'], cohortdf['GRE A'], 'Average GRE writing z-score','Average Prelim raw score')


# Correlation table
# dfcorr = df[['Cum GPA','Und GPA','GRE SUB','GRE V','GRE Q','GRE A','zscore','Time diff AC prelim','Time diff Degree prelim','Time diff Degree AC']]
# dfcorrF = df[df.Gender=='F'][['Cum GPA','Und GPA','GRE SUB','GRE V','GRE Q','GRE A','zscore','Time diff AC prelim','Time diff Degree prelim','Time diff Degree AC']]
# dfcorrM = df[df.Gender=='M'][['Cum GPA','Und GPA','GRE SUB','GRE V','GRE Q','GRE A','zscore','Time diff AC prelim','Time diff Degree prelim','Time diff Degree AC']]
# print(dfcorr.corr(method='pearson'))
#print(dfcorrF.corr(method='pearson'))
#print(dfcorrM.corr(method='pearson'))

# Run a multiple regression for each prelim predicting 'Qtrs from Admit to AC' given Total Score and Gender
# for year in year_codes:
#     print('Year:', year)
#
#     regression_mask = (df.Gender != 'U') & (~(df['Qtrs from Admit to Degree'].isnull())) & (df['Letter of 1st prelim'] == year)
#
#     y = df[regression_mask]['Qtrs from Admit to Degree']
#     x = df[regression_mask][['Total Score', 'Gender']]
#     x.Gender = (x.Gender.str.strip() == 'F').astype(int)
#     x['Interaction'] = x['Total Score'] * x['Gender']
#
#     model = sm.OLS(y, sm.add_constant(x)).fit()
#     print(model.params)
#     plt.figure()
#     plt.scatter(x['Total Score'],y)
#     plt.plot(x['Total Score'],model.params[0] + x['Total Score']*model.params[1])
#     plt.plot(x['Total Score'],model.params[0] + x['Total Score']*model.params[1] + x['Total Score']*model.params[3])
#     plt.xlabel('Preliminary Exam Score')
#     plt.ylabel('Quarters from Admit to Complete Degree')
#     plt.show()
#     print(model.summary())
#     print('\n\n')


# # Run a multiple regression for each prelim predicting Total Score given Cum GPA and Gender
# fig, axs = plt.subplots(4,3,sharex=True,sharey=True)
# axs = axs.ravel()
# i=0
# for year in year_codes:
#     print('Year:', year)
#
#     regression_mask = (df.Gender != 'U') & (~(df['GRE SUB'].isnull())) & (df['Prelim Letter'] == year)
#
#     y = df[regression_mask]['Total Score']
#     x = df[regression_mask][['GRE SUB', 'Gender']]
#     x.Gender = (x.Gender.str.strip() == 'F').astype(int)
#     #x['Interaction'] = x['Cum GPA'] * x['Gender']
#
#     model = sm.OLS(y, sm.add_constant(x)).fit()
#     print(model.params)
#     axs[i].scatter(x['GRE SUB'],y, edgecolors='black',label='Male')
#     axs[i].scatter(x['GRE SUB'][x.Gender==1],y[x.Gender==1],edgecolors='black',label='Female')
#     axs[i].plot(x['GRE SUB'],model.params[0] + x['GRE SUB']*model.params[1],label='Best fit')
#     #plt.plot(x['Cum GPA'],model.params[0] + x['Cum GPA']*model.params[1] + x['Cum GPA']*model.params[3])
#     #axs[i].xlim([2.8,4.1])
#     #axs[i].ylim([0,1000])
#     #plt.xlabel('Cumulative graduate GPA')
#     axs[i].set_title('%s' % year)
#     #axs[i].legend(loc='best')
#     #plt.show()
#     print(model.summary())
#     print('\n\n')
#     i = i + 1
# plt.show()

# # For each prelim show some descriptive information
# print('Prelim Total Score Descriptive Information')
# for year in year_codes:
#     data = df[df['Prelim Letter'] == year]
#     print('Prelim:', year, ', N:', len(data), ', N Female:', len(data[data.Gender.str.strip() == 'F']),', N Male:',
#           len(data[data.Gender.str.strip() == 'M']), ', Mean score:', np.mean(data['Total Score']).astype(int),
#           ', St Dev:', np.std(data['Total Score']).astype(int))

# For each prelim show some descriptive information
print('Prelim Total Score Descriptive Information')
for year in year_codes:
    data = df[df['Prelim Letter'] == year]
    #print('Prelim:', year,', N:', len(data), ', Mean score:', np.mean(data['Total Score']).astype(int), ', St Dev:', np.std(data['Total Score']).astype(int))
    print(year, len(data[data.Gender=='M']), len(data[data.Gender=='F']), np.mean(data['Total Score']).astype(int), np.std(data['Total Score']).astype(int))

# Run a multiple regression for combined prelim zscores predicting zscore given Cum GPA and Gender
regression_mask = (df.Gender != 'U') & (~(df['Cum GPA'].isnull()))
y = df[regression_mask]['zscore']
print(len(y))
x = df[regression_mask][['Cum GPA', 'Gender','formB','formC','formD','formE','formF','formG','formH','formI','formJ','formK','formM']]
print('female',len(x[x.Gender=='F']))
print('male',len(x[x.Gender=='M']))
x.Gender = (x.Gender.str.strip() == 'F').astype(int)
#x['Interaction'] = x['Cum GPA'] * x['Gender']
regress_plot(y, x, [2.2, 4.1],[-3.4,3.4], 'Cumulative graduate GPA', 'Preliminary Exam z-score')

# # Run a multiple regression for combined prelim zscores predicting zscore given Und GPA and Gender
# regression_mask = (df.Gender != 'U') & (~(df['Und GPA'].isnull()))
# y = df[regression_mask]['zscore']
# print(len(y))
# x = df[regression_mask][['Und GPA', 'Gender','formB','formC','formD','formE','formF','formG','formH','formI','formJ','formK','formM']]
# print('female',len(x[x.Gender=='F']))
# print('male',len(x[x.Gender=='M']))
# x.Gender = (x.Gender.str.strip() == 'F').astype(int)
# x['Interaction'] = x['Und GPA'] * x['Gender']
# regress_plot(y, x, [2.0, 4.1],[-2.5,2.5], 'Undergraduate GPA', 'Preliminary Exam z-score')

# # Run a multiple regression for combined prelim zscores predicting zscore given GRE SUB and Gender
# regression_mask = (df.Gender != 'U') & (~(df['GRE SUB'].isnull()))
# y = df[regression_mask]['zscore']
# print(len(y))
# x = df[regression_mask][['GRE SUB', 'Gender','formB','formC','formD','formE','formF','formG','formH','formI','formJ','formK','formM']]
# print('female',len(x[x.Gender=='F']))
# print('male',len(x[x.Gender=='M']))
# x.Gender = (x.Gender.str.strip() == 'F').astype(int)
# # x['Interaction'] = x['GRE SUB'] * x['Gender']
# regress_plot(y, x, [-2.5,2.5],[-2.5,2.5], 'PGRE z-score', 'Preliminary Exam z-score')

# # Run a multiple regression for combined prelim zscores predicting zscore given GRE Q and Gender
# regression_mask = (df.Gender != 'U') & (~(df['GRE Q'].isnull()))
# y = df[regression_mask]['zscore']
# print(len(y))
# x = df[regression_mask][['GRE Q', 'Gender','formB','formC','formD','formE','formF','formG','formH','formI','formJ','formK','formM']]
# print('female',len(x[x.Gender=='F']))
# print('male',len(x[x.Gender=='M']))
# x.Gender = (x.Gender.str.strip() == 'F').astype(int)
# x['Interaction'] = x['GRE Q'] * x['Gender']
# regress_plot(y, x, [-2.5,2.5],[-2.5,2.5], 'GRE Quantitative z-score', 'Preliminary Exam z-score')

# # Run a multiple regression for combined prelim zscores predicting zscore given GRE V and Gender
# regression_mask = (df.Gender != 'U') & (~(df['GRE V'].isnull()))
# y = df[regression_mask]['zscore']
# print(len(y))
# x = df[regression_mask][['GRE V', 'Gender','formB','formC','formD','formE','formF','formG','formH','formI','formJ','formK','formM']]
# print('female',len(x[x.Gender=='F']))
# print('male',len(x[x.Gender=='M']))
# x.Gender = (x.Gender.str.strip() == 'F').astype(int)
# x['Interaction'] = x['GRE V'] * x['Gender']
# regress_plot(y, x, [-2.5,2.5],[-2.5,2.5], 'GRE Verbal z-score', 'Preliminary Exam z-score')

# # Run a multiple regression for combined prelim zscores predicting zscore given GRE A and Gender
# regression_mask = (df.Gender != 'U') & (~(df['GRE A'].isnull()))
# y = df[regression_mask]['zscore']
# print(len(y))
# x = df[regression_mask][['GRE A', 'Gender','formB','formC','formD','formE','formF','formG','formH','formI','formJ','formK','formM']]
# print('female',len(x[x.Gender=='F']))
# print('male',len(x[x.Gender=='M']))
# x.Gender = (x.Gender.str.strip() == 'F').astype(int)
# x['Interaction'] = x['GRE A'] * x['Gender']
# regress_plot(y, x, [-2.5,2.5],[-2.5,2.5], 'GRE Writing z-score', 'Preliminary Exam z-score')

# # Run a multiple regression predicting Qtrs from Prelim pass to AC given zscore and Gender
# regression_mask = (df.Gender != 'U') & (~(df['Time diff AC prelim'].isnull()))
# y = (df[regression_mask]['Time diff AC prelim'])
# print(len(y))
# x = df[regression_mask][['zscore', 'Gender','formB','formC','formD','formE','formF','formG','formH','formI','formJ','formK','formM']]
# print('female',len(x[x.Gender=='F']))
# print('male',len(x[x.Gender=='M']))
# x.Gender = (x.Gender.str.strip() == 'F').astype(int)
# # x['Interaction'] = x['zscore'] * x['Gender']
# regress_plot(y, x, [-2.5,2.5],[-1,21], 'Preliminary Exam z-score', 'Quarters between Prelim and QE')

# # Run a multiple regression predicting Qtrs from prelim pass to Degree given zscore and Gender
# regression_mask = (df.Gender != 'U') & (~(df['Time diff Degree prelim'].isnull()))
# y = df[regression_mask]['Time diff Degree prelim']
# print(len(y))
# x = df[regression_mask][['zscore', 'Gender','formB','formC','formD','formE','formF','formG','formH','formI','formJ','formK','formM']]
# print('female',len(x[x.Gender=='F']))
# print('male',len(x[x.Gender=='M']))
# x.Gender = (x.Gender.str.strip() == 'F').astype(int)
# # x['Interaction'] = x['zscore'] * x['Gender']
# regress_plot(y, x, [-2.5,2.5],[5,36], 'Preliminary Exam z-score', 'Quarters between Prelim and Degree')

# # Run a multiple regression predicting Qtrs from AC to Degree given zscore and Gender
# regression_mask = (df.Gender != 'U') & (~(df['Time diff Degree AC'].isnull()))
# y = df[regression_mask]['Time diff Degree AC']
# print(len(y))
# x = df[regression_mask][['zscore', 'Gender','formB','formC','formD','formE','formF','formG','formH','formI','formJ','formK','formM']]
# print('female',len(x[x.Gender=='F']))
# print('male',len(x[x.Gender=='M']))
# x.Gender = (x.Gender.str.strip() == 'F').astype(int)
# #x['Interaction'] = x['zscore'] * x['Gender']
# regress_plot(y, x, [-2.5,2.5],[0,30], 'Preliminary Exam z-score', 'Quarters between QE and Degree')

# Run a multiple regression predicting Qtrs to prelim pass given zscore and Gender
# regression_mask = (df.Gender != 'U') & (~(df['Qtrs from admit to prelim pass'].isnull()))
# y = df[regression_mask]['Qtrs from admit to prelim pass']
# print(len(y))
# x = df[regression_mask][['zscore', 'Gender','formB','formC','formD','formE','formF','formG','formH','formI','formJ','formK','formM']]
# print('female',len(x[x.Gender=='F']))
# print('male',len(x[x.Gender=='M']))
# x.Gender = (x.Gender.str.strip() == 'F').astype(int)
# #x['Interaction'] = x['zscore'] * x['Gender']
# regress_plot(y.astype(int), x, [-2.5,2.5],[-1,15], 'Preliminary Exam z-score', 'Quarters to pass Prelim')

# # Run a multiple regression for combined prelim zscores predicting zscore given Cum GPA and Ethnicity
# regression_mask = (~df.Ethnicity.isnull()) & (~(df['Cum GPA'].isnull()))
# y = df[regression_mask]['zscore']
# x = df[regression_mask][['Cum GPA', 'Ethnicity']]
# x.Ethnicity = (~(x.Ethnicity.str.strip() == 'WH')).astype(int)
# #x['Interaction'] = x['Cum GPA'] * x['Gender']
# regress_plot_eth(y, x, [2.85, 4.1],[-2.5,2.5], 'Cumuluative graduate GPA', 'Preliminary Exam z-score')
#
# # Run a multiple regression for combined prelim zscores predicting zscore given Cum GPA and Citizen
# regression_mask = (~df.Citizen.isnull()) & (~(df['Cum GPA'].isnull()))
# y = df[regression_mask]['zscore']
# x = df[regression_mask][['Cum GPA', 'Citizen']]
# x.Citizen = (~(x.Citizen.str.strip() == 'Y') & ~(x.Citizen.str.strip() == 'PR')).astype(int)
# #x['Interaction'] = x['Cum GPA'] * x['Gender']
# regress_plot_cit(y, x, [2.85, 4.1],[-2.5,2.5], 'Cumuluative graduate GPA', 'Preliminary Exam z-score')

# # Run a multiple regression for combined prelim zscores predicting zscore given Cum GPA and Degree Status
# regression_mask = (~df['Degree Status'].isnull()) & (~(df['Cum GPA'].isnull()))
# y = df[regression_mask]['zscore']
# x = df[regression_mask][['Cum GPA', 'Degree Status']]
# x['Degree Status'] = (~(x['Degree Status'].str.strip() == 'DA')).astype(int)
# # x['Interaction'] = x['Cum GPA'] * x['Degree Status']
# regress_plot_deg(y, x, [2.85, 4.1],[-2.5,2.5], 'Cumuluative graduate GPA', 'Preliminary Exam z-score')

# Compare prelim z-scores for those who get PHD to those who do not (Degree Status = DA or not)
# Is average prelim score higher for those who complete PHD? (Prelim predicts for success?)
# If so, could this be caused by the fact that people get kicked out for low Prelim scores?







# Plot histogram of total scores in A year
# df[df['Letter of 1st prelim'] == 'A']['Total Score'].hist()

# # Create a pairwise correlation matrix of all the numeric data
# print(df.corr())
