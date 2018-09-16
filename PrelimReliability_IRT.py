import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import statsmodels.api as sm
from scipy.stats import pearsonr

# Coefficient Alpha function downloaded from https://github.com/statsmodels/statsmodels/issues/1272
def CronbachAlpha(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=0, ddof=1)
    tscores = itemscores.sum(axis=1)
    nitems = itemscores.shape[1]
    calpha = nitems / float(nitems-1) * (1 - itemvars.sum() / float(tscores.var(ddof=1)))

    return calpha

# Read the data file
df = pd.read_excel(open('AllPrelimData.xls','rb'), sheetname='Sheet1')

# Filter rows by existence of prelim data
#   1) Get rid of rows with no prelim data
#   2) Get rid of rows with all 0's on first questions, and then second 10 questions
cols = [i+1 for i in range(20)]
even_cols = [i+1 for i in range(20) if (i + 1) % 2 == 0]
odd_cols = [i+1 for i in range(20) if (i + 1) % 2 == 1]

df = df.dropna(how = 'any', subset = cols)
df = df[~(df[cols[:10]] == 0).all(axis = 1)]
df = df[~(df[cols[10:]] == 0).all(axis = 1)]

# Create extra column to store total Prelim score by adding up individual item scores
df['Total Score'] = df[cols].sum(axis = 1)

# Separate prelim by form (year of administration) using code 'Letter of 1st prelim'
year_codes = sorted(set(df['Prelim Letter']))

# Calculate z scores
year_mean = {year: df[df['Prelim Letter'] == year]['Total Score'].mean() for year in year_codes}
year_std = {year: df[df['Prelim Letter'] == year]['Total Score'].std(ddof=0) for year in year_codes}

df['zscore'] = df.apply((lambda x: (x['Total Score'] - year_mean[x['Prelim Letter']]) / year_std[x['Prelim Letter']]), axis = 1)


# Reliability first/second:
# Create a new column for First Half Score (sum Prob1-Prob10)
# Create a new column for Second Half Score (sum Prob11-Prob20)
df['First Half Score'] = df[cols[:10]].sum(axis = 1)
df['Second Half Score'] = df[cols[10:]].sum(axis = 1)
# For each prelim find correlation between First Half Score and Second Half Score
print('Split Halves Reliability First/Second and Spearman-Brown Prophecy correction')
for year in year_codes:
    data = df[df['Prelim Letter'] == year]
    corrs = np.array([pearsonr(data['First Half Score'], data['Second Half Score'])[0]])
    print(year, corrs)

    # Use Spearman-Brown Prophecy formula to correct reliability coefficient
    SBP_corrs =  2*corrs / (1 + corrs)
    print(year,SBP_corrs)
print('\n\n')

# Reliability even/odd:
# Create a new column for Even Half Score (sum Prob2,4,6,8,10,12,14,16,18,20)
# Create a new column for Odd Half Score (sum Prob1,3,5,7,9,11,13,15,17,19)
df['Even Half Score'] = df[even_cols].sum(axis = 1)
df['Odd Half Score'] = df[odd_cols].sum(axis = 1)
# For each prelim find correlation between Even Half Score and Odd Half Score
print('Split Halves Reliability Even/Odd and Spearman-Brown Prophecy correction')
for year in year_codes:
    data = df[df['Prelim Letter'] == year]
    corrs = np.array([pearsonr(data['Even Half Score'], data['Odd Half Score'])[0]])
    print(year, corrs)

    # Use Spearman-Brown Prophecy formula to correct reliability coefficient
    SBP_corrs =  2*corrs / (1 + corrs)
    print(year,SBP_corrs)
print('\n\n')

# Internal consistency coefficient alpha aka "Cronbach's alpha":
# For each prelim calculate coefficient alpha
print('Coefficient Alpha')
for year in year_codes:
    data = df[df['Prelim Letter'] == year]
    alpha = CronbachAlpha(data[cols])
    print(year, alpha)
print('\n\n')

# Item Response Theory:
# For each problem of each prelim calculate item Difficulty = Average Score / Maximum Score (=50)
print('Item Difficulty')
for year in year_codes:
    data = df[df['Prelim Letter'] == year]
    diff = [np.mean(data[col])/50 for col in cols]
    print(year, diff)
print('\n\n')

# For each problem of each prelim calculate item Discrimination = correlation between item scores and Total Scores
print('Item Discrimination')
for year in year_codes:
    data = df[df['Prelim Letter'] == year]
    corrs = [pearsonr(data[col], data['Total Score'])[0] for col in cols]
    print(year, corrs)


# For each prelim show some descriptive information
print('Prelim Total Score Descriptive Information')
for year in year_codes:
    data = df[df['Prelim Letter'] == year]
    #print('Prelim:', year,', N:', len(data), ', Mean score:', np.mean(data['Total Score']).astype(int), ', St Dev:', np.std(data['Total Score']).astype(int))
    print(year, len(data), np.mean(data['Total Score']).astype(int), np.std(data['Total Score']).astype(int))

# # Plot histogram of total scores in A year
# for year in year_codes:
#     df[df['Prelim Letter'] == year]['Total Score'].hist(range=(0,1000),bins=20)
#     plt.xlim([0,1000])
#     plt.ylim([0,15])
#     plt.xlabel('Preliminary Exam Score')
#     plt.ylabel('Counts')
#     plt.title('Year:%s' % year)
#     plt.show()
fig, axs = plt.subplots(4,3,sharex=True,sharey=True)
axs = axs.ravel()
i=0
for year in year_codes:
    axs[i].hist(df[df['Prelim Letter'] == year]['Total Score'], range=(0,1000),bins=20)
    #plt.plot(x['Cum GPA'],model.params[0] + x['Cum GPA']*model.params[1] + x['Cum GPA']*model.params[3])
    #axs[i].xlim([2.8,4.1])
    #axs[i].ylim([0,1000])
    #plt.xlabel('Cumulative graduate GPA')
    axs[i].set_title('%s' % year)
    #axs[i].legend(loc='best')
    #plt.show()
    print('\n\n')
    i = i + 1
plt.show()

# Plot histogram combined all scores across years
df['Total Score'].hist(range=(0,1000),bins=15)
plt.xlim([0, 1000])
plt.ylim([0, 71])
plt.xlabel('Preliminary Exam Score')
plt.ylabel('Counts')
plt.title('Combined Years A:M')
plt.show()

# Plot histogram combined all zscores across years
df['zscore'].hist(range=(-3,3),bins=15)
plt.xlim([-3, 3])
plt.ylim([0, 71])
plt.xlabel('Preliminary Exam z-score')
plt.ylabel('Counts')
plt.title('Combined Years A:M')
plt.show()

# # Create a pairwise correlation matrix of all the numeric data
# print(df.corr())
