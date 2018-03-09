import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('~/stylesheets/cbcc.mplstyle')

def e3fig(data, save_file):

    # setup the grid
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot2grid((2, 5), (0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((2, 5), (0, 1), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((2, 5), (0, 2), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((2, 5), (0, 3), rowspan=1, colspan=1)
    ax5 = plt.subplot2grid((2, 5), (0, 4), rowspan=1, colspan=1)
    ax6 = plt.subplot2grid((2, 5), (1, 0), rowspan=1, colspan=5)

    # # load in the crp data
    # data = pd.DataFrame.from_csv('~/Heal16implicit_analysis.csv')
    3
    # make the figure each axis at a time

    # AX1
    cond1_data = data[(data['task_condition'] == 5)] # just the weight condition

    g = sns.factorplot(x="lag", y="crp", data=cond1_data, units='subject', ax=ax1)
    ax1.set(ylim=[0., 0.15], xticks=[0, 5, 10], xticklabels=[-5, 0, 5])
    ax1.xaxis.label.set_visible(False)
    ax1.set(ylabel="Cond. Resp. Prob.")
    plt.close()

    # AX2
    cond2_data = data[(data['task_condition'] == 4)] # just the animacy condition

    sns.factorplot(x="lag", y="crp", data=cond2_data, units='subject', ax=ax2, color = '#fc4f30')
    ax2.set(ylim=[0., 0.15], xticks=[0, 5, 10], xticklabels=[-5, 0, 5])
    ax2.xaxis.label.set_visible(False)
    ax2.yaxis.label.set_visible(False)
    ax2.yaxis.set_visible(False)
    plt.close()

    # AX3
    cond3_data = data[(data['task_condition'] == 3)] # just the scenario condition

    sns.factorplot(x="lag", y="crp", data=cond3_data, units='subject', ax=ax3, color = '#e5ae38')
    ax3.set(ylim=[0., 0.15], xticks=[0, 5, 10], xticklabels=[-5, 0, 5])
    ax3.yaxis.label.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.set(xlabel="Lag")
    plt.close()

    # AX4
    cond4_data = data[(data['task_condition'] == 1)] # just the movie condition

    sns.factorplot(x="lag", y="crp", data=cond4_data, units='subject', ax=ax4, color = '#6d904f')
    ax4.set(ylim=[0., 0.15], xticks=[0, 5, 10], xticklabels=[-5, 0, 5])
    ax4.xaxis.label.set_visible(False)
    ax4.yaxis.set_visible(False)
    plt.close()

    # AX5
    cond5_data = data[(data['task_condition'] == 2)] # just the relational condition

    sns.factorplot(x="lag", y="crp", data=cond5_data, units='subject', ax=ax5, color = '#8b8b8b')
    ax5.set(ylim=[0., 0.15], xticks=[0, 5, 10], xticklabels=[-5, 0, 5])
    ax5.xaxis.label.set_visible(False)
    ax5.yaxis.set_visible(False)
    ax5.yaxis.label.set_visible(False)
    plt.close()

    # AX6
    barplot_data = data[(data['task_condition'] >= 1) & (data['task_condition'] <=5 )] # just the 5 conditions
    barplot_data = barplot_data[(barplot_data['lag'] == 0)] # only need one lag

    g = sns.barplot(x="task_condition", y="all_tf_z", data=barplot_data, ax=ax6, order=[5, 4, 3, 1, 2])
    g.set_xticklabels(["Weight", "Animacy", "Moving Scenario", "Movie", "Relational"])
    ax6.set(xlabel="Judgment Task", ylabel="z(TCE)", ylim=[-.025, 0.2])
    ax6.axhline(y=0, linewidth=1, linestyle='--', color='k')

    # save the figure
    plt.savefig(save_file + '.pdf')
    plt.close()