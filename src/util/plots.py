import matplotlib.pyplot as plt
# from matplotlib import colors
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

def colored_corr_matrix(df, name=None, file=None):
    # set the colormap and centre the colorbar on an arbitary midpoint

    # class MidpointNormalize(colors.Normalize):
    #     """
    #     Normalise the colorbar so that diverging bars work way either side from a prescribed midpoint value)
    #
    #     e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    #     """
    #
    #     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
    #         self.midpoint = midpoint
    #         colors.Normalize.__init__(self, vmin, vmax, clip)
    #
    #     def __call__(self, value, clip=None):
    #         # I'm ignoring masked values and all kinds of edge cases to make a
    #         # simple example...
    #         x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
    #         return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

    if len(df.columns) > 9:
        size = (9, 7)
    else:
        size = (6.5, 6)
    fig, ax = plt.subplots(figsize=size)
    matrix = df.values
    # matrix[matrix == 1] = np.NaN
    if np.nanmin(matrix) > 0:
        if np.nanmax(matrix) <= 1.0:
            im = ax.matshow(matrix, cmap=plt.cm.coolwarm_r)
        else:  # [0, inf+]
            im = ax.matshow(matrix, cmap=plt.cm.coolwarm_r)
    else:
        im = ax.matshow(matrix, cmap=plt.cm.Spectral,
                        # clim=(np.nanmin(matrix), np.nanmax(matrix)),
                        # norm=MidpointNormalize(midpoint=0, vmin=np.max(matrix), vmax=np.max(matrix))
                        )
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index.to_numpy())))
    ax.set_xticklabels(df.columns, rotation='horizontal')
    ax.set_yticklabels(df.index.to_numpy())
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

    #     print(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j and matrix.shape[0] == matrix.shape[1]:
                continue
            num = matrix[i, j]
            # seems like .matshow and .text have different xy orientation...
            ax.text(j, i, '{:.2f}'.format(num), fontsize=13, va='center', ha='center')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('correlation', rotation=90)
    plt.tight_layout()
    if file:
        plt.savefig(file, bbox_inches='tight')
    return fig, ax

def partial_dependence_titanic(pth):
    from data.real_data import get_titanic
    df = get_titanic(original=True)
    # lending= get_lending(2000, original=True)
    # X = df.drop(columns=['Survived'])
    df = df.dropna(subset=['Pclass', 'Age', 'Fare', 'Sex'])
    X = df[['Pclass', 'Age', 'Fare', 'Parch', 'Sex']]
    X['Sex'].replace({'male': 1, 'female': 0}, inplace=True)
    # X = df.drop(columns=['Survived', ])
    y = df['Survived']
    # X,y = lending.drop(columns=['loan_amnt']), lending['loan_amnt']
    # clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, max_depth = 1, random_state = 0).fit(X,y)
    clf = RandomForestClassifier(200, min_samples_split=20, min_samples_leaf=5, max_features=3).fit(X,y)
    # clf = LogisticRegression().fit(X,y)
    fig,_ = plt.subplots(ncols=3, figsize=(8,4))
    plot_partial_dependence(clf, X, ['Pclass', 'Age', 'Fare'], fig=fig, grid_resolution=50)
    fig = plt.gcf()
    axes = fig.get_axes()

    axes[1].set_ylabel('Partial dependence Survived')
    axes[1].set_xticks([1,2,3])
    axes[1].set_xticklabels([1,2,3])
    fig.tight_layout()
    fig.savefig(pth, bbox_inches='tight' )
    plt.show()


if __name__ == '__main__':
    # g1 = multivariate_normal([1, 1], [1, 2])
    # g2 = multivariate_normal([1, 1], [3, 4])
    # plt.scatter(g1)
    # plt.scatter(g2)
    #
    # plt.show()
    partial_dependence_titanic('../../_figures/review/partial_dependence.pdf')
