from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from imblearn.over_sampling import SMOTE
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

# Unused in this script but kept here since used in notebook
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import confusion_matrix
from scipy import stats


def plot_churn_percent(time_delta, time_unit, plot_title, min_value, max_value):
    '''Function that given a series of timedeltas will plot a barchart on a given time
    resolution (i.e. D,M,m,s)'''

    # turn account age into timedelta of user defined time unit
    account_age = time_delta.astype('timedelta64[{}]'.format(time_unit))

    # turn the account ages into a pandas series of percentage overall user churn by given time unit
    plot_this = account_age.value_counts(normalize=True).sort_index() * 100

    # subset pandas series on the range of interest
    plot_this = plot_this[(plot_this.index >= min_value) & (plot_this.index <= max_value)]

    plt.bar(plot_this.index, plot_this.values)

    plt.title('User churn by {}'.format(plot_title))

    plt.xlabel('length of account in {}'.format(plot_title))
    plt.ylabel('% of user (n={})'.format(len(account_age)))
    #plt.ylim((0, 100))
    plt.xlim(min_value, max_value)
    return plt


def df_with_headers(cur, table_name):
    """pull a sql table in as a pandas dataframe

    Takes a table in a cursor object (which is a connection to the entire database) and selects a table from the database and pulls it into a pandas dataframe with column names

    :param cur: mysql cursor object
    :param table_name: table name in sql database
    :return: pandas data frame
    """
    # Run a SQL command on cursor object
    do_this = "DESCRIBE {}".format(table_name)
    cur.execute(do_this)

    # The method .fetchall() fetches all (or all remaining) rows of a query result set and returns a list of tuples.
    # If no more rows are available, it returns an empty list.

    # get the column names of a table and character info, primary key (yes/no), default value
    stuff = cur.fetchall()
    # turn it into a pd dataframe
    df = pd.DataFrame(list(stuff))

    # Turn first column into pd vector of colnames
    col_names = [str(x) for x in list(df[0])]
    # turn that into a list
    col_list = ",".join(col_names)

    # Create an sql statement using the list of column names
    do_this2 = "SELECT {} FROM {}".format(col_list, table_name)

    # run the sql statement on the cursor object
    cur.execute(do_this2)

    # fetch remaining rows and return as tuples
    stuff2 = cur.fetchall()

    df2 = pd.DataFrame(list(stuff2), columns=col_names)

    return df2


def table_returner(data_base, table_name):
    """

    :param data_base:
    :param table_name:
    :return:
    """
    cnx = connectmysql(data_base)
    cur = cnx.cursor()

    # Pull out a table
    table = df_with_headers(cur, table_name)

    return table


def connectmysql(db_name):
    """creates a MySQL connection

    :param db_name: str
        databases are constant_therapy and ct_customer
    :return: mysql connection
    """
    config = {
        'user': 'user_name',
        'password': 'pass_word',
        'host': '127.0.0.1',
        'port': '3307',
        'database': db_name
    }

    # open database connection
    cnx = None
    try:
        cnx = mysql.connector.connect(**config)
    except mysql.connector.Error as err:
        if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        raise

    print ('Connected to {} database'.format(db_name))
    return cnx


def run_sql(sql_query):
    """ connects to remote db and runs sql statement returning dataframe with results
    :param sql_query: str
    :return: pandas dataframe
    """
    # Connect to database
    cnx = connectmysql('constant_therapy')

    # create a cursor object
    cur = cnx.cursor()

    # execute the sql command
    cur.execute(sql_query)

    # Pull the data in the cursor into a list of tuples stored in a notebook (13 min)
    sql_query = cur.fetchall()

    # Pull the list of tuples into a dataframe
    sql_results = pd.DataFrame(sql_query)
    return sql_results


def view_percent(data_frame, feature_name):
    """View feature frequency distribution

    :param data_frame: pandas dataframe
    :param feature_name: str
        column name with feature of interest
    :return: feature frequencies among users
    """

    print('Number of feature classes:')
    print(len(data_frame[feature_name].unique()))
    print
    print(data_frame[feature_name].value_counts(normalize=True)*100)


def view_counts(data_frame, feature_name):
    """counts the instances of each feature and plots their distribution

    :param data_frame: pandas dataframe
    :param feature_name: str
        column name with feature of interest
    :return: feature frequencies among users
    """
    '''counts the instances of each feature and plots their distribution'''

    print('Number of feature classes:')
    print(len(data_frame[feature_name].unique()))
    print
    print(data_frame[feature_name].value_counts())
    data_frame[feature_name].value_counts().plot(kind='bar')


def run_cv(x, y, clf_class, **kwargs):
    """tests the accuracy of a given classifier

    :param x: numpy array
        training set
    :param y: numpy array
        target variable
    :param clf_class: str
        classifier to apply:
        LR = Logistic regression
        GBC = Gradient Boosting Classifier
        SVC = Support Vector Machine
        RF = Random Forest
        KNN = K-nearest neighbor

    :param kwargs: key word arguments
    :return:
    """

    # Construct a k-folds object
    kf = KFold(len(y), n_folds=3, shuffle=True)
    # create a copy of target variable
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]

        # Initialize a classifier with key word arguments (**kwargs)
        # this produces LR(x_resampled,y_resampled) by replacing using kwargs
        clf = clf_class(**kwargs)
        clf.fit(x_train, y_train)
        y_pred[test_index] = clf.predict(x_test)
    return y_pred


def feature_importance(df, dropped_features):

    """Calculates the importance of features in a random forest classifier and model accuracy

    df= pandas data frame of features and target
    dropped_features = list of names of features to drop (as they appear in dataframe

    :param df: pandas data frame
    data frame of features and target variable
    :param dropped_features: list
    list of names of features to drop (as they appear in dataframe)
    :return: data frame
    data frame of feature importance for each model
    """

    # Make a copy so we don't alter original data frame
    features = df.copy()

    # Isolate target data
    completed_session = features['session_completed']
    y = np.asarray(completed_session)

    # Remove y from the features data frame
    features.pop('session_completed')

    # Encode the categorical variables using dummy encoding
    # This encodes categorical data 1-0
    cleaned_features_dummy = pd.get_dummies(features)

    # Remove unwanted features from the features data frame
    cleaned_features_dummy = cleaned_features_dummy.drop(dropped_features, axis=1)

    # change the data frame to a matrix of floats so that a scaler can be applied
    x = cleaned_features_dummy.as_matrix().astype(np.float)

    # Apply the standard scaler
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Apply regular SMOTE
    sm = SMOTE(kind='regular')
    x_resampled, y_resampled = sm.fit_sample(x, y)

    # Build a random forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(x_resampled, y_resampled)
    importances = forest.feature_importances_

    print("Accuracy:")
    # Calculate the model accuracy
    print("%.3f" % accuracy(y_resampled, run_cv(x_resampled, y_resampled, RF)))

    # Standard deviation for each importance
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    # 1-D array of the feature indices in descending order of importance
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    num_of_features = len(list(cleaned_features_dummy))
    labels = np.asarray(list(cleaned_features_dummy))

    for f in range(num_of_features):
        print("%d. feature %s (%f)" % (f + 1, labels[indices[f]], importances[indices[f]]))


def feature_importance_df(df):
    """Calculates the importance of features in a random forest classifier (after applying smote and
    a standard scaler),returns a dataframe with feature importance, the standard error for that feature importance,
    and the linear correlation between the feature and the target variable (first session completion).

    df= pandas data frame of features and target"""

    # Make copies so we don't alter original dataframe
    features1 = df.copy()
    features2 = df.copy()

    # Create a correlation matrix
    cm = corr_dataframe(features1)
    sc_only = cm[cm['level_0'] == 'session_completed']

    # Isolate target data
    completed_session = features2['session_completed']
    y = np.asarray(completed_session)

    # Remove y from the features dataframe
    features2.pop('session_completed')

    # Encode the categorical variables using dummy encoding
    # This encodes categorical data 1-0
    cleaned_features_dummy = pd.get_dummies(features2)

    # change the data frame to a matrix of floats so that scaler can be applied
    X = cleaned_features_dummy.as_matrix().astype(np.float)

    # Apply the standard scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Apply regular SMOTE
    sm = SMOTE(kind='regular')
    X_resampled, y_resampled = sm.fit_sample(X, y)

    # set parameters for random forest
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    # build random forest
    forest.fit(X_resampled, y_resampled)

    # empty data frame to hold output
    df = pd.DataFrame()

    # num_of_features = len(list(cleaned_features_dummy))
    labels = np.asarray(list(cleaned_features_dummy))
    df['features'] = pd.Series(labels)

    # compute the feature importance for random forest; importance returned as numpy array
    importances = forest.feature_importances_
    df['feature_importance'] = pd.Series(importances)  # add to output df

    # numpy array of standard deviation for each importance
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    df['standard_error'] = pd.Series(std)  # add to output df

    # Data frame of absolute values of feature vs session_completion correlations
    # out = corr_dataframe(features)
    # sc_only = out[out['level_0'] == 'session_completed']

    merged_df = df.merge(sc_only, how='outer', left_on='features', right_on='level_1')

    df_sorted = merged_df.sort_values('feature_importance', ascending=False)
    df_sorted.pop('level_0')
    df_sorted.pop('level_1')
    df_sorted.rename(columns={'correlation': 'corr_to_session_compl'}, inplace=True)
    return df_sorted


def draw_confusion_matrices(confusion_matrices, class_names):
    """Creates a confusion matrix for a list of confusion matrices parameters

    :param confusion_matrices: list of tuples
    list of tuples of model name, confusion_matrix function prompt
    :param class_names: str
    the two class names for the model
    :return: confusion matrix plots for models
    """
    class_names = class_names.tolist()
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        print(cm)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.savefig('x_figures/4_modelling/confusion_matrix_%s.jpg' %classifier)
        plt.show()


def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # NumPy interpretes True and False as 1. and 0.
    return np.mean(y_true == y_pred)


def plot_churn_percent(time_delta, time_unit, plot_title, min_value, max_value, fig_name):
    """Plots a bar chart of percent user churn over time

    :param time_delta: pandas series
        Series containing account lengths for users as a time delta
    :param time_unit: str
        Time unit of interest (i.e. D,M,m,s)
    :param plot_title: str
        Title at top of plot
    :param min_value: int
        minimum account length of interest for given time unit (i.e. 1 day)
    :param max_value: int
        maximum account length of interest for given time unit (i.e. 10 days)
    :param fig_name:
        name to save figure as

    :return: matplot lib figure
    """

    # turn account age into timedelta of user defined time unit
    account_age = time_delta.astype('timedelta64[{}]'.format(time_unit))

    # turn the account ages into a pandas series of percentage overall user churn by given time unit
    plot_this = account_age.value_counts(normalize=True).sort_index() * 100

    # subset pandas series on the range of interest
    plot_this = plot_this[(plot_this.index >= min_value) & (plot_this.index <= max_value)]

    df = pd.DataFrame()
    df['x_plot'] = plot_this.index
    df['y_plot'] =plot_this.to_frame()
    df.x_plot = df.x_plot.astype(int)

    # Set font size for axes
    sns.set_context("paper", rc={"font.size": 12,
                                 "axes.titlesize": 16,
                                 "axes.labelsize": 12})

    # Set up rotation of x labels
    # plt.xticks(rotation=30,)

    ax = sns.barplot(
        y=df['y_plot'],
        x=df['x_plot'],
        data=df,
        label='Label',
        ci=95,
        palette='Blues_r'  # r reverse the color palette
        )


    ax.set(
        xlabel=('length of account in {}'.format(plot_title.lower())),
        ylabel='% of user (n={})'.format(len(account_age)),
        ylim=(0, max(plot_this.values) + max(plot_this.values) * 0.05),
        xlim=(min_value, max_value),
        # xticklabels=(range(1,21), 5), TODO show every n tick when axis is crowded
        title=('User churn by {}'.format(plot_title))
        )

    ax.figure.tight_layout()
    # Save figure to figures folder for project
    ax.figure.savefig('x_figures/2_initial_exploration/{}.jpg'.format(fig_name),dpi =300)


def plot_churn2(data_frame, time_unit, plot_title, unit, min_value, max_value):
    # Make a copy of the dataframe
    df = data_frame.copy()

    # convert the timedelta to a number
    df['account_age'] = df['account_age'].astype('timedelta64[{}]'.format(time_unit))


    # Splits the dataframe into two dataframes, one with all users that completed the first session
    # and one with users that didn't complete the first session
    not_completed = df[df['session_completed'] == 0]
    completed = df[df['session_completed'] == 1]

    # normalize counts to percentage of total user base
    plot_not_completed = not_completed['account_age'].value_counts().sort_index() / len(df) * 100
    plot_completed = completed['account_age'].value_counts().sort_index() / len(df) * 100

    not_comp = pd.DataFrame(plot_not_completed)
    comp = pd.DataFrame(plot_completed)

    result = pd.concat([not_comp, comp], axis=1)
    result.columns = ['didn\'t complete first session', 'completed first session']

    subset_index = result[(result.index >= min_value) & (result.index <= max_value)]

    account_age_plt = subset_index.plot(
        title='{}'.format(plot_title),
        kind='bar',
        color=['red', 'green'])

    account_age_plt.set_xlabel('length of account in {}'.format(unit))
    account_age_plt.set_ylabel('% of total user base')
    return account_age_plt


def funnel(df, time_unit, num):
    '''function takes a dataframe, time unit (months,days,hours,minutes,seconds) and a number
    and returns the percentage of users that drop out during in this period
    (i.e. funnel(df,'days', 31)will return what percentage of users in dataframe df stopped using the app within the
    first 31 days)'''

    # number of users with an account age less than or equal to the given number of time units input by user
    churn = len(df[df['account_age'] <= datetime.timedelta(**{time_unit: num})])

    # turn into percentage
    percent_churn = round(100 * float(churn) / float(len(df)), 2)

    return percent_churn


def visualize_smote(X,y,X_resampled, y_resampled):
    """Visualizes the effect of SMOTE using a PCA.

    The function takes original feature numpy array (X),original classifier array (y), smote re-sampled feature array (X_re-sampled), and smote re-sampled feature array (X_re-sampled)

    :param X: numpy array
        the original training sets
    :param y: numpy array
        the original target data
    :param X_resampled: numpy array
        the smote re-sampled training set
    :param y_resampled: numpy array
        the smote resampled target data
    :return: Matplotlib figure
        Figure showing the PCA plots for the training sets before and after the SMOTE over sampling is applied
    """

    #print(__doc__)

    #set seaborn plot aesthetics
    sns.set()
    # Define some color for the plotting
    almost_black = '#262626'
    palette = sns.color_palette()

    # Instanciate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=2)

    # Fit and transform x to visualise inside a 2D feature space
    X_vis = pca.fit_transform(X)
    X_res_vis = pca.transform(X_resampled)

    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
                edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
    ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=0.5,
                edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)
    ax1.set_title('Original set')

    ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
                label="Class #0", alpha=.5, edgecolor=almost_black,
                facecolor=palette[0], linewidth=0.15)
    ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
                label="Class #1", alpha=.5, edgecolor=almost_black,
                facecolor=palette[2], linewidth=0.15)
    ax2.set_title('SMOTE regular')

    plt.savefig('x_figures/4_modelling/smote.jpg')

    plt.show()


def plot_roc(x, y, clf_class, **kwargs):
    """TODO improve comments for this function"""
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y), 2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    # all_tpr = []
    for i, (train_index, test_index) in enumerate(kf):
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(x_train, y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(x_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    plt.savefig('x_figures/4_modelling/roc_%s.jpg' % clf_class)
    plt.show()


def heat_map(dataframe):
    '''Creates a correlation heat map for columns in a dataframe
    df = pandas dataframe
    '''

    # Features are dummy encoded
    df = pd.get_dummies(dataframe.copy())

    # Set the seaborn plot space
    sns.set(context="paper", font="monospace")

    # Calculate feature correlation
    corrmat = df.corr()

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 9))

    # Draw the heatmap using seaborn
    sns.heatmap(corrmat, vmax=.8, square=True)

    return f.tight_layout()


def corr_dataframe(feature_dataframe):
    '''calulates the correlations between all columns in a dataframe and
    returns them as a new dataframe
    '''

    # Features are dummy encoded
    df = pd.get_dummies(feature_dataframe.copy())
    #calculate correlation between features in the dataframe
    c = df.corr()#.abs()
    #unstack the series
    s = c.unstack()

    #make the series a dataframe
    df = s.to_frame(name='correlation')
    df['abs_val'] = df['correlation'].abs()
    #reset the indices so that indices become columns
    df.reset_index(inplace=True)
    #remove rows where a feature is compared to itself
    feature_correlation = df[df['level_0']!=df['level_1']]
    #sort by absolute value of feature correlation
    new = feature_correlation.sort_values('abs_val', ascending=False)
    new.pop('abs_val')
    return new


def model_performance(l):
    true_negatives = l[0]
    false_positives = l[1]
    false_negatives = l[2]
    true_positives = l[3]

    total = (true_negatives+false_positives+false_negatives+true_positives)

    print('Precision:')
    print(true_positives/(true_positives + false_positives))

    print('Recall:')
    print(true_positives/(true_positives + false_negatives))

    print('Accuracy:')
    print((true_positives+true_negatives)/total)


def performance(model, X, y):
    """Calculates performance of a given model

    :param model: model name
    :param X: features
    :param y: target variable
    :return: model performance
    """
    # calculate confusion matrix and extract it from an array to a list
    l = confusion_matrix(y, run_cv(X, y, model)).tolist()
    # extract list of lists to one list
    cm = [item for sublist in l for item in sublist]
    # calculate model performance
    return model_performance(cm)



