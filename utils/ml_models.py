import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import config as cf

from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels


def evaluate_pipeline(estimator, X, y):
    """
    Stratified split and Evaluate a model/pipeline
    :param estimator: estimator object implementing 'fit'
        Could be a model or pipeline.
    :param X: array-like :
        The data to fit. Can be for example a list, or an array.
    :param y: array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    :param pred_proba: boolean : default: True
        If True: then the result will include predict probability.
    :return: pd.DataFrame with columns 'X_test', 'y_test', 'pred', 'pred_proba'(optional)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=99)
    print('Train :', len(X_train), ' Test :', len(X_test), ' Total :', len(X_train) + len(X_test))

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    print(classification_report(y_test, y_pred))
    show_confusion_matrix(y_test, y_pred, True)

    return y_pred


def evaluate_cv_model(estimator, X, y, cv=5, print_result=True):
    """
    Stratified split and Evaluate by Cross-Validation for a model/pipeline

    :param estimator: estimator object implementing 'fit'
        Could be a model or pipeline.
    :param X: array-like :
        The data to fit. Can be for example a list, or an array.
    :param y: array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    :param cv: int, number of fold
    :param print_result: boolean, default: True
    :return: dict() with keys 'accuracy', 'precision', 'recall', 'f1'
    """
    print('===== Cross validation : CV=', cv)
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=99)
    scoring = {'accuracy': 'accuracy', 'precision': 'precision_macro', 'recall': 'recall_macro', 'f1': 'f1_macro'}
    score = cross_validate(estimator, X, y, cv=kfold, scoring=scoring, return_train_score=False, n_jobs=-1)
    score_df = pd.DataFrame(score)
    accuracy = score_df['test_accuracy']
    precision = score_df['test_precision']
    recall = score_df['test_recall']
    f1 = score_df['test_f1']

    if print_result:
        print("accuracy: {:.2f}% (+/- {:.2f}%)".format(accuracy.mean(), accuracy.std()))
        print("precision: {:.2f}% (+/- {:.2f}%)".format(precision.mean(), accuracy.std()))
        print("recall: {:.2f}% (+/- {:.2f}%)".format(recall.mean(), accuracy.std()))
        print("f1 score: {:.2f}% (+/- {:.2f}%)".format(f1.mean(), f1.std()))

    return {'accuracy': accuracy.mean(), 'precision': precision.mean(), 'recall': recall.mean(), 'f1': f1.mean()}


def evaluate_cv_all_models(X, y, model_dict, vectorizer, cv=5):
    result_list = []
    for name, model in model_dict.items():
        print('********** Evaluating : ', name, '*' * 10)
        pipeline = make_pipeline(vectorizer, model)
        eval_result = evaluate_cv_model(pipeline, X, y, cv, print_result=False)
        eval_result['model'] = name
        result_list.append(eval_result)

    result_all = pd.DataFrame(result_list).set_index('model')
    return result_all


# ---------- Benchmark models ----------------------------------------------


def evaluate_cv_test_size(estimator, X, y, train_size, n_jobs=None):
    """
    Evaluate an estimator with the specified training size.
    Note that this function use 'StratifiedShuffleSplit'
    :param estimator:
    :param X: array-like
    :param y: array-like
    :param train_size: int
    :param cv: int
    :param n_jobs: n_jobs for cross_validate
    :return:
    """
    if train_size < len(y):
        test_size = 1 - (train_size / float(len(y)))
        sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=99)
        score = cross_validate(estimator, X, y, cv=sss, scoring='f1_macro', return_train_score=True, n_jobs=n_jobs)
        return np.mean(score['train_score']), np.mean(score['test_score'])

    else:
        print('Train_size exceed len(X) : ', train_size)
        return np.nan, np.nan


def benchmark_models(selected_models, X, y, train_sizes=None, export_results=False,
                     export_path=cf.EXPORT_PATH + 'embedding_benchmark_results/', n_jobs=None):
    """
    Benchmark models with a list of train_sizes
    :param selected_models: dict(), (model_name: estimator)
    :param X: array-like
    :param y: array-like
    :param train_sizes: list of int
    :param export_results: boolean, if true, export result to export_path
    :param export_path: String, Destination directory to export results
                        The results is saved in separated file by model
    :param n_jobs: n_jobs for cross_validate function
    :return: pd.dataFrame of 'model', 'score', 'score_type', 'train_size'
    """
    all_results = []
    if train_sizes is None:
        train_sizes = [100, 200, 400, 800, 1600, 3200, 6400, 10000]

    for model_name, evaluator in selected_models.items():
        print(model_name, ': ', end='')
        model_result = []
        for n in train_sizes:
            print(n, '...', end='')
            train_scores, test_scores = evaluate_cv_test_size(evaluator, X, y, n, n_jobs=n_jobs)
            model_result.append({'model': model_name, 'score_type': 'train', 'score': train_scores, 'train_size': n})
            model_result.append({'model': model_name, 'score_type': 'test', 'score': test_scores, 'train_size': n})
        print('done !')
        model_result = pd.DataFrame(model_result)
        if (export_results):
            file_name = export_path + model_name + '.csv'
            model_result.to_csv(file_name, index=False)
            print('Export done! :', file_name)
        all_results.append(model_result)
    all_results = pd.concat(all_results, axis=0, ignore_index=True, sort=False)
    return all_results


def show_confusion_matrix(y_test, y_pred, is_scaled=False, title=None, figsize=(20, 15), rot_x_label=90):
    index = unique_labels(y_test, y_pred)
    index = unique_labels(y_test, y_pred)
    temp = pd.DataFrame(index)[0].str.split('::')
    index = temp.str[1] + '::' + temp.str[2]
    cm = confusion_matrix(y_test, y_pred)
    fmt = 'd'
    if is_scaled:
        cm = (cm.astype('float').round(2) / cm.sum(axis=1)[:, np.newaxis]).round(2)
        fmt = '.2g'

    confusion_mat = pd.DataFrame(cm, columns=index, index=index)
    confusion_mat.index.name = 'Actual'
    confusion_mat.columns.name = 'Predicted'
    plt.figure(figsize=figsize)
    sns.set_context('notebook', font_scale=0.8)
    fig = sns.heatmap(confusion_mat, fmt=fmt, annot=True, annot_kws={"size": 8},
                        cbar_kws = dict(use_gridspec=False,location="top"))
    fig.set(title=title)
    loc, labels = plt.xticks()
    fig.set_xticklabels(labels, rotation=rot_x_label)


def plot_benchmark_training_sizes(data, hue, title, figsize=(15, 5)):
    plt.figure(figsize=figsize)
    fig = sns.pointplot(x='train_size', y='score', hue=hue, data=data)
    sns.set_context('notebook', font_scale=1.5)
    sns.set_style('darkgrid')
    fig.set(xlabel='Training examples')
    fig.set(title=title)
    fig.set(ylabel="F1-Score")


def plot_benchmark_train_test_score(df, model_name, figsize=(15, 5)):
    df_filtered = df[df['model'] == model_name]
    title = 'Training and Testing F1-score at different training sizes for \'' + model_name + '\' model'
    plot_benchmark_training_sizes(df_filtered, 'score_type', title, figsize)


def plot_benchmark_test_score(df, model_list=None, figsize=(15, 5)):
    df_filtered = df
    if model_list is not None:
        df_filtered = df[df['model'].isin(model_list)]
    df_filtered = df_filtered[df_filtered['score_type'] == 'test']
    title = 'Benchmark at different training sizes'
    plot_benchmark_training_sizes(df_filtered, 'model', title, figsize)


def plot_f1_threshold(result, model_name=''):
    result['pred_proba'].plot.hist(title='Histogram of predict_proba')
    performance = []
    for threshold in np.arange(0, 1.0, 0.01):
        confident_result = result[result['pred_proba'] >= threshold]
        weak_result = result[result['pred_proba'] < threshold]

        confident_score = f1_score(confident_result['Category'], confident_result['pred'], average='macro')
        weak_score = f1_score(weak_result['Category'], weak_result['pred'], average='macro')
        performance.append({'threshold': threshold, 'confident': confident_score, 'weak': weak_score})
    performance = pd.DataFrame(performance)
    performance = performance.set_index('threshold')
    performance.plot.line(title=('F1-score vs threshold for model: {}'.format(model_name)))


def export_results(model_name, score):
    result = pd.DataFrame(
        {'Model': model_name, 'Accuracy': score['test_accuracy'].mean(), 'Precision': score['test_precision'].mean(),
         'Recall': score['test_recall'].mean(), 'F1-score': score['test_f1'].mean()}, index=[0])
    file_name = '{}embedding_results/{}.csv'.format(cf.EXPORT_PATH, model_name)
    result.to_csv((file_name), index=False)
    print('Export done! : ', file_name)
