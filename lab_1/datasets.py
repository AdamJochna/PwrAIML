import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import PROJECT_PATH

datasets_data = {
    'iris': {
        'columns': [
            'sepal_length',
            'sepal_width',
            'petal_length',
            'petal_width',
            'class'
        ],
        'y_column': 'class',
        'class_mapping': {
            0: 'Iris-setosa',
            1: 'Iris-versicolor',
            2: 'Iris-virginica'
        },
        'plot_0_cols': ['sepal_width', 'sepal_length']
    },
    'glass': {
        'columns': [
            'Id',
            'RI',
            'Na',
            'Mg',
            'Al',
            'Si',
            'K',
            'Ca',
            'Ba',
            'Fe',
            'type_of_glass'
        ],
        'y_column': 'type_of_glass',
        'class_mapping': None,
        'class_mapping': {
            1: 'building_windows_float_processed',
            2: 'building_windows_non_float_processed',
            3: 'vehicle_windows_float_processed',
            4: 'vehicle_windows_non_float_processed (none in this database)',
            5: 'containers',
            6: 'tableware',
            7: 'headlamps',
        },
        'plot_0_cols': ['RI', 'Si']
    },
    'wine': {
        'columns': [
            'class',
            'alcohol',
            'malic_acid',
            'ash',
            'alcalinity_of_ash',
            'magnesium',
            'total_phenols',
            'flavanoids',
            'nonflavanoid_phenols',
            'proanthocyanins',
            'color_intensity',
            'hue',
            'OD280/OD315_of_diluted_wines',
            'proline'
        ],
        'y_column': 'class',
        'class_mapping': {
            1: 'class_1',
            2: 'class_2',
            3: 'class_3',
        },
        'plot_0_cols': ['alcohol', 'malic_acid']
    },
}


def get_dataset(dataset_name, normalize_x):
    assert dataset_name in ['iris', 'glass', 'wine']
    df = pd.read_csv(
        '{}/lab_0/datasets/{}/{}.data'.format(
            PROJECT_PATH,
            dataset_name,
            dataset_name
        ),
        header=None
    )

    df.columns = datasets_data[dataset_name]['columns']
    y_col = datasets_data[dataset_name]['y_column']

    x_cols = datasets_data[dataset_name]['columns'].copy()
    x_cols.remove(datasets_data[dataset_name]['y_column'])

    if 'Id' in x_cols:
        x_cols.remove('Id')

    if dataset_name == 'iris':
        class_inv_mapping = {v: k for k, v in datasets_data[dataset_name]['class_mapping'].items()}
        df[y_col] = df[y_col].apply(lambda x: class_inv_mapping[x])

    df = df[x_cols + [y_col]]
    df.columns = x_cols + ['class_idx']
    class_mapping = datasets_data[dataset_name]['class_mapping']

    if normalize_x:
        df[x_cols] = StandardScaler().fit_transform(df[x_cols].values)

    return df, class_mapping, x_cols
