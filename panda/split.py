import pandas as pd
from sklearn.model_selection import GroupKFold


def make_split(n_folds: int = 5):
    df = pd.read_csv('data/train.csv')
    pen_markings = pd.read_csv('pen_markings.csv')
    df = df[~df['image_id'].isin(pen_markings['image_id'])]
    groups = pd.read_csv('groups.csv')
    image_groups = dict(zip(groups['image_id'], groups['group']))
    df['group'] = df['image_id'].apply(lambda x: str(image_groups.get(x, x)))
    df = df.sample(frac=1, random_state=42).reset_index()
    kfold = GroupKFold(n_folds)
    split = df[['image_id']].copy()
    split['fold'] = -1
    split.set_index('image_id')
    for i, (_, valid_ids) in enumerate(kfold.split(df, groups=df['group'])):
        split.loc[valid_ids, 'fold'] = i
    assert set(split['fold']) == set(range(n_folds))
    split.to_csv('split.csv', index=None)

    for fold in range(n_folds):
        fold_df = df[df['image_id'].isin(split[split['fold'] == fold]['image_id'])]
        print(
            fold,
            len(fold_df),
            dict(fold_df['data_provider'].value_counts()),
            dict(fold_df['isup_grade'].value_counts()),
        )


if __name__ == '__main__':
    make_split()
