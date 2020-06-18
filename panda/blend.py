import argparse
from pathlib import Path
import json

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+')
    args = parser.parse_args()
    inputs = [json.loads(Path(p).read_text()) for p in args.inputs]
    image_ids = inputs[0]['image_ids']
    for x in inputs:
        assert x['image_ids'] == image_ids
    bins = np.mean([x['bins'] for x in inputs], 0)
    predictions = np.mean([x['predictions'] for x in inputs], 0)
    predictions = np.digitize(predictions, bins)
    submission = pd.DataFrame({
        'image_id': image_ids, 'isup_grade': predictions})
    submission.to_csv('submission.csv', index=None)


if __name__ == '__main__':
    main()
