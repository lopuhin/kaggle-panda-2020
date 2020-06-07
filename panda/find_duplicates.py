from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import pandas as pd
import turbojpeg
import tqdm


def main():
    def id_by_path(path):
        return path.name.split('_')[0]
        
    paths = list(sorted(Path('data/train_images/').glob('*_2.jpeg')))
    path_by_id = {id_by_path(p): p for p in paths}
    index_by_id = {id_by_path(p): i for i, p in enumerate(paths)}

    jpeg = turbojpeg.TurboJPEG()
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def read_image(path):
        image = jpeg.decode(path.read_bytes())
        ratio = 1024 / max(image.shape[:2])
        if ratio < 1:
            image = cv2.resize(
                image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        return image

    def descriptor_worker(path):
        image = read_image(path)
        return orb.detectAndCompute(image, None)

    with ThreadPool(processes=16) as pool:
        all_descriptors = list(tqdm.tqdm(
            pool.imap(descriptor_worker, paths, chunksize=10),
            total=len(paths),
            desc='descriptors'))

    df_train = pd.read_csv('data/train.csv')
    all_pairs = []
    for data_provider in ['radboud']:
        for grade in range(6):
            sim_paths = [path_by_id[image_id] for image_id in df_train.query(
                f'data_provider == "{data_provider}" and '
                f'isup_grade == {grade}')['image_id']]
            print(f'provider {data_provider} grade {grade}: '
                  f'{len(sim_paths)} paths')

            def match_worker(p1p2):
                p1, p2 = p1p2
                (kp1, d1), (kp2, d2) = [
                    all_descriptors[index_by_id[id_by_path(p)]]
                    for p in [p1, p2]]
                try:
                    matches = sorted(bf.match(d1, d2), key=lambda x: x.distance)
                except cv2.error as e:
                    print(e)
                    matches = []
                return p1, p2, kp1, kp2, matches

            pairs_to_match = [
                (p1, p2) for p1 in sim_paths for p2 in sim_paths if p1 < p2]

            all_matches = list(tqdm.tqdm(
                map(match_worker, pairs_to_match),
                total=len(pairs_to_match),
                desc='matching'))

            good_matches = [x for x in all_matches
                            if sum(x.distance < 25 for x in x[-1]) > 10]
            print(f'found {len(good_matches)} matches')
            all_pairs.extend(
                (p1, p2) for p1, p2, _, _, _ in good_matches)

    pairs_df = pd.DataFrame([
        {'a': id_by_path(p1), 'b': id_by_path(p2)}
        for p1, p2 in all_pairs])
    pairs_df.to_csv('orb_duplicates.csv')


if __name__ == '__main__':
    main()
