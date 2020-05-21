Training
--------


Build docker image::

    docker build -t panda .

Put data into ``./data`` folder::

    $ ls data
    train.csv  sample_submission.csv  train_images

Prepare jpegs::

    ./docker-run.sh python3 -m panda.to_jpeg

Train::

    ./docker-run.sh panda-train _runs/some-name

Submission
----------

(not all stuff is in git atm)

Update code::

    git archive HEAD -o data/panda-2020-src/src.zip
    kaggle datasets version -p data/panda-2020-src -m 'update'


Update model::

    cp run-folder/model.pt data/panda-2020-models/
    kaggle datasets version -p data/panda-2020-models/ -m 'update'
