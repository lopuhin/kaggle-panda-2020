Train::

    python -m panda.train run-folder

Update code::

    git archive HEAD -o data/panda-2020-src/src.zip
    kaggle datasets version -p data/panda-2020-src -m 'update'


Update model::

    cp run-folder/model.pt run-folder/params.json data/panda-2020-models/
    kaggle datasets version -p data/panda-2020-models/ -m 'update'
