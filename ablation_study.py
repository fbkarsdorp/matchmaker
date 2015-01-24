import numpy as np

from gensim.models import Word2Vec
from sklearn.datasets import load_svmlight_file

from pysofia import SofiaML
from learning_rank_lovers import load_lovers, extract_features, create_candidates
from learning_rank_lovers import powerset, create_folds


par_model = Word2Vec.load_word2vec_format('plays-paragraphs-dbow.bin', binary=True)
par_model.init_sims()
word_model = Word2Vec.load_word2vec_format("play-paragraphs-coref-sg.bin", binary=True)
word_model.init_sims()

lover_pairs = load_lovers("lovers.txt")
features = extract_features("../Dropbox/plays/plays")
characters = set(w for w in par_model.vocab if '_' in w and w in features)
candidate_pairs = create_candidates(characters, lover_pairs)

feature_names = np.array(['par_sim', 'par_disp', 'word_sim', 'word_disp', 'sex', 'interaction', 'scene_cooccurrence'])
for f_select in list(powerset(range(7)))[::-1]:
    if f_select:
        MRR, MRR_1 = [], []
        f_select = list(f_select)
        if len(f_select) > 1:
            continue
#        if not (0 in f_select or 1 in f_select):
#            continue
        # experiment with scaling (best score so far...)
        for i, (X_train, X_test, y_train, y_test, Q_train, Q_test) in enumerate(create_folds(
            lover_pairs, candidate_pairs, par_model, word_model, features, f_select, normalizer='minmax')):
            foldname = 'fold-%s-%s' % (i, '-'.join(feature_names[f_select]))
            ranker = SofiaML(name=foldname, alpha=0.3, learner='pegasos', model='query-norm-rank', 
                n_features=8, max_iter=100000)
            ranker.fit(X_train, y_train, Q_train)
            ranking = ranker.predict(X_test, y_test)
            fold_mrr = []
            fold_mrr_1 = []
            for _ in range(10):
                X_train = foldname + '-train.svm'
                X_test = foldname + '-test.svm'
                ranker.fit(X_train)
                ranking = ranker.predict(X_test)
                _, y_test = load_svmlight_file(X_test)
                fold_mrr.append(1. / (ranking.tolist().index(y_test.tolist().index(1)) + 1))
                fold_mrr_1.append(0. if fold_mrr[-1] != 1 else 1.)
            MRR.append(np.array(fold_mrr).mean())
            MRR_1.append(np.array(fold_mrr_1).mean())
        print '=' * 80
        print feature_names[f_select]
        print 'MRR', np.array(MRR).mean()
        print 'MRR@1', np.array(MRR_1).mean()
        print '=' * 80
        print 
        