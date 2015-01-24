import codecs
import glob
import os
from itertools import product, combinations, chain

import numpy as np
from gensim.models import Word2Vec
from gensim.matutils import unitvec
from extract_features import parse_play
from pysofia import SofiaML

def extract_features(play_directory):
    """Return for each unique actor the scene co-occurrence, 
    interaction frequency and gender feature."""
    actors = {actor: features for fname in glob.glob(os.path.join(play_directory, "*.xml"))
                              for actor, features in parse_play(fname).iteritems()}
    return actors

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def load_lovers(lover_file):
    return list(set(tuple(pair.strip().split(' : ')[2:]) for pair in codecs.open(lover_file, encoding='utf-8')))

def create_candidates(characters, lover_pairs):
    return list(set((a, b) for a, b in product(characters, characters)
                    if a.endswith(b[-4:]) and a != b and 
                       not ((a, b) in lover_pairs or (b, a) in lover_pairs)))

def displacement_relation(a, b, c, d, model):
    pair_a = unitvec(np.array([ 1 * model.syn0norm[model.vocab[a].index], 
                               -1 * model.syn0norm[model.vocab[b].index]]).mean(axis=0))
    pair_b = unitvec(np.array([ 1 * model.syn0norm[model.vocab[c].index],
                               -1 * model.syn0norm[model.vocab[d].index]]).mean(axis=0))
    return np.dot(pair_a, pair_b)

def max_displacement_sim(a, b, lovers, model):
    return max(displacement_relation(a, b, c, d, model) for (c, d) in lovers
               if not (a in (c, d) or b in (c, d)))

def create_folds(lover_pairs, candidate_pairs, par_model, word_model, features, f_select=None):
    loving_persons = list(set(pair[0] for pair in lover_pairs))
    for loving_person in loving_persons:
        X_train, X_test, y_train, y_test, Q_train, Q_test = [], [], [], [], [], []
        for (a, b) in lover_pairs + candidate_pairs:
            if a not in loving_persons:
                continue
            if loving_person in (a, b):
                X_append, y_append, Q_append = X_test.append, y_test.append, Q_test.append
            else:
                X_append, y_append, Q_append = X_train.append, y_train.append, Q_train.append
            Q_append(loving_persons.index(a) + 1)
            y_append(1 if (a, b) in lover_pairs else 0)
            interaction = features[a].interaction.edge[a][b]['weight'] if b in features[a].interaction.edge[a] else 0.0
            scene_cooccurrence = features[a].neighbors.edge[a][b]['weight'] if b in features[a].neighbors.edge[a] else 0.0
            sex = 0.0 if (features[a].sex == features[b].sex or features[b].sex is None) else 1
            par_sim = par_model.similarity(a, b)
            par_disp = max_displacement_sim(a, b, lover_pairs, par_model)
            if a in word_model.vocab and b in word_model.vocab:
                word_sim = word_model.similarity(a, b)
                word_disp = max_displacement_sim(a, b, lover_pairs, word_model)
            else:
                word_sim = 0.0
                word_disp = 0.0
            X_append((par_sim, par_disp, word_sim, word_disp, sex, interaction, scene_cooccurrence))
        if f_select is None:
            f_select = range(len(X_train[0]))
        X_train, X_test = np.array(X_train)[:,f_select], np.array(X_test)[:,f_select]
        y_train, y_test = np.array(y_train), np.array(y_test)
        Q_train, Q_test = np.array(Q_train), np.array(Q_test)
        train_positions = Q_train.argsort()
        X_train, y_train, Q_train = X_train[train_positions], y_train[train_positions], Q_train[train_positions]
        test_positions = Q_test.argsort()
        X_test, y_test, Q_test = X_test[test_positions], y_test[test_positions], Q_test[test_positions]
        yield X_train, X_test, y_train, y_test, Q_train, Q_test


if __name__ == '__main__':
    par_model = Word2Vec.load_word2vec_format('plays-paragraphs.bin', binary=True)
    par_model.init_sims()
    word_model = Word2Vec.load_word2vec_format("play-paragraphs-coref-sg.bin", binary=True)
    word_model.init_sims()
    
    lover_pairs = load_lovers("lovers.txt")
    features = extract_features("../Dropbox/plays/plays")
    characters = set(w for w in par_model.vocab if '_' in w and w in features)
    candidate_pairs = create_candidates(characters, lover_pairs)
    MRR, MRR_1 = [], []
    for i, (X_train, X_test, y_train, y_test, Q_train, Q_test) in enumerate(create_folds(
        lover_pairs, candidate_pairs, par_model, word_model, features)):
        ranker = SofiaML(name='fold-%s' % i, alpha=0.3, learner='pegasos', model='rank', 
            n_features=X_train.shape[1]+1, max_iter=1000)
        ranker.fit(X_train, y_train, Q_train)
        ranking = ranker.predict(X_test, y_test)
        MRR.append(1. / (ranking.tolist().index(y_test.tolist().index(1)) + 1))
        MRR_1.append(0. if MRR[-1] != 1 else 1.)
    print 'MRR', np.array(MRR).mean()
    print 'MRR@1', np.array(MRR_1).mean()
        
