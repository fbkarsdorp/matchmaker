from collections import namedtuple
from itertools import product

from lxml import etree

import networkx as nx

import unicodedata
def strip_accents(s):
    if not isinstance(s, unicode):
        s = unicode(s)
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

Actor = namedtuple("Actor", ["name", "sex", "neighbors", "interaction"])

def standardize_name(name, fileid):
    return '_'.join((name + "_" + fileid[2:]).replace('-', ' ').split()).upper()

def ngrams(sequence, n=3):
    count = max(0, len(sequence) - n + 1)
    return (tuple(sequence[i:i+n]) for i in range(count))

def parse_play(filename):
    xml = etree.parse(filename)
    fileid = filename.split('/')[-1].replace('.xml', '')
    cast = {}
    for c in xml.iterfind('//castItem/role'):
        name = standardize_name(c.attrib['id'], fileid)
        civil = c.attrib['civil'] if 'civil' in c.attrib else None
        cast[name] = Actor(name, civil, nx.Graph(), nx.Graph())
    # find cast representations in text
    text_speakers = set()
    for speaker in xml.iterfind('//sp'):
        text_speakers.add(standardize_name(speaker.attrib['who'], fileid))
    for speaker in cast:
        if speaker not in text_speakers:
            # 1 cast_speaker heeft accent, text niet
            if strip_accents(speaker) in text_speakers:
                features = cast[speaker]
                del cast[speaker]
                cast[strip_accents(speaker)] = features
            # 2 cast speaker geen accent, text wel
            else:
                for candidate in text_speakers:
                    if strip_accents(candidate) == speaker:
                        features = cast[speaker]
                        del cast[speaker]
                        cast[candidate] = features
    for scene in xml.iterfind('//div2'):
        scene_cast = set()
        speaker_turns = []
        for speaker_turn in scene.iterfind('sp'):
            if speaker_turn.attrib['who']:
                # make each speaker unique
                speaker = standardize_name(speaker_turn.attrib['who'], fileid)
                scene_cast.add(speaker)
                speaker_turns.append(speaker)
        for a, b in product(scene_cast, scene_cast):
            if a not in cast:
                cast[a] = Actor(a, None, nx.Graph(), nx.Graph())
            if a != b:
                if cast[a].neighbors.has_edge(a, b):
                    cast[a].neighbors.edge[a][b]['weight'] += 1.0
                else:
                    cast[a].neighbors.add_edge(a, b, weight=1.0)
        for l, target, r in ngrams([None] + speaker_turns + [None], 3):
            if target in cast:
                if l is not None:
                    if cast[target].interaction.has_edge(target, l):
                        cast[target].interaction.edge[target][l]['weight'] += 1.0
                    else:
                        cast[target].interaction.add_edge(target, l, weight=1.0)                
    for speaker in cast:
        name, civil, neighbors, interaction = cast[speaker]
        neighbor_sum = sum(c['weight'] for _, _, c in neighbors.edges(data=True))
        interact_sum = sum(c['weight'] for _, _, c in interaction.edges(data=True))
        for a, b in neighbors.edges():
            neighbors.edge[a][b]['weight'] /= neighbor_sum
        for a, b in interaction.edges():
            interaction.edge[a][b]['weight'] /= interact_sum
    return cast
