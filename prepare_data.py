import codecs
import glob
import random
import re
import os
import sys

from collections import defaultdict
from itertools import islice

from lxml import etree
from pattern.fr import tokenize as tokenizer

I = re.compile(r"\b([jJ][e']|[mM][oae'][ni]?)(\b|\s+)", re.UNICODE)
YOU = re.compile(r"\b(tu|toi|ton|vous|votre|vos|ta|tes)\b", re.UNICODE)

import unicodedata
def strip_accents(s):
    if not isinstance(s, unicode):
        s = unicode(s)
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def standardize_name(name, fileid):
    return '_'.join((name + "_" + fileid[2:]).replace('-', ' ').split()).upper()

def parse_play(filename, concat=False, paragraphs=True, replace_names=False,
               tokenize=True, lowercase=True):
    xml = etree.parse(filename)
    fileid = filename.split('/')[-1].replace('.xml', '')
    author = xml.find('//author').text
    speaker_lines = defaultdict(list)
    if lowercase:
        lowercase = lambda s: s.lower()
    else:
        lowercase = lambda s: s
    text_cast = {}
    for c in xml.iterfind('//castItem/role'):
        if c.text:
            text_cast[c.text.capitalize()] = standardize_name(c.attrib['id'], fileid)
        else:
            text_cast[c.attrib['id'].capitalize()] = standardize_name(c.attrib['id'], fileid)
    # names in cast list do not always match with those in the text. fix that!
    text_speakers = set()
    for speaker in xml.iterfind('//sp'):
        text_speakers.add(standardize_name(speaker.attrib['who'], fileid))
    for text_name, speaker in text_cast.iteritems():
        if speaker not in text_speakers:
            # 1 cast_speaker heeft accent, text niet
            if strip_accents(speaker) in text_speakers:
                text_cast[text_name] = strip_accents(speaker)
            # 2 cast speaker geen accent, text wel
            else:
                for candidate in text_speakers:
                    if strip_accents(candidate) == speaker:
                        text_cast[text_name] = candidate
    standardized_names = {name: text_name for text_name, name in text_cast.iteritems()}
    for name in standardized_names:
        if strip_accents(name) in standardized_names:
            standardized_names[name] = strip_accents(name)
        else:
            standardized_names[name] = name
    print standardized_names
    # replace in text names with standardized names                        
    for speaker_turn in xml.iterfind('//sp'):
        if not speaker_turn.attrib['who']:
            # author text?
            speaker = author
        # make each speaker unique
        else:
            if standardize_name(speaker_turn.attrib['who'], fileid) not in standardized_names:
                speaker = standardize_name(speaker_turn.attrib['who'], fileid)
            else:
                speaker = standardized_names[standardize_name(speaker_turn.attrib['who'], fileid)]
        # extract the text of this speaker turn
        sentences = [sent for sent in islice(speaker_turn.itertext(), 1, None)]
        if tokenize and replace_names:
            # replace 1st person forms with speaker name
            sentences = [I.sub(" " + speaker + " ", toksent) for sent in sentences for toksent in tokenizer(sent)]
            for i in range(len(sentences)):
                # replace names with standardized versions
                for text_name in text_cast:
                    sentences[i] = sentences[i].replace(text_name, text_cast[text_name])
        elif tokenize:
            sentences = [toksent for sent in sentences for toksent in tokenizer(sent)]
        # get rid off extra spaces
        sentences = [' '.join(lowercase(sent).split()) for sent in sentences]
        # recapitalize all cast_members
        for i in range(len(sentences)):
            for cast_member in text_cast.values():
                sentences[i] = sentences[i].replace(cast_member.lower(), cast_member)
        if paragraphs:
            sentences = [' '.join(sentences)]
        speaker_lines[speaker] += sentences
    if concat:
        speaker_lines = {speaker: [' '.join(lines)] for speaker, lines in speaker_lines.iteritems()}
    return speaker_lines

def to_labeled_sentences(play):
    for speaker, sentences in play.iteritems():
        for sentence in sentences:
            yield speaker, sentence

if __name__ == '__main__':
    sentences = []
    concat = int(sys.argv[3])
    paragraphs = int(sys.argv[4])
    replace_names = int(sys.argv[5])
    paragraph_vectors = int(sys.argv[6])
    for filename in glob.glob(os.path.join(sys.argv[1], "*.xml")):
        for speaker, sentence in to_labeled_sentences(
            parse_play(filename, concat=concat, paragraphs=paragraphs, 
                       replace_names=replace_names)):
            if paragraph_vectors:
                sentences.append("%s %s" % (speaker, sentence))
            else:
                sentences.append("%s" % sentence)
    random.shuffle(sentences)
    with codecs.open(sys.argv[2], "w", "utf-8") as outfile:
        outfile.write('\n'.join(sentences))
