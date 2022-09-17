#!/usr/bin/env python3

# Copyright 2022 Peter Grajcar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import visualisation
import pathlib
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from collections import defaultdict
from typing import Generator, Tuple, Callable
import snowballstemmer


stemmer = snowballstemmer.stemmer('english');


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


STOP_WORDS = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz",]


class Index:
    @dataclass
    class Item:
        index: int
        frequency: int
        last_occurence: int
        document_frequency: int

    def __init__(self):
        self._index = dict()
        self._counter = 0

    def add(self, term: str, doc_idx: int):
        if term in self._index:
            self._index[term].frequency += 1
            if self._index[term].last_occurence != doc_idx:
                self._index[term].last_occurence = doc_idx
                self._index[term].document_frequency += 1
            return
        self._index[term] = Index.Item(self._counter, 0, doc_idx, 1)
        self._counter += 1

    def __len__(self):
        return self._counter

    def __getitem__(self, term: str):
        return self._index[term].index

    def get_document_frequency(self, term: str):
        try:
            return self._index[term].document_frequency
        except KeyError:
            return 0

    def get_term_frequency(self, term: str):
        try:
            return self._index[term].frequency
        except KeyError:
            return 0


WeightFunction = Callable[[int, int], float]


def compute_weight_and_index(tokens: list[str], index: Index, 
                             weight_function: WeightFunction) -> Generator[Tuple[float, int], None, None]:
    term_freq = defaultdict(int)
    term_count = 0
    for token in tokens:
        term_freq[token] += 1
    
    for term, freq in term_freq.items():
        try:
            yield weight_function(freq, index.get_document_frequency(term)), index[term]
        except KeyError:
            yield 0, -1


def construct_matrix(clauses: list[str], index: Index, weight_function: WeightFunction) -> np.ndarray:
    matrix = np.zeros(shape=[len(clauses), len(index)])

    for i, clause in enumerate(clauses):
        terms = stemmer.stemWords(clause.lower().split())
        for weight, j in compute_weight_and_index(terms, index, weight_function):
            if j == -1:
                continue
            matrix[i, j] = weight

    return matrix


def align_clauses(in_clauses: list[str], ref_clauses: list[str]) -> list[int]:
    index = Index()
    for i, clause in enumerate(ref_clauses):
        for word in stemmer.stemWords(clause.lower().split()):
            if word in STOP_WORDS:
                continue
            index.add(word, i)

    def tfidf(tf, df): 
        if tf == 0 or df == 0:
            return 0
        else:
            return (1 + np.log10(tf)) * (np.log10(len(ref_clauses) / df))

    in_matrix = construct_matrix(in_clauses, index, tfidf)
    ref_matrix = construct_matrix(ref_clauses, index, tfidf)
    sim_matrix = cosine_similarity(in_matrix, ref_matrix).T

    logger.info("Similarity matrix: \n%s", sim_matrix)
    logger.info("Max similarity: \n%s", np.max(sim_matrix, axis=0))
    
    # return list(np.where(np.max(sim_matrix, axis=0) < 0.1, -1, np.argmax(sim_matrix, axis=0)))
    return list(np.argmax(sim_matrix, axis=0))


def export_diagram(file_name: str, in_clauses: list[str], ref_clauses: list[str], alignment: list[int]) -> None:
    diagram = visualisation.draw_alignment(in_clauses, ref_clauses, alignment)
    with open(file_name, "w") as f:
        diagram.asSvg(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_clauses", type=str, required=True)
    parser.add_argument("--ref_clauses", type=str, required=True)
    parser.add_argument("--in_sentences", type=str, required=False)
    parser.add_argument("--ref_sentences", type=str, required=False)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--export_diagrams", type=str, default=None)
    parser.add_argument("--sep", type=str, default="\t")
    args = parser.parse_args()

    if args.export_diagrams:
        export_path = pathlib.Path(args.export_diagrams)
        export_path.mkdir(exist_ok=True)
        svg_path = (export_path / "svg")
        svg_path.mkdir(exist_ok=True)

    with open(args.in_clauses) as in_file, open(args.ref_clauses) as ref_file, open(args.output, "w") as out_file:
        for i, (in_line, ref_line) in enumerate(zip(in_file, ref_file)):
            in_clauses = in_line.rstrip("\n").split(args.sep)
            ref_clauses = ref_line.rstrip("\n").split(args.sep)

            logging.info("Input clauses: %s", in_clauses)
            logging.info("Reference clauses: %s", ref_clauses)

            alignment = align_clauses(in_clauses, ref_clauses)

            logging.info("Alignment: %s", alignment)

            print(" ".join(str(i) for i in alignment), file=out_file)

            if args.export_diagrams:
                export_diagram(str(svg_path / f"alignment_{i:03d}.svg"), in_clauses, ref_clauses, alignment)

        total = i

    if args.export_diagrams:
        with (export_path / "index.html").open("w") as index:
            print("<html>", file=index)
            print("  <body>", file=index)
            if args.in_sentences:
                in_file = open(args.in_sentences)
            if args.ref_sentences:
                ref_file = open(args.in_sentences)
            for i in range(total):
                if args.in_sentences:
                    print(f"    Original sentences: <i>", in_file.readline(), "</i>", file=index)
                print(f"    <img src=\"svg/alignment_{i:03d}.svg\" />", file=index)
                if args.ref_sentences:
                    print(f"    Reference sentences: <i>", ref_file.readline(), "</i>", file=index)
                print("    <hr />", file=index)
            print("  </body>", file=index)
            print("</html>", file=index)
                

