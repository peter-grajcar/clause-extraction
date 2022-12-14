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
import re
import conllu
import logging
import requests
from ufal.udpipe import Model, Pipeline, ProcessingError
from allennlp.predictors.predictor import Predictor
from itertools import cycle


DEFAULT_UDPIPE_MODEL = "models/english-ewt-ud-2.5-191206.udpipe"
DEFAULT_SRL_MODEL = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
DEFAULT_UDPIPE_2_API_URL = "https://lindat.mff.cuni.cz/services/udpipe/api"


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def tokenise(model: Model, example: str) -> list[conllu.TokenList]:
    """
    Tokenisation is done using UDPipe1.
    """
    pipeline = Pipeline(model, "tokenize", Pipeline.NONE, Pipeline.NONE, "conllu")
    error = ProcessingError()
    processed = pipeline.process(example, error)
    return conllu.parse(processed)

def parse(sentence: str, api_url:str=DEFAULT_UDPIPE_2_API_URL) -> list[conllu.TokenList]:
    """
    UDPipe2 is used for the dependency parsing via the REST API.
    """
    url = f"{api_url}/process"
    params = {
        "model": "english-ewt-ud-2.10-220711",
        "input": "horizontal",
        "tagger": "",
        "parser": "",
        "data": sentence
    }
    response = requests.get(url, params)
    data = response.json()
    return conllu.parse(data["result"])


def iter_token_tree(tree: conllu.TokenTree, ignore_ids: list[int]=None, ignore_deprel: list[str]=None) -> None:
    nodes = [tree]
    while nodes:
        tree = nodes.pop()
        if ignore_ids and tree.token["id"] in ignore_ids:
            continue
        if ignore_deprel and tree.token["deprel"] in ignore_deprel:
            continue
        nodes.extend(reversed(tree.children))
        yield tree


def flatten_tree(tree: conllu.TokenTree, ignore_ids: list[int]=None, ignore_deprel: list[str]=None) -> conllu.TokenList:
    tokens = [node.token for node in iter_token_tree(tree, ignore_ids, ignore_deprel)]
    tokens = sorted(tokens, key=lambda t: t['id'])
    return conllu.TokenList(tokens, tree.metadata)


def extract_clauses_single(srl: Predictor, dep: Model, sentence: str, udpipe_server: str=DEFAULT_UDPIPE_2_API_URL) -> list[str]:
    """
    Extracts sentences from a single sentence.
    """
    labels = srl.predict(sentence)
    words = labels["words"].copy()
    split_indices = []
    for verb in labels["verbs"]:
        last_arg_num = -1
        last_arg_indices = []

        logging.info(verb["description"])
        
        # Wh Handling
        for i, (word, tag) in enumerate(zip(labels["words"], verb["tags"])):
            if "R-ARG" in tag:
                words[i] = " ".join(labels["words"][j] for j in last_arg_indices)
                split_indices.append(i)
            m = re.search(r"ARG(.*)", tag)
            if m:
                num = m.group(1)
                if num == last_arg_num:
                    last_arg_indices.append(i)
                else:
                    last_arg_num = num
                    last_arg_indices = [i]

        # Conjunction Handling
        for i, (word, tag) in enumerate(zip(labels["words"], verb["tags"])):
            if word == "and":
                next_tag = verb["tags"][i + 1]
                if "ARG" not in tag and "ARG" in next_tag:
                    words[i] = ""
                    split_indices.append(i)
                elif next_tag == "B-V":
                    words[i] = " ".join(labels["words"][j] for j in last_arg_indices)
                    split_indices.append(i)
            m = re.search(r"ARG(.*)", tag)
            if m:
                num = m.group(1)
                if num == last_arg_num:
                    last_arg_indices.append(i)
                else:
                    last_arg_num = num
                    last_arg_indices = [i]

    # Insertion Handling
    token_list = parse(" ".join(labels["words"]), api_url=udpipe_server)[0]
    tree = token_list.to_tree()
    # Make sure the udpipe's tokenisation matches the srl's one
    assert len(token_list) == len(words)

    clauses = []
    subj = []
    for node in iter_token_tree(tree):
        token = node.token
        if token["deprel"] == "conj" and any(t.token["form"] == "and" and (t.token["id"] - 1) in split_indices for t in node.children):
            clause = [words[t["id"] - 1] for t in flatten_tree(node)]
            clauses.append((token["id"], clause))
            split_indices.append(token["id"] - 1)

            logging.info("Conjunction split on %s[%s]: %s", token["form"], token["deprel"], " ".join(clause))
        elif any((t.token["id"] - 1) in split_indices for t in node.children):
            clause = [words[t["id"] - 1] for t in flatten_tree(node)]
            clauses.append((token["id"], clause))
            split_indices.append(token["id"] - 1)

            logging.info("Wh split on %s[%s]: %s", token["form"], token["deprel"], " ".join(clause))
        elif token["deprel"] == "appos" and any(t["upostag"] != "PROPN" and t["upostag"] != "PUNCT" for t in flatten_tree(node)):
            clause = subj + [words[t["id"] - 1] for t in flatten_tree(node)]
            clauses.append((token["id"], clause))
            split_indices.append(token["id"] - 1)

            logging.info("Insertion split on %s[%s]: %s", token["form"], token["deprel"], " ".join(clause))
        elif token["deprel"] in ["advcl", "acl", "acl:relcl"]:
            clause = subj + [words[t["id"] - 1] for t in flatten_tree(node)]
            clauses.append((token["id"], clause))
            split_indices.append(token["id"] - 1)

            logging.info("Insertion split on %s[%s]: %s", token["form"], token["deprel"], " ".join(clause))
        
        if "subj" in token["deprel"]:
            subj = [words[t["id"] - 1] for t in flatten_tree(node, ignore_deprel=["acl:relcl"])]
            logger.info("Subject '%s'", " ".join(subj))

    clause = [words[t["id"] - 1] for t in flatten_tree(tree, [i + 1 for i in split_indices])]
    clauses.append((tree.token["id"], clause))
        
    class ClauseFilter:
        def __init__(self):
            self.prev = None

        def __call__(self, word: str) -> bool:
            ret = (self.prev != "," or word != ",") and word.strip()
            self.prev = word
            return ret

    clauses = sorted(clauses, key=lambda t: t[0])
    clauses = [" ".join(filter(ClauseFilter(), clause)) for (id, clause) in clauses]
    return [clause for clause in clauses if clause]


def extract_clauses(srl: Predictor, dep: Model, example: str, udpipe_server: str=DEFAULT_UDPIPE_2_API_URL) -> list[str]:
    """
    Extracts clauses from the given example. The example may consist of multiple sentences. Each sentence is processed
    separately and the clauses are merged together.
    """
    token_lists = tokenise(dep, example)

    clauses = []
    for token_list in token_lists:
        sentence = " ".join(token["form"] for token in token_list)
        clauses.extend(extract_clauses_single(srl, dep, sentence))

    return clauses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--udpipe_model", 
                        type=str, 
                        default=DEFAULT_UDPIPE_MODEL, 
                        help="Path to the UDPipe 1 model (used for tokenisation).")
    parser.add_argument("--udpipe_server", 
                        type=str, 
                        default=DEFAULT_UDPIPE_2_API_URL, 
                        help="URL of the UDPipe 2 server (used for parsing).")
    parser.add_argument("--srl_model", 
                        type=str, 
                        default=DEFAULT_SRL_MODEL, 
                        help="Path to the AllenNLP semantic role labeling model.")
    parser.add_argument("--sep", 
                        type=str, 
                        default="\t", 
                        help="Clause separator used in the output file.")
    parser.add_argument("--input", 
                        type=str, 
                        required=True, 
                        help="Input file.")
    parser.add_argument("--output", 
                        type=str, 
                        required=True, 
                        help="Output file.")
    args = parser.parse_args()

    udpipe_model = Model.load(args.udpipe_model)
    srl_predictor = Predictor.from_path(args.srl_model)

    with open(args.input) as f:
        examples = [line.rstrip() for line in f]

    colours = [31, 32, 33, 34]

    with open(args.output, "w") as f:
        for example in examples:
            logger.info("Original: %s", example)
            clauses = extract_clauses(srl_predictor, udpipe_model, example, udpipe_server=args.udpipe_server)
            logger.info("Splits: %s", 
                        " ".join(f"\033[{colour}m{clause}\033[0m" for clause, colour in zip(clauses, cycle(colours))))
            print(args.sep.join(clauses), file=f)

