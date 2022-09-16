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
from ufal.udpipe import Model, Pipeline, ProcessingError
from allennlp.predictors.predictor import Predictor


DEFAULT_UDPIPE_MODEL = "models/english-ewt-ud-2.5-191206.udpipe"
DEFAULT_SRL_MODEL = "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def parse_dependencies(pipeline: Pipeline, sentence: str) -> list[conllu.TokenList]:
    error = ProcessingError()
    processed = pipeline.process(sentence, error)
    return conllu.parse(processed)


def iter_token_tree(tree: conllu.TokenTree, ignore_ids: list[int]=None) -> None:
    nodes = [tree]
    while nodes:
        tree = nodes.pop()
        if ignore_ids and tree.token["id"] in ignore_ids:
            continue
        nodes.extend(reversed(tree.children))
        yield tree


def flatten_tree(tree: conllu.TokenTree, ignore_ids: list[int]=None) -> conllu.TokenList:
    tokens = [node.token for node in iter_token_tree(tree, ignore_ids)]
    tokens = sorted(tokens, key=lambda t: t['id'])
    return conllu.TokenList(tokens, tree.metadata)


def extract_clauses_single(srl: Predictor, dep: Model, sentence: str) -> list[str]:
    """
    Extracts sentences from a single sentence.
    """
    labels = srl.predict(sentence)
    words = labels["words"].copy()
    split_indices = []
    for verb in labels["verbs"]:
        last_arg_num = -1
        last_arg_indices = []

        # Wh Handling
        for i, (word, tag) in enumerate(zip(labels["words"], verb["tags"])):
            if "R-ARG" in tag:
                words[i] = " ".join(words[j] for j in last_arg_indices)
                split_indices.append(verb["tags"].index("B-V"))
            if re.search(r"ARG\d+", tag):
                num = int(re.search(r"\d+", tag).group())
                if num == last_arg_num:
                    last_arg_indices.append(i)
                else:
                    last_arg_num = num
                    last_arg_indices = [i]

        # Conjunction Handling
        for i, (word, tag) in enumerate(zip(labels["words"], verb["tags"])):
            if word == "and":
                if "ARG" in tag:
                    split_indices.append(i)
                elif verb["tags"][i + 1] == "B-V":
                    words[i] = " ".join(words[j] for j in last_arg_indices)
                    split_indices.append(i + 1)
            if re.search(r"ARG\d+", tag):
                num = int(re.search(r"\d+", tag).group())
                if num == last_arg_num:
                    last_arg_indices.append(i)
                else:
                    last_arg_num = num
                    last_arg_indices = [i]

    # Insertion Handling
    pipeline = Pipeline(udpipe_model, "horizontal", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
    token_list = parse_dependencies(pipeline, " ".join(labels["words"]))[0]
    tree = token_list.to_tree()
    # Make sure the udpipe's tokenisation matches the srl's one
    assert len(token_list) == len(words)

    clauses = []
    subj = []
    for node in iter_token_tree(tree):
        token = node.token
        # print(token["deprel"], token["form"], token["id"], sep="\t")
        if "subj" in token["deprel"]:
            subj = [words[token["id"] - 1] for token in flatten_tree(node)]
        
        if (token["id"] - 1) in split_indices:
            clauses.append([words[token["id"] - 1] for token in flatten_tree(node)])
        elif token["deprel"] in ["appos", "advcl", "acl", "acl:relcl"]:
            clauses.append(subj + [words[token["id"] - 1] for token in flatten_tree(node)])
            split_indices.append(token["id"] - 1)

    clauses.append([words[token["id"] - 1] for token in flatten_tree(tree, [i + 1 for i in split_indices])])
        
    def clause_filter(clause: list[str]) -> str:
        return " ".join(word for word in clause if word != ",")

    return [clause_filter(clause) for clause in clauses]


def extract_clauses(srl: Predictor, dep: Model, example: str) -> list[str]:
    """
    Extracts clauses from the given example. The example may consist of multiple sentences. Each sentence is processed
    separately and the clauses are merged together.
    """
    pipeline = Pipeline(udpipe_model, "tokenize", Pipeline.NONE, Pipeline.NONE, "conllu")
    token_lists = parse_dependencies(pipeline, example)

    clauses = []
    for token_list in token_lists:
        sentence = " ".join(token["form"] for token in token_list)
        clauses.extend(extract_clauses_single(srl, dep, sentence))

    return clauses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--udpipe_model_path", 
                        type=str, 
                        default=DEFAULT_UDPIPE_MODEL, 
                        help="Path to the UDPipe model.")
    parser.add_argument("--srl_model_path", 
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

    udpipe_model = Model.load(args.udpipe_model_path)
    
    srl_predictor = Predictor.from_path(args.srl_model_path)

    with open(args.input) as f:
        examples = [line.rstrip() for line in f]

    with open(args.output, "w") as f:
        for example in examples:
            logger.info("Original: %s", example)
            clauses = extract_clauses(srl_predictor, udpipe_model, example)
            logger.info("Split:    %s", " \033[31m|\033[0m ".join(clauses))
            print(args.sep.join(clauses), file=f)

