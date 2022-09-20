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

import drawSvg as draw
from itertools import cycle

COLOURS = ["#dc322f", "#859900", "#268bd2", "#b58900", "#6c71c4", "#cb4b16", "#d33682"]

def draw_alignment(in_clauses: list[str], ref_clauses: list[str], alignment: list[int]) -> draw.Drawing:
    d = draw.Drawing(2000, 140, origin=(0, -140))

    arrow_head = draw.Marker(-1, -1, 1, 1, scale=4)
    path = draw.Path(d="M-0.5,-0.5 V0.5 L0.5,0 Z")
    arrow_head.append(path)
    
    text = draw.Text("", 12, 10, 20)
    in_clause_spans = []
    for clause, colour in zip(in_clauses, cycle(COLOURS)):
        span = draw.TSpan(clause, y=20, dx="1em", fill=colour)
        in_clause_spans.append(span)
        text.append(span)

    d.append(text)

    text = draw.Text("", 12, 10, 130)
    ref_clause_spans = []
    for clause, colour in zip(ref_clauses, cycle(COLOURS)):
        span = draw.TSpan(clause, y=130, dx="1em", fill=colour)
        ref_clause_spans.append(span)
        text.append(span)

    d.append(text)

    for (i, j), colour in zip(enumerate(alignment), cycle(COLOURS)):
        if j == -1:
            continue
        x1 = 10 + sum(len(clause) for clause in in_clauses[:i]) * 5.25 + len(in_clauses[i]) * 5.25 / 2
        x2 = 10 + sum(len(clause) for clause in ref_clauses[:j]) * 5.25 + len(ref_clauses[j]) * 5.25 / 2
        d.append(draw.Path(f"M{x1},25 {x2},115", marker_end=arrow_head, stroke=colour))

    return d

if __name__ == "__main__":
    in_clauses = [
        "Aaron Turner plays electric guitar .",
        "Aaron Turner played with the band Twilight",
        "Aaron Turner Twilight",
        "Twilight performs black metal music .",
        "black metal music when part of a musical fusion is called death metal",
        "The associated band / associated musical artist of Aaron Turner is Old Man Gloom .",
    ]

    ref_clauses = [
        "NULL",
        "Aaron Turner is an electric guitar player .",
        "an electric guitar player has played with the black metal @ band Twilight and with Old Man Gloom",
        "Death metal is a musical fusion of black metal .",
    ]

    d = draw_alignment(in_clauses, ref_clauses, [1, 2, 0, 0, 3, 2])
    with open("example.svg", "w") as f:
        d.asSvg(f)

