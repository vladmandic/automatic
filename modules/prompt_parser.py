# pylint: disable=anomalous-backslash-in-string

"""
import os
import sys
from rich import print
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
"""

import os
import re
from collections import namedtuple
from typing import List
import lark
import torch
from compel import Compel
from modules.shared import opts, log, native

# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][ in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25, 'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50, 'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60, 'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75, 'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']

round_bracket_multiplier = 1.1
square_bracket_multiplier = 1.0 / 1.1
re_AND = re.compile(r"\bAND\b")
# re_weight = re.compile(r"^(.*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")
re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")
ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])
schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER "]"
alternate: "[" prompt ("|" prompt)+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")
re_clean = re.compile(r"^\W+", re.S)
re_whitespace = re.compile(r"\s+", re.S)
re_break = re.compile(r"\s*\bBREAK\b|##\s*", re.S)
re_attention_v2 = re.compile(r"""
\(|\[|\\\(|\\\[|\\|\\\\|
:([+-]?[.\d]+)|
\)|\]|\\\)|\\\]|
[^\(\)\[\]:]+|
:
""", re.X)
re_attention_v1 = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)


debug_output = os.environ.get('SD_PROMPT_DEBUG', None)
debug = log.trace if debug_output is not None else lambda *args, **kwargs: None
debug('Trace: PROMPT')


def get_learned_conditioning_prompt_schedules(prompts, steps):
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    >>> g("[a|(b:1.1)]")
    [[1, 'a'], [2, '(b:1.1)'], [3, 'a'], [4, '(b:1.1)'], [5, 'a'], [6, '(b:1.1)'], [7, 'a'], [8, '(b:1.1)'], [9, 'a'], [10, '(b:1.1)']]
    """

    def collect_steps(steps, tree):
        res = [steps]
        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                tree.children[-1] = float(tree.children[-1])
                if tree.children[-1] < 1:
                    tree.children[-1] *= steps
                tree.children[-1] = min(steps, int(tree.children[-1]))
                res.append(tree.children[-1])
            def alternate(self, tree): # pylint: disable=unused-argument
                res.extend(range(1, steps+1))
        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, _, when = args
                try:
                    yield before or () if step <= when else after
                except StopIteration:
                    yield ''
            def alternate(self, args):
                try:
                    yield next(args[(step - 1) % len(args)]) # pylint: disable=stop-iteration-return
                except StopIteration:
                    yield ''
            def start(self, args):
                def flatten(x):
                    if type(x) == str:
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)
                return ''.join(flatten(args))
            def plain(self, args):
                yield args[0].value
            def __default__(self, data, children, meta):
                yield from children
        return AtStep().transform(tree)

    def get_schedule(prompt):
        try:
            tree = schedule_parser.parse(prompt)
        except Exception:
            return [[steps, prompt]]
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]


def get_learned_conditioning(model, prompts, steps):
    """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),
    and the sampling step at which this condition is to be replaced by the next one.
    Input:
        (model, ['a red crown', 'a [blue:green:5] jeweled crown'], 20)
    Output:
    [
        [ ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0523,  ..., -0.4901, -0.3066,  0.0674], ..., [ 0.3317, -0.5102, -0.4066,  ...,  0.4119, -0.7647, -1.0160]], device='cuda:0')) ],
        [ ScheduledPromptConditioning(end_at_step=5, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.0192,  0.3867, -0.4644,  ...,  0.1135, -0.3696, -0.4625]], device='cuda:0')),
          ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.7352, -0.4356, -0.7888,  ...,  0.6994, -0.4312, -1.2593]], device='cuda:0')),
        ]
    ]
    """
    res = []
    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps)
    cache = {}
    for prompt, prompt_schedule in zip(prompts, prompt_schedules):
        debug(f'Prompt schedule: {prompt_schedule}')
        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue
        texts = [x[1] for x in prompt_schedule]
        conds = model.get_learned_conditioning(texts)
        cond_schedule = []
        for i, (end_at_step, _text) in enumerate(prompt_schedule):
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, conds[i]))
        cache[prompt] = cond_schedule
        res.append(cond_schedule)
    return res


def get_multicond_prompt_list(prompts):
    res_indexes = []
    prompt_flat_list = []
    prompt_indexes = {}
    for prompt in prompts:
        subprompts = re_AND.split(prompt)
        indexes = []
        for subprompt in subprompts:
            match = re_weight.search(subprompt)
            text, weight = match.groups() if match is not None else (subprompt, 1.0)
            weight = float(weight) if weight is not None else 1.0
            index = prompt_indexes.get(text, None)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index
            indexes.append((index, weight))
        res_indexes.append(indexes)
    return res_indexes, prompt_flat_list, prompt_indexes


class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0):
        self.schedules: List[ScheduledPromptConditioning] = schedules
        self.weight: float = weight


class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape  # the shape field is needed to send this object to DDIM/PLMS
        self.batch: List[List[ComposableScheduledPromptConditioning]] = batch


def get_multicond_learned_conditioning(model, prompts, steps) -> MulticondLearnedConditioning:
    """same as get_learned_conditioning, but returns a list of ScheduledPromptConditioning along with the weight objects for each prompt.
    For each prompt, the list is obtained by splitting the prompt using the AND separator.
    https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
    """
    res_indexes, prompt_flat_list, _prompt_indexes = get_multicond_prompt_list(prompts)
    learned_conditioning = get_learned_conditioning(model, prompt_flat_list, steps)
    res = []
    for indexes in res_indexes:
        res.append([ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])
    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res)


def reconstruct_cond_batch(c: List[List[ScheduledPromptConditioning]], current_step):
    param = c[0][0].cond
    res = torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype)
    for i, cond_schedule in enumerate(c):
        target_index = 0
        for current, (end_at, _cond) in enumerate(cond_schedule):
            if current_step <= end_at:
                target_index = current
                break
        res[i] = cond_schedule[target_index].cond
    return res


def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step):
    param = c.batch[0][0].schedules[0].cond
    tensors = []
    conds_list = []
    for composable_prompts in c.batch:
        conds_for_batch = []
        for composable_prompt in composable_prompts:
            target_index = 0
            for current, entry in enumerate(composable_prompt.schedules):
                if current_step <= entry.end_at_step:
                    target_index = current
                    break
            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(composable_prompt.schedules[target_index].cond)
        conds_list.append(conds_for_batch)
    # if prompts have wildly different lengths above the limit we'll get tensors fo different shapes and won't be able to torch.stack them. So this fixes that.
    token_count = max([x.shape[0] for x in tensors])
    for i in range(len(tensors)):
        if tensors[i].shape[0] != token_count:
            last_vector = tensors[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - tensors[i].shape[0], 1])
            tensors[i] = torch.vstack([tensors[i], last_vector_repeated])
    return conds_list, torch.stack(tensors).to(device=param.device, dtype=param.dtype)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      ( - literal character '('
      [ - literal character '['
      ) - literal character ')'
      ] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('(literal]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """
    res = []
    round_brackets = []
    square_brackets = []
    if opts.prompt_attention == 'fixed':
        res = [[text, 1.0]]
        debug(f'Prompt: parser="{opts.prompt_attention}" {res}')
        return res
    elif opts.prompt_attention == 'compel':
        conjunction = Compel.parse_prompt_string(text)
        if conjunction is None or conjunction.prompts is None or conjunction.prompts is None or len(conjunction.prompts[0].children) == 0:
            return [["", 1.0]]
        res = []
        for frag in conjunction.prompts[0].children:
            res.append([frag.text, frag.weight])
        debug(f'Prompt: parser="{opts.prompt_attention}" {res}')
        return res
    elif opts.prompt_attention == 'a1111':
        re_attention = re_attention_v1
        whitespace = ''
    else:
        re_attention = re_attention_v1
        if native:
            text = text.replace('\n', ' BREAK ')
        else:
            text = text.replace('\n', ' ')
        whitespace = ' '

    def multiply_range(start_position, multiplier):
        try:
            for p in range(start_position, len(res)):
                res[p][1] *= multiplier
        except Exception as e:
            log(f'Prompt parser: {e}')

    for m in re_attention.finditer(text):
        try:
            section = m.group(0)
            weight = m.group(1)
            if section.startswith('\\'):
                res.append([section[1:], 1.0])
            elif section == '(':
                round_brackets.append(len(res))
            elif section == '[':
                square_brackets.append(len(res))
            elif weight is not None and len(round_brackets) > 0:
                multiply_range(round_brackets.pop(), float(weight))
            elif section == ')' and len(round_brackets) > 0:
                multiply_range(round_brackets.pop(), round_bracket_multiplier)
            elif section == ']' and len(square_brackets) > 0:
                multiply_range(square_brackets.pop(), square_bracket_multiplier)
            else:
                parts = re.split(re_break, section)
                for i, part in enumerate(parts):
                    if i > 0:
                        res.append(["BREAK", -1])
                    if opts.prompt_attention == 'native':
                        part = re_clean.sub("", part)
                        part = re_whitespace.sub(" ", part).strip()
                        if len(part) == 0:
                            continue
                    res.append([part, 1.0])
        except Exception as e:
            log.error(f'Prompt parser: section="{text[m.start():m.end()]}" position={m.start()}:{m.end()} text="{text}" error={e}')
    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)
    if len(res) == 0:
        res = [["", 1.0]]
    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += whitespace + res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    debug(f'Prompt: parser="{opts.prompt_attention}" {res}')
    return res

if __name__ == "__main__":
    input_text = '[black] [[grey]] (white) ((gray)) ((orange:1.1) yellow) ((purple) and [dark] red:1.1) [mouse:0.2] [(cat:1.1):0.5]'
    log.info(f'Prompt: {input_text}')
    all_schedules = get_learned_conditioning_prompt_schedules([input_text], 100)[0]
    log.info(f'Schedules: {all_schedules}')
    for schedule in all_schedules:
        log.info(f'Schedule: {schedule[0]}')
        opts.data['prompt_attention'] = 'fixed'
        output_list = parse_prompt_attention(schedule[1])
        log.info(f'  Fixed: {output_list}')
        opts.data['prompt_attention'] = 'compel'
        output_list = parse_prompt_attention(schedule[1])
        log.info(f'  Compel: {output_list}')
        opts.data['prompt_attention'] = 'a1111'
        output_list = parse_prompt_attention(schedule[1])
        log.info(f'  A1111: {output_list}')
        opts.data['prompt_attention'] = 'native'
        log.info = parse_prompt_attention(schedule[1])
        log.info(f'  Full:  {output_list}')
